"""``openagent network`` Click subgroup.

Wired into the top-level ``openagent`` group from ``openagent.cli``.
Commands:

    openagent network init [--name HOMELAB] [--personal]
        Promote this agent to network coordinator. Generates the
        coordinator signing key, writes the singleton ``network`` row,
        and registers this agent in its own ``network_agents`` table.

    openagent network invite [--role user|device|agent] [--bind-to HANDLE]
        Mint a one-shot invite code. Coordinator-only. Prints the code
        in the human-friendly hyphenated form.

    openagent network status
        Print the current network role, name, and node_id. Useful for
        copy-pasting node_id into a peer-add flow.

    openagent network users
        List users registered in this network. Coordinator-only.

    openagent network agents
        List agents registered in this network. Works on member-mode
        too (delegates to the coordinator's ``list_agents`` RPC).

    openagent network revoke --device PUBKEY_HEX
        Mark a device as revoked. Coordinator-only.

The CLI commands open a short-lived MemoryDB connection rather than
piggy-backing on a running gateway: most of these are run once at
provisioning time when the gateway isn't up yet.
"""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from openagent.core import paths as core_paths
from openagent.core.config import load_config
from openagent.memory.db import MemoryDB
from openagent.network.coordinator.store import CoordinatorStore
from openagent.network.identity import (
    Identity,
    load_or_create_identity,
)

console = Console()


def _agent_dir_or_die() -> Path:
    """Return the active agent dir or exit with a clear error."""
    p = core_paths.get_agent_dir()
    if p is None:
        console.print(
            "[red]No agent directory active.[/red] Pass [cyan]--agent-dir <path>[/cyan] "
            "to ``openagent`` so the network commands know which agent to operate on."
        )
        raise click.Abort()
    return p


def _identity_path(agent_dir: Path) -> Path:
    """Return the agent's iroh identity path inside the agent dir.

    A single key serves both as the agent's iroh node identity and (in
    coordinator mode) as the coordinator's cert-signing key. Splitting
    them buys nothing without rotation logic and was the source of a
    NodeId/pubkey mismatch in tickets.
    """
    return agent_dir / "identity.key"


def _db_path_from_config(config: dict) -> str:
    memory_cfg = config.get("memory", {}) or {}
    return memory_cfg.get("db_path") or str(core_paths.default_db_path())


async def _open_db(config: dict) -> MemoryDB:
    db = MemoryDB(_db_path_from_config(config))
    await db.connect()
    return db


# ── ``openagent network`` group ─────────────────────────────────────────


@click.group("network")
@click.pass_context
def network_group(ctx):
    """Manage the OpenAgent network membership of this agent."""
    pass


@network_group.command("init")
@click.option(
    "--name", default=None,
    help="Network name (e.g. ``homelab``). Defaults to ``<agent>-net``.",
)
@click.option(
    "--personal", is_flag=True,
    help="Auto-name the network ``<agent>-personal`` and skip prompts.",
)
@click.pass_context
def cmd_init(ctx, name: str | None, personal: bool):
    """Promote this agent to network coordinator (one-time).

    Generates the coordinator's Ed25519 signing key, writes the
    ``network`` row with role='coordinator', registers the agent
    itself in its own ``network_agents`` registry. After this you
    typically run ``openagent network invite`` to add users / devices.
    """
    asyncio.run(_run_init(ctx, name, personal))


async def auto_init_if_standalone(
    *,
    agent_dir: Path,
    config: dict,
    quiet: bool = False,
) -> dict | None:
    """Promote a standalone agent to its own coordinator network.

    Idempotent: returns the existing ``network`` row when one is already
    present (no init runs). Returns the freshly-created row otherwise.
    Returns ``None`` if init couldn't run for some reason (caller logs
    a warning and continues).

    Used by both ``openagent network init`` (CLI) and ``openagent serve``
    (auto-bootstrap on first run, so the user only needs one command).
    """
    agent_name = config.get("name", "openagent")
    name = f"{agent_name}-personal"

    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        existing = await store.get_network_role()
        if existing is not None and existing["role"] != "standalone":
            return existing

        identity = load_or_create_identity(_identity_path(agent_dir))
        network_id = uuid.uuid4().hex

        await store.set_network_role(
            role="coordinator",
            network_id=network_id,
            name=name,
            coordinator_node_id=_node_id_for(identity),
            coordinator_pubkey=identity.public_bytes,
        )
        await store.register_agent(
            handle=agent_name,
            node_id=_node_id_for(identity),
            owner_handle="system",
            label=f"{agent_name} (this agent)",
        )
        if not quiet:
            console.print(
                f"[dim]No network configured — auto-created personal network "
                f"[cyan]{name}[/cyan].[/dim]"
            )
        return await store.get_network_role()
    finally:
        if db._conn is not None:
            await db._conn.close()


async def mint_first_user_invite(
    *,
    agent_dir: Path,
    config: dict,
    network_row: dict,
) -> tuple[str, dict] | None:
    """Return a usable user-invite ticket while the coordinator has zero users.

    Reuses an existing unspent auto-bootstrap invite if one is still
    valid — otherwise mints a fresh one. Returns ``None`` once any user
    has joined, so the "first-time join" banner stops nagging.
    """
    from openagent.network.coordinator_addr_cache import read_cache
    from openagent.network.ticket import InviteTicket

    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        users = await store.list_users()
        if users:
            return None
        # Reuse the most recent unspent auto-bootstrap invite if it's
        # still valid — saves cluttering the DB with one invite per boot.
        existing_invites = await store.list_invitations(include_expired=False)
        invite = next(
            (
                inv for inv in existing_invites
                if inv.role == "user"
                and inv.uses_left > 0
                and inv.created_by == "auto-bootstrap"
            ),
            None,
        )
        if invite is None:
            invite = await store.create_invitation(
                role="user",
                created_by="auto-bootstrap",
                ttl_seconds=7 * 24 * 3600,
                uses=1,
            )
        identity = load_or_create_identity(_identity_path(agent_dir))
        relay_url, addresses = read_cache(agent_dir)
        ticket = InviteTicket(
            code=invite.code,
            coordinator_node_id=_node_id_for(identity),
            network_name=network_row["name"],
            network_id=network_row["network_id"],
            role="user",
            bind_to="",
            relay_url=relay_url,
            addresses=addresses or None,
        )
        return ticket.encode(), {
            "code": invite.code,
            "expires_at": invite.expires_at,
        }
    finally:
        if db._conn is not None:
            await db._conn.close()


async def list_active_invite_tickets(
    *,
    agent_dir: Path,
    config: dict,
    network_row: dict,
) -> list[dict]:
    """Return every unspent, unexpired invite packed as a ticket string.

    Each entry is ``{ticket, code, role, bind_to, uses_left, expires_at,
    created_by}``. Used by ``serve`` to surface all redeemable invites
    on startup so the operator doesn't have to run ``network invite``
    just to see what's already minted.
    """
    from openagent.network.coordinator_addr_cache import read_cache
    from openagent.network.ticket import InviteTicket

    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        invites = await store.list_invitations(include_expired=False)
        identity = load_or_create_identity(_identity_path(agent_dir))
        node_id = _node_id_for(identity)
        relay_url, addresses = read_cache(agent_dir)
        addresses_or_none = addresses or None
        out: list[dict] = []
        for inv in invites:
            if inv.uses_left <= 0:
                continue
            ticket = InviteTicket(
                code=inv.code,
                coordinator_node_id=node_id,
                network_name=network_row["name"],
                network_id=network_row["network_id"],
                role=inv.role,
                bind_to=inv.bind_to_handle or "",
                relay_url=relay_url,
                addresses=addresses_or_none,
            )
            out.append({
                "ticket": ticket.encode(),
                "code": inv.code,
                "role": inv.role,
                "bind_to": inv.bind_to_handle or "",
                "uses_left": inv.uses_left,
                "expires_at": inv.expires_at,
                "created_by": inv.created_by,
            })
        return out
    finally:
        if db._conn is not None:
            await db._conn.close()


async def _run_init(ctx, name: str | None, personal: bool):
    agent_dir = _agent_dir_or_die()
    config = ctx.obj["config"]
    agent_name = config.get("name", "openagent")

    if name is None:
        name = f"{agent_name}-personal" if personal else f"{agent_name}-net"

    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        existing = await store.get_network_role()
        if existing is not None and existing["role"] != "standalone":
            console.print(
                f"[yellow]This agent already has role={existing['role']!r} "
                f"(network={existing.get('name')!r}). "
                "To re-init, delete the ``network`` row from the DB first.[/yellow]"
            )
            return

        identity = load_or_create_identity(_identity_path(agent_dir))
        network_id = uuid.uuid4().hex

        await store.set_network_role(
            role="coordinator",
            network_id=network_id,
            name=name,
            coordinator_node_id=_node_id_for(identity),
            coordinator_pubkey=identity.public_bytes,
        )

        # Register the agent itself so ``list_agents`` immediately shows
        # something useful. Owner is "system" — invitations bound to a
        # real handle override this when a user-driven add_agent runs.
        await store.register_agent(
            handle=agent_name,
            node_id=_node_id_for(identity),
            owner_handle="system",
            label=f"{agent_name} (this agent)",
        )

        console.print(f"[green]Network created.[/green]")
        table = Table(show_header=False, padding=(0, 2))
        table.add_row("Name", name)
        table.add_row("Network ID", network_id)
        table.add_row("NodeId", _node_id_for(identity))
        table.add_row("Identity path", str(_identity_path(agent_dir)))
        console.print(table)
        console.print(
            "\n[dim]Next: run [cyan]openagent network invite[/cyan] to issue an invite for "
            "your client (CLI/app), then [cyan]openagent-cli connect <handle>@" + name +
            " --invite <code>[/cyan].[/dim]"
        )
    finally:
        if db._conn is not None:
            await db._conn.close()


def _node_id_for(identity: Identity) -> str:
    """Compute the Iroh NodeId without binding an Endpoint.

    iroh-py 0.35 doesn't expose a Python ``SecretKey`` type; we derive
    the public key locally via ``cryptography`` (which we depend on
    anyway) and format it through ``iroh.PublicKey`` for canonical
    encoding.
    """
    from openagent.network.iroh_node import _node_id_from_secret

    return _node_id_from_secret(identity.secret_bytes)


@network_group.command("status")
@click.pass_context
def cmd_status(ctx):
    """Print this agent's current network role + identifiers."""
    asyncio.run(_run_status(ctx))


async def _run_status(ctx):
    agent_dir = _agent_dir_or_die()
    config = ctx.obj["config"]
    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None or row["role"] == "standalone":
            console.print(
                "[yellow]This agent is standalone (no network).[/yellow] "
                "Run [cyan]openagent network init[/cyan] to create one, or join "
                "an existing network from the desktop app."
            )
            return
        identity = load_or_create_identity(_identity_path(agent_dir))
        table = Table(title="Network status", show_header=False, padding=(0, 2))
        table.add_row("Role", row["role"])
        table.add_row("Name", row["name"] or "?")
        table.add_row("Network ID", row["network_id"] or "?")
        table.add_row("NodeId", _node_id_for(identity))
        console.print(table)
    finally:
        if db._conn is not None:
            await db._conn.close()


@network_group.command("invite")
@click.option(
    "--role",
    type=click.Choice(["user", "device", "agent"]),
    default="user",
    help="What this invite grants.",
)
@click.option(
    "--bind-to", default=None,
    help="For role=device: bind the invite to a specific user handle.",
)
@click.option(
    "--ttl", default=7 * 24 * 3600, show_default=True, type=int,
    help="Invite TTL in seconds.",
)
@click.option("--uses", default=1, show_default=True, type=int)
@click.pass_context
def cmd_invite(ctx, role: str, bind_to: str | None, ttl: int, uses: int):
    """Mint a one-shot invite code (coordinator-only)."""
    asyncio.run(_run_invite(ctx, role, bind_to, ttl, uses))


async def _run_invite(ctx, role: str, bind_to: str | None, ttl: int, uses: int):
    from openagent.network.coordinator_addr_cache import read_cache
    from openagent.network.ticket import InviteTicket

    agent_dir = _agent_dir_or_die()
    config = ctx.obj["config"]
    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None or row["role"] != "coordinator":
            console.print("[red]Not a coordinator-mode agent.[/red] "
                          "Run ``openagent network init`` first.")
            return
        invite = await store.create_invitation(
            role=role,
            created_by="cli",
            ttl_seconds=ttl,
            uses=uses,
            bind_to_handle=bind_to,
        )
        identity = load_or_create_identity(_identity_path(agent_dir))
        # Optional address hints — only present when the coordinator
        # has run at least once on this machine since the addr-cache
        # feature shipped. Missing/empty = old behaviour (client falls
        # back to iroh discovery on the dial).
        relay_url, addresses = read_cache(agent_dir)
        ticket = InviteTicket(
            code=invite.code,
            coordinator_node_id=_node_id_for(identity),
            network_name=row["name"],
            network_id=row["network_id"],
            role=invite.role,
            bind_to=bind_to or "",
            relay_url=relay_url,
            addresses=addresses or None,
        )
        ticket_str = ticket.encode()
        console.print()
        console.print(f"[green]Invite ticket — copy/paste this whole string:[/green]")
        console.print(f"\n  [bold cyan]{ticket_str}[/bold cyan]\n")
        console.print(
            f"  [dim]role={invite.role}"
            + (f", bound to {bind_to}" if bind_to else "")
            + f", uses={invite.uses_left}, expires {time.ctime(invite.expires_at)}[/dim]"
        )
        console.print(
            f"\n  Redeem with: [cyan]openagent-cli connect {ticket_str}[/cyan]\n"
        )
    finally:
        if db._conn is not None:
            await db._conn.close()


@network_group.command("users")
@click.pass_context
def cmd_users(ctx):
    """List users registered in this (coordinator-mode) network."""
    asyncio.run(_run_users(ctx))


async def _run_users(ctx):
    config = ctx.obj["config"]
    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None or row["role"] != "coordinator":
            console.print("[red]Not a coordinator-mode agent.[/red]")
            return
        users = await store.list_users()
        table = Table(title=f"Users ({len(users)})")
        table.add_column("Handle", style="cyan")
        table.add_column("Status")
        table.add_column("PAKE algo", style="dim")
        table.add_column("Created", style="dim")
        for u in users:
            table.add_row(u.handle, u.status, u.pake_algo, time.ctime(u.created_at))
        console.print(table)
    finally:
        if db._conn is not None:
            await db._conn.close()


@network_group.command("agents")
@click.pass_context
def cmd_agents(ctx):
    """List agents registered in this network."""
    asyncio.run(_run_agents(ctx))


async def _run_agents(ctx):
    config = ctx.obj["config"]
    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None:
            console.print("[red]Not part of any network.[/red]")
            return
        if row["role"] != "coordinator":
            console.print(
                "[yellow]This agent is a member, not the coordinator. "
                "Use the desktop app or ``openagent-cli agents`` to list "
                "via the coordinator's RPC.[/yellow]"
            )
            return
        agents = await store.list_agents()
        table = Table(title=f"Agents in {row['name']!r} ({len(agents)})")
        table.add_column("Handle", style="cyan")
        table.add_column("NodeId", style="dim")
        table.add_column("Owner", style="dim")
        table.add_column("Label")
        for a in agents:
            table.add_row(a.handle, a.node_id[:24] + "…", a.owner_handle, a.label or "")
        console.print(table)
    finally:
        if db._conn is not None:
            await db._conn.close()


@network_group.command("revoke")
@click.option("--device", "device_pubkey_hex", required=True, help="Hex-encoded device pubkey")
@click.pass_context
def cmd_revoke(ctx, device_pubkey_hex: str):
    """Mark a device as revoked (coordinator-only)."""
    asyncio.run(_run_revoke(ctx, device_pubkey_hex))


async def _run_revoke(ctx, device_pubkey_hex: str):
    config = ctx.obj["config"]
    db = await _open_db(config)
    try:
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None or row["role"] != "coordinator":
            console.print("[red]Not a coordinator-mode agent.[/red]")
            return
        try:
            pubkey = bytes.fromhex(device_pubkey_hex)
        except ValueError:
            console.print("[red]device pubkey must be hex.[/red]")
            return
        ok = await store.revoke_device(pubkey)
        console.print(f"[{'green' if ok else 'yellow'}]"
                      f"{'Revoked.' if ok else 'No active device with that pubkey.'}"
                      f"[/]")
    finally:
        if db._conn is not None:
            await db._conn.close()


# ── Helper for first-time CLI consumers ─────────────────────────────────


@network_group.command("loopback")
@click.argument("target")
@click.option("--handle", "handle_override", default=None,
              help="When redeeming a user-role ticket, the handle to register as.")
@click.option("--agent", "agent_handle", default=None,
              help="Specific agent handle (default: first registered)")
@click.option("--password-stdin", is_flag=True,
              help="Read password from stdin (1 line). For non-interactive use.")
@click.option("--print-port", is_flag=True,
              help="Print only the bound port number on stdout (machine-readable).")
@click.pass_context
def cmd_loopback(ctx, target, handle_override, agent_handle, password_stdin, print_port):
    """Run a localhost ↔ Iroh proxy for the desktop app.

    \b
    Two forms (same as ``openagent-cli connect``):
      openagent network loopback oa1abcdef…       # invite ticket
      openagent network loopback alice@homelab    # existing membership

    Prints the bound port on stdout; the caller hits ``http://127.0.0.1:<port>``
    and ``ws://127.0.0.1:<port>/ws`` exactly like before. Exits when stdin
    closes — the Electron main process should keep the pipe open for the
    lifetime of the connection.
    """
    from openagent.network.iroh_node import DialError

    try:
        asyncio.run(_run_loopback(
            ctx, target, handle_override, agent_handle, password_stdin, print_port,
        ))
    except DialError as e:
        click.echo(str(e), err=True)
        raise click.exceptions.Exit(2)


async def _run_loopback(ctx, target, handle_override, agent_handle, password_stdin, print_port):
    import getpass
    import sys

    from openagent.network import user_store
    from openagent.network.auth.device_cert import verify_cert
    from openagent.network.client.login import (
        LoginError,
        list_agents as coord_list_agents,
        login as net_login,
        register as net_register,
    )
    from openagent.network.client.session import (
        LoopbackProxy, NetworkBinding, SessionDialer,
    )
    from openagent.network.identity import load_or_create_identity
    from openagent.network.iroh_node import IrohNode
    from openagent.network.peers import coordinator_node_id_to_pubkey_bytes
    from openagent.network.ticket import InviteTicket, TicketError, looks_like_ticket
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    coordinator_node_id: str | None = None
    invite_code: str | None = None
    ticket_role: str | None = None
    bind_to: str = ""
    if looks_like_ticket(target):
        try:
            ticket = InviteTicket.decode(target)
        except TicketError as e:
            click.echo(f"invalid ticket: {e}", err=True)
            raise click.Abort()
        coordinator_node_id = ticket.coordinator_node_id
        invite_code = ticket.code
        ticket_role = ticket.role
        bind_to = ticket.bind_to
        network_name = ticket.network_name
        handle = bind_to or (handle_override or "").strip().lower()
        if not handle:
            click.echo(
                "user-role tickets need --handle on the loopback flow "
                "(no interactive prompt available here)",
                err=True,
            )
            raise click.Abort()
    else:
        try:
            handle, network_name = parse_handle_at_network(target)
        except ValueError as e:
            click.echo(str(e), err=True)
            raise click.Abort()

    store = user_store.load()
    net = user_store.find(store, network_name)

    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass(f"Password for {handle}@{network_name}: ")
    if not password:
        click.echo("password is required", err=True)
        raise click.Abort()

    user_store.ensure_user_identity_dir()
    device_identity = load_or_create_identity(user_store.user_identity_path())
    node = IrohNode(device_identity)
    await node.start()

    if net is None:
        # First-time join — must have a ticket.
        if not coordinator_node_id or not invite_code:
            await node.stop()
            click.echo(
                f"unknown network {network_name!r}; paste an oa1… ticket "
                "(it carries the coordinator NodeId + invite code in one string)",
                err=True,
            )
            raise click.Abort()
        coord_pubkey = coordinator_node_id_to_pubkey_bytes(coordinator_node_id)
        try:
            if ticket_role == "device":
                cert_wire = await net_login(
                    node=node, coordinator_node_id=coordinator_node_id,
                    coordinator_pubkey_bytes=coord_pubkey, handle=handle,
                    password=password, device_identity=device_identity,
                    network_id="", invite_code=invite_code,
                )
            else:
                cert_wire = await net_register(
                    node=node, coordinator_node_id=coordinator_node_id,
                    coordinator_pubkey_bytes=coord_pubkey, handle=handle,
                    password=password, invite_code=invite_code,
                    device_identity=device_identity, network_id="",
                )
        except LoginError as e:
            await node.stop()
            click.echo(f"join failed: {e}", err=True)
            raise click.Abort()
        cert = verify_cert(
            cert_wire,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pubkey),
        )
        net = user_store.add_or_update(
            store, name=network_name, network_id=cert.network_id,
            coordinator_node_id=coordinator_node_id,
            coordinator_pubkey_hex=coord_pubkey.hex(), handle=handle,
        )
        user_store.write_cert(net, cert_wire)
        user_store.save(store)
    elif net.handle != handle:
        await node.stop()
        click.echo(f"network {network_name} bound to {net.handle}, not {handle}", err=True)
        raise click.Abort()
    else:
        try:
            cert_wire = await net_login(
                node=node,
                coordinator_node_id=net.coordinator_node_id,
                coordinator_pubkey_bytes=net.coordinator_pubkey_bytes,
                handle=handle,
                password=password,
                device_identity=device_identity,
                network_id=net.network_id,
            )
        except LoginError as e:
            await node.stop()
            click.echo(f"login failed: {e}", err=True)
            raise click.Abort()
        user_store.write_cert(net, cert_wire)
        user_store.save(store)

    agents = await coord_list_agents(node=node, coordinator_node_id=net.coordinator_node_id)
    if not agents:
        await node.stop()
        click.echo("no agents registered in network", err=True)
        raise click.Abort()
    chosen = None
    if agent_handle:
        chosen = next((a for a in agents if a.get("handle") == agent_handle), None)
    if chosen is None:
        chosen = agents[0]
    target_node_id = chosen["node_id"]

    binding = NetworkBinding(
        network_id=net.network_id, network_name=net.name,
        coordinator_node_id=net.coordinator_node_id,
        coordinator_pubkey_bytes=net.coordinator_pubkey_bytes,
        our_handle=handle,
    )
    dialer = SessionDialer(node=node, binding=binding, cert_wire=cert_wire)
    proxy = LoopbackProxy(dialer=dialer, target_node_id=target_node_id)
    host, port = await proxy.start()

    if print_port:
        # One line, just the port — easy to parse from a child-process pipe.
        click.echo(str(port))
        sys.stdout.flush()
    else:
        click.echo(f"loopback listening on http://{host}:{port}")
        click.echo(f"  ws:    ws://{host}:{port}/ws")
        click.echo(f"  agent: {chosen['handle']} ({target_node_id[:24]}…)")
        click.echo("Close stdin (Ctrl-D) or send SIGINT to stop.")

    # Block until stdin closes — the parent process is what owns our
    # lifetime. Reading stdin avoids burning CPU and lets the parent
    # keep us alive cleanly via an open pipe.
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sys.stdin.read)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await proxy.stop()
        await dialer.close()
        await node.stop()


def parse_handle_at_network(spec: str) -> tuple[str, str]:
    """Split ``alice@homelab`` into ``("alice", "homelab")``.

    Used by ``openagent-cli connect`` and the desktop app's onboarding
    screen. Raises ``ValueError`` for malformed input so the caller
    can render a friendly error.
    """
    if "@" not in spec:
        raise ValueError(f"expected handle@network, got {spec!r}")
    handle, _, network = spec.partition("@")
    handle = handle.strip().lower()
    network = network.strip().lower()
    if not handle or not network:
        raise ValueError(f"empty handle or network in {spec!r}")
    return handle, network
