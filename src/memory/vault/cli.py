"""``openagent vault`` CLI — run the quality gate and friends from a shell.

This is the Company-Brain ``gate.py`` script as a first-class command:
``openagent vault gate`` prints the report (the "0 errors" referto),
``openagent vault doctor --apply`` runs the fix loop, ``vault sync`` /
``vault derive`` / ``vault stats`` / ``vault search`` round it out. Operates
directly on the Markdown files, so it works without a running gateway.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json as _json
from pathlib import Path

import click


def _resolve_root(explicit: str | None) -> Path:
    from src.core.config import load_config
    from src.memory.vault.service import resolve_vault_root
    if explicit:
        return Path(explicit).expanduser().resolve()
    cfg = load_config()
    vp = (cfg.get("memory", {}) or {}).get("vault_path")
    return resolve_vault_root(vp)


def _service(explicit: str | None):
    from src.memory.vault.service import VaultService
    return VaultService(_resolve_root(explicit))


def _print_report(rep) -> None:
    click.echo(click.style(rep.summary_line(),
                           fg="green" if rep.ok else "red", bold=True))
    s = rep.stats
    click.echo(
        f"  notes={s.get('gated_notes')} links={s.get('links')} "
        f"broken={s.get('broken_links')} orphans={s.get('orphans')} "
        f"islands={s.get('islands')} components={s.get('components')}"
    )
    by_rule = rep.by_rule()
    for rule in sorted(by_rule):
        vs = by_rule[rule]
        click.echo(click.style(f"\n[{rule}] ({len(vs)})", bold=True))
        for v in vs[:50]:
            click.echo(f"  {v.severity:5} {v.path}: {v.message}")
        if len(vs) > 50:
            click.echo(f"  … and {len(vs) - 50} more")


@click.group("vault")
def vault_group():
    """Evaluate and maintain the memory vault (quality gate, doctor, index)."""


@vault_group.command("gate")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
@click.option("--strict", is_flag=True, help="Treat every issue as an error.")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw JSON report.")
def cmd_gate(vault_path, strict, as_json):
    """Run the quality gate over every note and print the report."""
    async def _run():
        svc = _service(vault_path)
        cfg = dataclasses.replace(svc.config, strict=True) if strict else svc.config
        rep = await svc.gate(config=cfg)
        await svc.close()
        return rep
    rep = asyncio.run(_run())
    if as_json:
        click.echo(_json.dumps(rep.to_dict(), indent=2))
    else:
        _print_report(rep)
    raise SystemExit(0 if rep.ok else 1)


@vault_group.command("doctor")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
@click.option("--apply", is_flag=True, help="Write fixes (default: dry run).")
def cmd_doctor(vault_path, apply):
    """Mechanically fix what code can; list the rest as suggestions."""
    async def _run():
        svc = _service(vault_path)
        res = await svc.doctor(apply=apply)
        await svc.close()
        return res
    res = asyncio.run(_run())
    fix = res["fix"]
    head = "Applied" if apply else "Would apply"
    click.echo(click.style(
        f"{head} mechanical fixes to {len(fix['fixed'])} note(s) "
        f"(files changed: {fix['files_changed']})", bold=True))
    for f in fix["fixed"][:80]:
        click.echo(f"  {f['path']}: {', '.join(f['fixes'])}")
    click.echo(click.style(
        f"\n{len(fix['suggestions'])} issue(s) need a human/AI decision:", bold=True))
    for s in fix["suggestions"][:80]:
        click.echo(f"  [{s['rule']}] {s['path']}: {s['message']}")


@vault_group.command("sync")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
@click.option("--force", is_flag=True, help="Re-parse every note.")
def cmd_sync(vault_path, force):
    """Reconcile the index with the notes on disk."""
    async def _run():
        svc = _service(vault_path)
        stats = await svc.sync(force=force)
        await svc.close()
        return stats
    click.echo(_json.dumps(asyncio.run(_run()), indent=2))


@vault_group.command("derive")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
def cmd_derive(vault_path):
    """Regenerate llms.txt and _showcase/showcase.md."""
    async def _run():
        svc = _service(vault_path)
        res = await svc.regenerate_derived()
        await svc.close()
        return res
    click.echo(_json.dumps(asyncio.run(_run()), indent=2))


@vault_group.command("stats")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
def cmd_stats(vault_path):
    """Print vault health (notes, links, orphans, components)."""
    async def _run():
        svc = _service(vault_path)
        st = await svc.stats()
        await svc.close()
        return st
    click.echo(_json.dumps(asyncio.run(_run()), indent=2))


@vault_group.command("init")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
@click.option("--no-git", is_flag=True, help="Don't install/initialise the git repo.")
def cmd_init(vault_path, no_git):
    """Scaffold the 11-folder system + journal tree + canon workspace, and
    set up the git repo that tracks every change."""
    import os as _os
    if no_git:
        # Honour the flag end-to-end: no install AND no repo/commits this run.
        _os.environ["OPENAGENT_VAULT_GIT_ENABLED"] = "0"
    else:
        import shutil as _shutil
        if not _shutil.which("git"):
            from src.setup.bootstrap import ensure_git as _ensure_git
            click.echo("Installing git for the vault repo…")
            if not _ensure_git():
                click.echo(click.style(
                    "Could not install git — the vault works, but changes "
                    "won't be tracked. Install git and re-run.", fg="yellow"))

    async def _run():
        svc = _service(vault_path)
        res = await svc.init_taxonomy()
        await svc.close()
        return res
    res = asyncio.run(_run())
    click.echo(click.style(
        f"Scaffolded {res['count']} folder(s)/file(s):", bold=True))
    for c in res["created"]:
        click.echo(f"  {c}")
    if not res["created"]:
        click.echo("  (everything already in place)")


@vault_group.command("mv")
@click.argument("old_path")
@click.argument("new_path")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
def cmd_mv(old_path, new_path, vault_path):
    """Move/rename a note or folder, rewriting every inbound wikilink."""
    async def _run():
        svc = _service(vault_path)
        res = await svc.move(old_path, new_path)
        await svc.close()
        return res
    res = asyncio.run(_run())
    if "error" in res:
        click.echo(click.style(res["error"], fg="red"))
        raise SystemExit(1)
    click.echo(
        f"Moved {res['notes_moved']} note(s); rewrote {res['links_rewritten']} "
        f"link(s) across {res['notes_updated']} note(s)."
    )


@vault_group.command("search")
@click.argument("query")
@click.option("--vault", "vault_path", default=None, help="Vault path override.")
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--files-only", is_flag=True, help="Search only file names/paths, not content.")
def cmd_search(query, vault_path, limit, files_only):
    """Search the vault — full-text by default, or file names only with --files-only."""
    async def _run():
        svc = _service(vault_path)
        if files_only:
            res = await svc.search_files(query, limit=limit)
        else:
            res = await svc.search(query, limit=limit)
        await svc.close()
        return res
    for r in asyncio.run(_run()):
        tags = f" [{', '.join(r['tags'])}]" if r.get("tags") else ""
        click.echo(f"{r['path']}  —  {r.get('title','')}{tags}")
