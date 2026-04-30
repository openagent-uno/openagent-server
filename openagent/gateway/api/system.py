"""GET /api/system + ``system_snapshot`` WS broadcast — host telemetry.

Cross-platform (Windows, macOS, Linux) snapshot of the machine running
the OpenAgent server: CPU, memory, swap, disks, network throughput, top
processes. Implemented entirely on top of ``psutil`` so the same code
returns the same shape on every supported OS.

Two surfaces:

  * **REST** ``GET /api/system`` — one-shot snapshot (initial paint,
    manual refresh).
  * **WS broadcast** ``system_snapshot`` — emitted on a background tick
    every :data:`BROADCAST_INTERVAL_S` seconds to all authenticated
    clients. The desktop app's System screen subscribes here and
    re-renders without polling.

Network and per-process CPU rates are deltas between consecutive ticks,
so the first snapshot has zeroes — every subsequent one has live values.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)

# How often the gateway pushes a snapshot. 2s feels live without
# burning CPU on the iter-processes call (which can take 50–200ms on
# busy hosts).
BROADCAST_INTERVAL_S = 2.0

# Cap on the process list returned per snapshot. Top N by CPU.
MAX_PROCESSES = 25


class SystemTelemetry:
    """Stateful telemetry sampler.

    Holds the previous network counters so each snapshot can compute
    bytes/sec rates, and primes ``psutil.cpu_percent`` so the first
    real snapshot returns non-zero usage instead of the documented
    "0.0 on first call" footgun.

    One instance lives on the Gateway; both the REST handler and the
    broadcast loop call ``snapshot()`` against it.
    """

    def __init__(self) -> None:
        self._prev_net_counters: Any = None
        self._prev_net_ts: float = 0.0
        self._lock = asyncio.Lock()
        self._primed = False
        # Disk usage cache. On macOS we shell out to osascript to read
        # the URL-resource keys that match Finder (purgeable-aware),
        # and that costs ~100ms — too much to redo every 2s tick. Disk
        # usage doesn't move second-to-second anyway, so a 30s TTL is
        # plenty fresh while keeping the broadcast loop cheap.
        self._disks_cache: list[dict[str, Any]] | None = None
        self._disks_cache_ts: float = 0.0

    async def prime(self) -> None:
        """First-call priming for psutil.cpu_percent.

        ``psutil.cpu_percent(interval=None)`` returns 0.0 on the very
        first call because it has no baseline to diff against. Calling
        it once at startup means the first real snapshot reports a
        meaningful CPU value.
        """
        if self._primed:
            return
        self._primed = True
        await asyncio.to_thread(_prime_cpu)

    async def snapshot(self) -> dict[str, Any]:
        """Collect a full telemetry snapshot.

        Runs the psutil calls in a worker thread so a slow
        ``process_iter`` (hundreds of processes on a busy box) doesn't
        stall the asyncio event loop.
        """
        async with self._lock:
            await self.prime()
            return await asyncio.to_thread(self._collect)

    def _collect(self) -> dict[str, Any]:
        import psutil

        now = time.time()
        snap: dict[str, Any] = {
            "timestamp": now,
            "host": _collect_host(),
            "cpu": _collect_cpu(),
            "memory": _collect_memory(),
            "swap": _collect_swap(),
            "disks": self._collect_disks_cached(now),
            "network": self._collect_network(now),
            "processes": _collect_processes(),
        }
        return snap

    def _collect_disks_cached(self, now: float) -> list[dict[str, Any]]:
        """Disk usage with a 30s TTL.

        Underlying call is cheap on Linux/Windows (statvfs) but takes
        ~100ms on macOS (osascript subprocess to read the
        Finder-matching URL-resource keys). 30s is a fine staleness
        budget for a UI metric that changes by megabytes per minute
        in normal use.
        """
        if self._disks_cache is not None and (now - self._disks_cache_ts) < 30.0:
            return self._disks_cache
        disks = _collect_disks()
        self._disks_cache = disks
        self._disks_cache_ts = now
        return disks

    def _collect_network(self, now: float) -> dict[str, Any]:
        import psutil

        try:
            counters = psutil.net_io_counters()
        except Exception:
            counters = None

        rx_bps = 0.0
        tx_bps = 0.0
        if counters is not None and self._prev_net_counters is not None:
            dt = max(now - self._prev_net_ts, 1e-6)
            rx_bps = max(0.0, (counters.bytes_recv - self._prev_net_counters.bytes_recv) / dt)
            tx_bps = max(0.0, (counters.bytes_sent - self._prev_net_counters.bytes_sent) / dt)
        if counters is not None:
            self._prev_net_counters = counters
            self._prev_net_ts = now

        primary_iface, ipv4, ipv6 = _primary_interface()
        connections = _safe_connection_count()

        return {
            "primary_iface": primary_iface,
            "ipv4": ipv4,
            "ipv6": ipv6,
            "rx_bytes_total": int(counters.bytes_recv) if counters else 0,
            "tx_bytes_total": int(counters.bytes_sent) if counters else 0,
            "rx_bps": rx_bps,
            "tx_bps": tx_bps,
            "connections": connections,
        }


# ── psutil collectors (sync; called via to_thread) ──────────────────────


def _prime_cpu() -> None:
    import psutil

    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)


def _collect_host() -> dict[str, Any]:
    import psutil

    try:
        loadavg = list(psutil.getloadavg())
    except (OSError, AttributeError):
        # Windows pre-5.6 had no getloadavg; recent psutil shims it but
        # still raises on some embedded contexts.
        loadavg = [0.0, 0.0, 0.0]

    boot = psutil.boot_time()
    try:
        users_count = len(psutil.users())
    except Exception:
        users_count = 0

    try:
        import openagent
        oa_version = getattr(openagent, "__version__", "?")
    except Exception:
        oa_version = "?"

    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "os": _human_os(),
        "release": platform.release(),
        "arch": platform.machine(),
        "uptime_seconds": int(time.time() - boot),
        "boot_time": boot,
        "loadavg": loadavg,
        "users": users_count,
        "python_version": platform.python_version(),
        "openagent_version": oa_version,
    }


def _human_os() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        mac = platform.mac_ver()[0]
        return f"macOS {mac}" if mac else "macOS"
    if sysname == "Windows":
        return f"Windows {platform.release()}"
    if sysname == "Linux":
        try:
            import distro  # type: ignore[import-not-found]
            return f"{distro.name(pretty=True)}"
        except Exception:
            return f"Linux {platform.release()}"
    return f"{sysname} {platform.release()}"


def _collect_cpu() -> dict[str, Any]:
    import psutil

    try:
        freq = psutil.cpu_freq()
        cur = float(freq.current) if freq and freq.current else 0.0
        fmin = float(freq.min) if freq and freq.min else 0.0
        fmax = float(freq.max) if freq and freq.max else 0.0
    except (NotImplementedError, OSError, AttributeError):
        cur = fmin = fmax = 0.0

    try:
        usage = float(psutil.cpu_percent(interval=None))
    except Exception:
        usage = 0.0
    try:
        per_core = list(psutil.cpu_percent(interval=None, percpu=True))
    except Exception:
        per_core = []

    return {
        "model": _cpu_model(),
        "cores_physical": psutil.cpu_count(logical=False) or 0,
        "cores_logical": psutil.cpu_count(logical=True) or 0,
        "freq_mhz": cur,
        "freq_min_mhz": fmin,
        "freq_max_mhz": fmax,
        "usage_pct": usage,
        "per_core_pct": per_core,
        "temp_c": _cpu_temp(),
    }


def _cpu_model() -> str:
    """Best-effort CPU model name. Falls back to ``platform.processor()``."""
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=1,
            )
            if out.returncode == 0:
                name = out.stdout.strip()
                if name:
                    return name
        elif sysname == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        elif sysname == "Windows":
            # platform.processor() is the canonical cross-platform value;
            # on Windows it returns the brand string. On macOS/Linux it
            # tends to be vague (e.g. "i386") which is why we override
            # those branches above.
            return platform.processor() or ""
    except Exception:
        pass
    return platform.processor() or ""


def _cpu_temp() -> float | None:
    """CPU temperature in °C, or None if unavailable.

    ``psutil.sensors_temperatures`` only ships data on Linux and very
    recent macOS builds — Windows always returns ``{}``. We pick a
    reasonable sensor key when present and just return ``None`` when
    the sensor surface is empty.
    """
    import psutil

    fn = getattr(psutil, "sensors_temperatures", None)
    if fn is None:
        return None
    try:
        data = fn()
    except Exception:
        return None
    if not data:
        return None
    # Prefer canonical keys when present; otherwise fall back to the
    # first entry the OS reports.
    for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
        entries = data.get(key)
        if entries:
            return float(entries[0].current)
    for entries in data.values():
        if entries:
            return float(entries[0].current)
    return None


def _collect_memory() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    cached = getattr(vm, "cached", None)
    return {
        "total_bytes": int(vm.total),
        "used_bytes": int(vm.used),
        "available_bytes": int(vm.available),
        "free_bytes": int(vm.free),
        "cached_bytes": int(cached) if cached is not None else None,
        "percent": float(vm.percent),
    }


def _collect_swap() -> dict[str, Any]:
    import psutil

    try:
        sw = psutil.swap_memory()
    except Exception:
        return {"total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "percent": 0.0}
    return {
        "total_bytes": int(sw.total),
        "used_bytes": int(sw.used),
        "free_bytes": int(sw.free),
        "percent": float(sw.percent),
    }


def _collect_disks() -> list[dict[str, Any]]:
    """User-visible disk partitions, deduplicated.

    macOS APFS synthesises a dozen mountpoints (``/System/Volumes/VM``,
    ``Preboot``, ``Update``, ``xarts``…) that all share the same storage
    pool and read identical usage numbers — surfacing them all just
    clutters the UI. We filter ``/System/Volumes`` and ``/private/var``
    on Darwin, then dedupe by ``(used_bytes, total_bytes)`` so any
    remaining APFS snapshots collapse into one row.

    On Linux/Windows the filter is a no-op (those paths don't exist),
    so the dedupe step still catches container-style collisions in
    bind mounts.

    On macOS we additionally overlay ``NSURLVolumeAvailableCapacity-
    ForImportantUsageKey`` (Finder's free-space number, which counts
    APFS *purgeable* space as available) on top of the psutil-derived
    list. Without that overlay psutil reports ~30 GB more "used" than
    System Settings shows on a typical Apple-silicon Mac, because
    statvfs has no concept of purgeable.
    """
    import psutil

    sysname = platform.system()
    out: list[dict[str, Any]] = []
    try:
        parts = psutil.disk_partitions(all=False)
    except Exception:
        return out

    seen_usage: set[tuple[int, int]] = set()
    for p in parts:
        if not p.mountpoint:
            continue
        if p.fstype.lower() in {"squashfs", "tmpfs", "devtmpfs", "overlay", "autofs"}:
            continue
        if sysname == "Darwin" and (
            p.mountpoint.startswith("/System/Volumes/")
            or p.mountpoint.startswith("/private/var/")
        ):
            continue
        try:
            usage = psutil.disk_usage(p.mountpoint)
        except (OSError, PermissionError):
            continue
        key = (int(usage.total), int(usage.used))
        if key in seen_usage:
            continue
        seen_usage.add(key)
        out.append({
            "mount": p.mountpoint,
            "device": p.device,
            "fs": p.fstype,
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "percent": float(usage.percent),
        })

    if sysname == "Darwin" and out:
        _macos_overlay_finder_capacities(out)

    return out


def _macos_overlay_finder_capacities(disks: list[dict[str, Any]]) -> None:
    """Patch ``free_bytes``/``used_bytes``/``percent`` on each macOS row
    with the Finder-matching value queried via Foundation.

    Mutates ``disks`` in place. On any failure the rows are left as
    psutil reported them so the caller still gets *some* number — the
    fallback is what we used to ship before this overlay landed.

    Implementation detail: the value we want is exposed only via
    ``NSURL`` resource keys, not via any POSIX/BSD syscall or the
    ``df`` family. ``psutil``'s statvfs path returns a smaller
    "available" because it doesn't count APFS purgeable space —
    Finder/System Settings do, which is why ours used to read
    ~30 GB higher used than the screenshot.
    """
    mounts = [d["mount"] for d in disks if d.get("mount")]
    if not mounts:
        return

    # Build a JXA snippet that batches all mountpoints into a single
    # ``osascript`` invocation so we pay the interpreter startup cost
    # once. Returns ``[{mount, total, important_available}, …]`` —
    # ``important_available`` is the same value Finder uses.
    import json
    import subprocess

    jxa_payload = json.dumps(mounts)
    jxa = (
        'ObjC.import("Foundation");\n'
        f'const mounts = {jxa_payload};\n'
        'const out = mounts.map(m => {\n'
        '  const url = $.NSURL.fileURLWithPath(m);\n'
        '  const r1 = Ref();\n'
        '  url.getResourceValueForKeyError(r1, $.NSURLVolumeAvailableCapacityForImportantUsageKey, null);\n'
        '  const imp = ObjC.unwrap(r1[0]);\n'
        '  const r2 = Ref();\n'
        '  url.getResourceValueForKeyError(r2, $.NSURLVolumeTotalCapacityKey, null);\n'
        '  const tot = ObjC.unwrap(r2[0]);\n'
        '  const r3 = Ref();\n'
        '  url.getResourceValueForKeyError(r3, $.NSURLVolumeAvailableCapacityKey, null);\n'
        '  const basic = ObjC.unwrap(r3[0]);\n'
        '  return {mount: m, total: tot, important: imp, basic: basic};\n'
        '});\n'
        'JSON.stringify(out);\n'
    )
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", jxa],
            capture_output=True, text=True, timeout=4,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("osascript probe failed: %s", e)
        return
    if proc.returncode != 0:
        logger.debug("osascript exit=%s stderr=%r", proc.returncode, proc.stderr[:200])
        return
    try:
        rows = json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return

    by_mount: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = r.get("mount")
        if not m:
            continue
        tot = r.get("total")
        if not isinstance(tot, (int, float)) or tot <= 0:
            continue
        imp = r.get("important")
        basic = r.get("basic")
        # Pick the larger of (important, basic) — both are valid free-
        # space numbers; ``important`` includes purgeable on user
        # volumes (Finder uses this) but returns ~0 on sealed
        # read-only volumes where ``basic`` is the right answer.
        candidates: list[int] = []
        if isinstance(imp, (int, float)):
            candidates.append(int(imp))
        if isinstance(basic, (int, float)):
            candidates.append(int(basic))
        if not candidates:
            continue
        free = max(candidates)
        by_mount[m] = {"total": int(tot), "free": free}

    for d in disks:
        meta = by_mount.get(d["mount"])
        if not meta:
            continue
        total = meta["total"]
        free = meta["free"]
        # Don't overlay if the result would *increase* used relative
        # to psutil — that means the URL keys returned stale or
        # smaller numbers (shouldn't happen, but be defensive).
        if free < d["free_bytes"]:
            continue
        used = max(0, total - free)
        d["total_bytes"] = total
        d["free_bytes"] = free
        d["used_bytes"] = used
        d["percent"] = round((used / total) * 100, 1) if total > 0 else 0.0


def _primary_interface() -> tuple[str, str, str]:
    """Pick the (interface, ipv4, ipv6) triple that's most likely "the" NIC.

    Cross-platform heuristic: the iface that has a non-loopback ipv4
    AND is reported as up (when ``net_if_stats`` is available — every
    modern psutil supports it). Falls back to ``("", "", "")`` when
    nothing matches.
    """
    import psutil

    try:
        addrs = psutil.net_if_addrs()
    except Exception:
        return ("", "", "")
    try:
        stats = psutil.net_if_stats()
    except Exception:
        stats = {}

    # Evaluate interfaces in a stable, deterministic order so successive
    # snapshots don't flip-flop between equally-eligible NICs.
    for name in sorted(addrs.keys()):
        if name.startswith(("lo", "Loopback")):
            continue
        s = stats.get(name)
        if s is not None and not s.isup:
            continue
        ipv4 = ""
        ipv6 = ""
        for a in addrs[name]:
            fam = getattr(a.family, "name", str(a.family))
            if fam == "AF_INET" and not ipv4 and not a.address.startswith("127."):
                ipv4 = a.address
            elif fam == "AF_INET6" and not ipv6 and a.address != "::1":
                ipv6 = a.address.split("%", 1)[0]  # strip scope id
        if ipv4:
            return (name, ipv4, ipv6)
    return ("", "", "")


def _safe_connection_count() -> int:
    """Best-effort socket count.

    ``psutil.net_connections`` requires elevated privileges on macOS to
    enumerate every process's sockets; without root it raises
    ``AccessDenied``. We catch and return 0 in that case rather than
    propagating — the field is informational, not load-bearing.
    """
    import psutil

    try:
        return len(psutil.net_connections(kind="inet"))
    except (psutil.AccessDenied, PermissionError):
        return 0
    except Exception:
        return 0


def _collect_processes() -> list[dict[str, Any]]:
    """Top processes by CPU%.

    Uses ``process_iter`` and grabs the named attrs in one psutil pass
    for speed. On the first invocation every ``cpu_percent()`` is 0.0
    (the documented psutil semantics) — by the second invocation, when
    the broadcast loop has prior baselines, the values are real.
    """
    import psutil

    rows: list[dict[str, Any]] = []
    attrs = ["pid", "name", "username", "cpu_percent", "memory_info",
             "num_threads", "status"]
    try:
        for p in psutil.process_iter(attrs=attrs):
            info = p.info
            mem = info.get("memory_info")
            rss = int(mem.rss) if mem is not None else 0
            rows.append({
                "pid": int(info.get("pid") or 0),
                "name": info.get("name") or "",
                "user": info.get("username") or "",
                "cpu_pct": float(info.get("cpu_percent") or 0.0),
                "rss_bytes": rss,
                "threads": int(info.get("num_threads") or 0),
                "status": info.get("status") or "",
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("process_iter failed: %s", e)
        return []

    rows.sort(key=lambda r: r["cpu_pct"], reverse=True)
    return rows[:MAX_PROCESSES]


# ── HTTP handler ────────────────────────────────────────────────────────


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    gw = request.app["gateway"]
    telemetry: SystemTelemetry = gw._system_telemetry  # set in Gateway.start
    snap = await telemetry.snapshot()
    return _web.json_response(snap)
