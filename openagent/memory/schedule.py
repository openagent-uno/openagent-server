"""Shared helpers for scheduled task expressions."""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Mapping

from croniter import croniter

ONE_SHOT_PREFIX = "@once:"


def is_one_shot_expression(expr: str | None) -> bool:
    return bool(expr and str(expr).startswith(ONE_SHOT_PREFIX))


def build_one_shot_expression(run_at: float) -> str:
    return f"{ONE_SHOT_PREFIX}{float(run_at)}"


def parse_one_shot_expression(expr: str) -> float:
    if not is_one_shot_expression(expr):
        raise ValueError(f"Not a one-shot schedule expression: {expr!r}")
    try:
        return float(str(expr)[len(ONE_SHOT_PREFIX):])
    except ValueError as exc:
        raise ValueError(f"Invalid one-shot schedule expression: {expr!r}") from exc


def validate_schedule_expression(expr: str) -> None:
    if is_one_shot_expression(expr):
        parse_one_shot_expression(expr)
        return
    try:
        croniter(expr)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid cron expression {expr!r}: {exc}") from exc


def next_run_for_expression(expr: str, base: float | None = None) -> float:
    if is_one_shot_expression(expr):
        return parse_one_shot_expression(expr)
    try:
        return croniter(expr, time.time() if base is None else base).get_next(float)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid cron expression: {exc}") from exc


def epoch_to_iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def decorate_scheduled_task(row: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    task = dict(row)
    task["enabled"] = bool(task.get("enabled"))
    task["run_once"] = is_one_shot_expression(task.get("cron_expression"))
    if task["run_once"]:
        run_at = parse_one_shot_expression(task["cron_expression"])
        task["run_at"] = run_at
        task["run_at_iso"] = epoch_to_iso(run_at)
    for ts_col in ("last_run", "next_run", "created_at", "updated_at"):
        value = task.get(ts_col)
        if isinstance(value, (int, float)):
            task[f"{ts_col}_iso"] = epoch_to_iso(value)
    return task
