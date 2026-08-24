"""``/api/skills`` — the file-backed skill library, over the gateway.

Exercises the handlers against a temporary skills root (the same
``OPENAGENT_SKILLS_PATH`` the MCP tools honour), so the REST façade and the
tools are proven to agree on where a skill lives.

The request stand-in models one thing deliberately: aiohttp lets the body be
read ONCE. An earlier version of ``handle_create`` parsed the payload to get
``name`` and then delegated to a helper that parsed it again — the second
read returned empty and the endpoint answered "body is required" to a caller
that had sent one. A permissive fake hid that; this one raises instead.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ._framework import TestContext, test


class _FakeRequest:
    def __init__(self, match=None, query=None, body=None):
        self.match_info = match or {}
        self.query = query or {}
        self.app = {}          # no gateway → broadcasts are skipped
        self._body = body
        self._read = False

    @property
    def can_read_body(self):
        return self._body is not None and not self._read

    async def json(self):
        if self._read:
            raise AssertionError("request body read twice — aiohttp would return empty")
        self._read = True
        return self._body


def _api():
    from src.gateway.api import skills as api

    return api


async def _json(resp):
    return resp.status, json.loads(resp.body.decode())


@test("rest_skills", "create → list → read → update round-trips on disk")
async def t_round_trip(ctx: TestContext) -> None:
    root = Path(tempfile.mkdtemp()) / "skills"
    root.mkdir(parents=True)
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        api = _api()

        status, body = await _json(await api.handle_list(_FakeRequest()))
        assert status == 200 and body["skills"] == []

        status, body = await _json(await api.handle_create(_FakeRequest(body={
            "name": "Release check", "description": "Before shipping",
            "category": "ops", "body": "1. Tests\n2. Changelog",
        })))
        assert status == 201 and body["ok"], body
        # A write lands on disk; the prompt index is a frozen snapshot.
        assert body["index_refreshed"] is False

        status, body = await _json(await api.handle_list(_FakeRequest()))
        assert len(body["skills"]) == 1
        row = body["skills"][0]
        assert row["name"] == "Release check" and row["category"] == "ops"
        assert row["agent_authored"] is True and row["archived"] is False

        status, body = await _json(
            await api.handle_get(_FakeRequest(match={"name": "Release check"})))
        assert status == 200 and "1. Tests" in body["body"]

        status, _ = await _json(await api.handle_update(
            _FakeRequest(match={"name": "Release check"},
                         body={"body": "1. Tests\n2. Changelog\n3. Tag"})))
        assert status == 200
        status, body = await _json(
            await api.handle_get(_FakeRequest(match={"name": "Release check"})))
        assert "3. Tag" in body["body"]
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev


@test("rest_skills", "validation: name and body required, duplicates refused")
async def t_validation(ctx: TestContext) -> None:
    root = Path(tempfile.mkdtemp()) / "skills"
    root.mkdir(parents=True)
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        api = _api()

        status, body = await _json(
            await api.handle_create(_FakeRequest(body={"body": "x"})))
        assert status == 400 and "name is required" in body["error"]

        # The regression: a body IS supplied, so this must not answer
        # "body is required" — that was the double-read bug.
        status, body = await _json(
            await api.handle_create(_FakeRequest(body={"name": "N", "body": "real body"})))
        assert status == 201, body

        status, body = await _json(
            await api.handle_create(_FakeRequest(body={"name": "N", "body": "again"})))
        assert status == 409 and "already exists" in body["error"]

        status, body = await _json(
            await api.handle_create(_FakeRequest(body={"name": "NoBody"})))
        assert status == 400 and "body is required" in body["error"]

        status, _ = await _json(
            await api.handle_get(_FakeRequest(match={"name": "missing"})))
        assert status == 404
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev


@test("rest_skills", "archive retires without deleting; delete removes")
async def t_archive_then_delete(ctx: TestContext) -> None:
    root = Path(tempfile.mkdtemp()) / "skills"
    root.mkdir(parents=True)
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        api = _api()
        await api.handle_create(_FakeRequest(body={"name": "S", "body": "b"}))

        status, _ = await _json(
            await api.handle_archive(_FakeRequest(match={"name": "S"})))
        assert status == 200

        # Gone from the default view (as it is from the prompt index)…
        _s, body = await _json(await api.handle_list(_FakeRequest()))
        assert body["skills"] == []
        # …but still on disk, and visible when asked for.
        _s, body = await _json(
            await api.handle_list(_FakeRequest(query={"include_archived": "1"})))
        assert len(body["skills"]) == 1 and body["skills"][0]["archived"] is True

        status, _ = await _json(
            await api.handle_delete(_FakeRequest(match={"name": "S"})))
        assert status == 200
        _s, body = await _json(
            await api.handle_list(_FakeRequest(query={"include_archived": "1"})))
        assert body["skills"] == []

        status, _ = await _json(
            await api.handle_delete(_FakeRequest(match={"name": "S"})))
        assert status == 404
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev


@test("rest_skills", "search requires a query")
async def t_search_requires_q(ctx: TestContext) -> None:
    status, body = await _json(await _api().handle_search(_FakeRequest(query={})))
    assert status == 400 and "q is required" in body["error"]
