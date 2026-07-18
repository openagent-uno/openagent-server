"""Native Skills subsystem — Hermes / Claude-Code SKILL.md progressive
disclosure.

A file-backed set of skills under ``<data_dir>/skills/<skill-name>/SKILL.md``
(each with YAML frontmatter carrying at least ``name`` and ``description``,
plus optional bundled files alongside). Only a small *index* (category →
``name: description``) is injected into the cached system prefix; full skill
bodies load on demand via the ``skill_view`` tool. This mirrors the MCP
catalog's progressive-disclosure pattern.

OFF BY DEFAULT (gated on ``skills.enabled``). This package is fully
separate from the inert Agno ``skills`` stub / the dead ``skills`` DB table
— it never touches either.
"""
