# Example skills

These are **seed skills** — ready-made `SKILL.md` playbooks you can drop into
an agent's skills directory as templates. They demonstrate the format the
native Skills subsystem expects and give you a starting point to edit.

- [`support-triage/`](support-triage/SKILL.md) — a generic customer-support
  triage playbook.
- [`git-commit/`](git-commit/SKILL.md) — how to write a good git commit.

## The SKILL.md format

Each skill lives in its own folder as `SKILL.md`, with YAML frontmatter and a
markdown body:

```markdown
---
name: support-triage
description: One line describing when to reach for this skill.
category: support
---

# Body

The full instructions, steps, and examples. Only the `name` + `description`
(grouped by `category`) are loaded into the system prompt up front —
progressive disclosure. The agent reads this whole body on demand via
`skill_view` right before it acts on the skill.
```

Only `name` is required. `description` and `category` are strongly
recommended (a skill with no `category` falls into `general`). A malformed or
nameless `SKILL.md` is skipped, never fatal.

## Activating skills

The Skills subsystem is **OFF by default**. Turn it on in `openagent.yaml`:

```yaml
skills:
  enabled: true
  # Optional: point at any directory of skill folders. Defaults to
  # <data_dir>/skills (honours OPENAGENT_SKILLS_PATH, like the vault).
  path: /absolute/path/to/skills
```

Then either set `skills.path` to a directory containing these folders, or
copy them into the default location:

```bash
# Default skills directory (macOS shown; Linux uses XDG data dir):
cp -R examples/skills/support-triage \
      examples/skills/git-commit \
      "$HOME/Library/Application Support/OpenAgent/skills/"
```

Restart the agent so the skills index is rescanned into the (cached) system
prompt.

## Provenance and the skill-curator

The optional **skill-curator** is a scheduled self-improvement task
("dream-mode for skills") that reviews, merges, and archives skills the agent
taught *itself*. It is gated separately and OFF by default:

```yaml
skills:
  enabled: true
  curator_enabled: true            # second, independent switch
  curator_schedule: "0 4 * * 0"    # optional cron; default is weekly (Sun 04:00)
```

The curator only ever touches skills whose frontmatter carries
`created_by: agent` — the stamp `skill_manage` writes when the agent creates a
skill. **These seed skills deliberately carry no `created_by` field**, so the
curator will never merge, rewrite, or archive them. Hand-curated seed and user
content is off-limits to consolidation by design.
