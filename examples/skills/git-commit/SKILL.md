---
name: git-commit
description: Write a clear, reviewable git commit — one logical change, an imperative subject, a body that explains why.
category: dev
---

# Writing a good commit

A commit is a message to the person (often you) who runs `git blame` in six
months. Optimize for their understanding, not for speed now.

## Scope: one logical change per commit

- Group only the changes that belong to a single idea. If you cannot
  describe the commit without the word "and", it is probably two commits.
- Do not mix a refactor with a behavior change — reviewers cannot tell which
  line caused a regression when both ride together.

## Subject line

- Imperative mood, as if completing "If applied, this commit will…":
  `Fix race in the scheduler`, not `Fixed` / `Fixes` / `Fixing`.
- Keep it under ~50 characters and do not end it with a period.
- Optionally prefix a scope: `scheduler: fix race on task cancel`.

## Body (the important part)

Separate it from the subject with a blank line, then explain **why**, not
what — the diff already shows what changed:

- What problem did this solve, and how would someone reproduce it?
- Why this approach over the obvious alternative?
- Any consequence a reader should know (migration, follow-up, known gap).

Wrap the body at ~72 columns so it reads well in `git log` and terminals.

## Before you commit

- Re-read the diff. Remove debug prints, stray files, and unrelated hunks.
- Make sure the change builds and its tests pass — a commit that does not
  build breaks `git bisect` for everyone after you.
- Reference the issue/ticket when there is one, so the context is one click
  away.

## Example

```
scheduler: fix race when a task is cancelled mid-run

get_due_tasks() and the cancellation drain both read the row, so a task
cancelled in the same tick it fired could be re-enqueued after its run
finished. Take the per-task lock before advancing next_run so the two
paths serialize.

Fixes #482.
```
