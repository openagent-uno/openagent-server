"""In-process ``vault-gate`` MCP — the agent's handle on vault quality.

Gives the agent native, offline tools to evaluate and repair its own memory
vault: run the quality gate, see what is wrong (orphans, broken links, over-
long notes, duplicates, missing frontmatter …), mechanically fix what code
can fix, validate a note before writing it, search, and regenerate the
derived ``llms.txt`` / showcase. No external process, no network — pure
Python over the Markdown files, so it works on a fully self-hosted agent.
"""
