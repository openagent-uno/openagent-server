"""Expose the shared filesystem/editor implementations through MCPPool.

The server intentionally instantiates the standalone package's core servers
directly: no client consent, broker, principal, or client path metadata crosses
this boundary. Only implementation code and manifest semantics are shared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openagent_host_tools.builtins import EditorServer, FilesystemServer
from openagent_host_tools.types import HostError, ToolResult, tool_error_result


def _wire(result: ToolResult) -> dict[str, Any]:
    return result.to_wire()


async def _call(server: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _wire(await server.call(tool, args))
    except HostError as exc:
        return tool_error_result(exc).to_wire()


def _args(**values: Any) -> dict[str, Any]:
    return {name: value for name, value in values.items() if value is not None}


def _exact_tools(server: Any, callables: list[Any]) -> list[Any]:
    """Bind runtime entrypoints to the package's authoritative wire schemas."""
    from src.mcp._runtime import Function

    by_name = {function.__name__: function for function in callables}
    return [
        Function(
            name=manifest.name,
            description=manifest.description,
            parameters=manifest.input_schema,
            classification=manifest.classification.value,
            entrypoint=by_name[manifest.name],
        )
        for manifest in server.manifest.tools
    ]


def build_filesystem_runtime_toolkit(
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    from src.mcp._runtime import Toolkit

    roots = _filesystem_roots(args, env)
    server = FilesystemServer(Path.cwd(), allowed_roots=roots)

    async def read_text_file(
        path: str, head: int | None = None, tail: int | None = None
    ) -> dict:
        """Read a UTF-8 text file on the OpenAgent server."""
        return await _call(server, "read_text_file", _args(path=path, head=head, tail=tail))

    async def read_media_file(path: str) -> dict:
        """Read an image or audio file on the OpenAgent server."""
        return await _call(server, "read_media_file", {"path": path})

    async def read_multiple_files(paths: list[str]) -> dict:
        """Read several UTF-8 files on the OpenAgent server."""
        return await _call(server, "read_multiple_files", {"paths": paths})

    async def write_file(path: str, content: str) -> dict:
        """Create or overwrite a UTF-8 file on the OpenAgent server."""
        return await _call(server, "write_file", {"path": path, "content": content})

    async def edit_file(
        path: str, edits: list[dict[str, str]], dryRun: bool = False
    ) -> dict:
        """Apply exact text replacements on the OpenAgent server."""
        return await _call(
            server, "edit_file", {"path": path, "edits": edits, "dryRun": dryRun}
        )

    async def create_directory(path: str) -> dict:
        """Create a directory and missing parents on the OpenAgent server."""
        return await _call(server, "create_directory", {"path": path})

    async def list_directory(path: str) -> dict:
        """List a directory on the OpenAgent server."""
        return await _call(server, "list_directory", {"path": path})

    async def list_directory_with_sizes(path: str, sortBy: str = "name") -> dict:
        """List server directory entries with byte sizes."""
        return await _call(
            server,
            "list_directory_with_sizes",
            {"path": path, "sortBy": sortBy},
        )

    async def directory_tree(
        path: str, max_depth: int = 5, max_entries: int = 5000
    ) -> dict:
        """Return a bounded recursive tree from the OpenAgent server."""
        return await _call(
            server,
            "directory_tree",
            {"path": path, "max_depth": max_depth, "max_entries": max_entries},
        )

    async def move_file(source: str, destination: str) -> dict:
        """Move or rename a server file without overwriting."""
        return await _call(
            server, "move_file", {"source": source, "destination": destination}
        )

    async def search_files(
        path: str,
        pattern: str,
        excludePatterns: list[str] | None = None,
        max_results: int = 1000,
    ) -> dict:
        """Find server paths matching a glob pattern."""
        return await _call(
            server,
            "search_files",
            _args(
                path=path,
                pattern=pattern,
                excludePatterns=excludePatterns,
                max_results=max_results,
            ),
        )

    async def get_file_info(path: str) -> dict:
        """Return metadata for a server file or directory."""
        return await _call(server, "get_file_info", {"path": path})

    async def list_allowed_directories() -> dict:
        """Describe the filesystem visible to the server process."""
        return await _call(server, "list_allowed_directories", {})

    return Toolkit(
        name="filesystem",
        tools=_exact_tools(server, [
            read_text_file,
            read_media_file,
            read_multiple_files,
            write_file,
            edit_file,
            create_directory,
            list_directory,
            list_directory_with_sizes,
            directory_tree,
            move_file,
            search_files,
            get_file_info,
            list_allowed_directories,
        ]),
        instructions=(
            "These paths are on the OpenAgent server. Never describe them as client-local."
        ),
    )


def _filesystem_roots(
    args: list[str] | None,
    env: dict[str, str] | None,
) -> list[str]:
    """Resolve the server filesystem allowlist without changing client policy."""

    if args:
        raw_roots = list(args)
    else:
        override = (env or {}).get("OPENAGENT_FILESYSTEM_ROOTS")
        if override is None:
            override = os.environ.get("OPENAGENT_FILESYSTEM_ROOTS", "")
        raw_roots = [part for part in override.split(os.pathsep) if part] if override else ["/"]
    roots = [
        str(Path(raw).expanduser().resolve())
        for raw in raw_roots
        if Path(raw).expanduser().is_dir()
    ]
    if not roots:
        raise HostError(
            "invalid_configuration",
            "filesystem MCP has no existing allowed root",
        )
    return roots


def build_editor_runtime_toolkit() -> Any:
    from src.mcp._runtime import Toolkit

    server = EditorServer(Path.cwd())

    async def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict:
        """Surgically replace text in a file on the OpenAgent server."""
        return await _call(
            server,
            "edit",
            {
                "file_path": file_path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
        )

    async def grep(
        pattern: str,
        path: str = ".",
        file_pattern: str | None = None,
        context: int = 0,
        case_insensitive: bool = False,
        max_results: int = 100,
    ) -> dict:
        """Search a regex across server files with context."""
        return await _call(
            server,
            "grep",
            _args(
                pattern=pattern,
                path=path,
                file_pattern=file_pattern,
                context=context,
                case_insensitive=case_insensitive,
                max_results=max_results,
            ),
        )

    async def glob(pattern: str, path: str = ".", max_results: int = 100) -> dict:
        """Find server files matching a glob, newest first."""
        return await _call(
            server,
            "glob",
            {"pattern": pattern, "path": path, "max_results": max_results},
        )

    return Toolkit(
        name="editor",
        tools=_exact_tools(server, [edit, grep, glob]),
        instructions=(
            "These paths are on the OpenAgent server. Never describe them as client-local."
        ),
    )
