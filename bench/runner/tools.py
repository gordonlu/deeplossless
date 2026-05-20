"""Simulated tool calls for deterministic benchmarking. No real file I/O beyond initial load."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ToolResult:
    output: str
    exit_code: int
    duration_ms: float


class ToolRegistry:
    """Simulated tool execution. Reads from cached file contents, not disk."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def execute(self, tool: str, args: str, files: Dict[str, str]) -> ToolResult:
        """Execute a simulated tool call."""
        if tool in ("read_file", "read"):
            return self._read_file(args, files)
        elif tool in ("grep", "search_content", "search"):
            return self._grep(args, files)
        elif tool in ("run_tests", "test"):
            return self._run_tests(args, files)
        elif tool in ("edit_file", "edit"):
            return self._edit_file(args, files)
        elif tool in ("list_dir", "ls", "list_files"):
            return self._list_dir(args, files)
        else:
            return ToolResult(f"Unknown tool: {tool}", 1, 0.1)

    def _read_file(self, path: str, files: Dict[str, str]) -> ToolResult:
        """Read file from cached contents."""
        # Normalize path
        path = path.strip().strip('"\'')
        if path in files:
            content = files[path]
            lines = content.split('\n')
            # Return first 100 lines (simulates partial read)
            shown = '\n'.join(lines[:100])
            return ToolResult(f"[{len(lines)} lines]\n{shown}", 0, 2.0)
        # Try partial match
        for fpath in files:
            if fpath.endswith(path) or path in fpath:
                content = files[fpath]
                lines = content.split('\n')
                shown = '\n'.join(lines[:100])
                return ToolResult(f"[{len(lines)} lines]\n{shown}", 0, 2.5)
        return ToolResult(f"File not found: {path}", 1, 0.5)

    def _grep(self, pattern: str, files: Dict[str, str]) -> ToolResult:
        """Search for pattern across all files."""
        pattern = pattern.strip().strip('"\'')
        results = []
        for path, content in sorted(files.items()):
            for i, line in enumerate(content.split('\n'), 1):
                if pattern in line:
                    results.append(f"{path}:{i}: {line.strip()}")
        if results:
            return ToolResult(f"found {len(results)} matches\n" + '\n'.join(results[:20]), 0, 3.0)
        return ToolResult("found 0 matches", 0, 1.0)

    def _run_tests(self, _args: str, files: Dict[str, str]) -> ToolResult:
        """Run simulated tests based on file contents.
        A test passes if:
        - No reference to old/unfixed symbols or patterns
        - All planned edits appear to be applied
        """
        errors = []

        # Check for remaining bug patterns (these should be gone if fixed)
        bug_patterns = [
            ("old_db_driver", "ImportError: old_db_driver is incompatible"),
            ("from fasthttp import Response", "ImportError: fasthttp.Response renamed to HTTPResponse"),
            ("import async_timeout", "DeprecationWarning: async_timeout"),
            ("UserService(", "NameError: UserService renamed to AccountService"),
            ("from user_service import UserService", "ImportError: UserService not found"),
            ("DATABASE_URL", "ConfigError: DATABASE_URL renamed to DB_URL"),
            ('APP_PORT', "ConfigError: APP_PORT renamed to PORT"),
            ('RETRY_COUNT', "ConfigError: RETRY_COUNT renamed to MAX_RETRIES"),
            ('"/etc/app/config.json"', "ConfigError: config path should be ./config.json"),
        ]

        for pattern, error_msg in bug_patterns:
            for path, content in files.items():
                if path.endswith(('.py', '.txt', '.env', '.json')) and pattern in content:
                    errors.append(f"{path}:0: {error_msg}")

        if errors:
            return ToolResult('\n'.join(errors[:10]), 1, 5.0)
        return ToolResult("All tests passed.", 0, 5.0)

    def _edit_file(self, args: str, files: Dict[str, str]) -> ToolResult:
        """Simulated file edit."""
        parts = args.split('|')
        if len(parts) >= 3:
            path, old, new = parts[0], parts[1], parts[2]
            path = path.strip()
            if path in files:
                content = files[path]
                if old in content:
                    files[path] = content.replace(old, new)
                    return ToolResult(f"Edited {path}: replaced '{old[:30]}' -> '{new[:30]}'", 0, 0.5)
                return ToolResult(f"Pattern not found in {path}: '{old[:30]}'", 1, 0.5)
        return ToolResult("edit_file: invalid format, use path|old|new", 1, 0.5)

    def _list_dir(self, path: str, files: Dict[str, str]) -> ToolResult:
        """List directory contents from cached file paths."""
        path = path.strip().strip('"\'')
        if not path or path == ".":
            dirs = set()
            for f in sorted(files):
                parts = f.split('/')
                if len(parts) > 1:
                    dirs.add(parts[0] + '/')
                else:
                    dirs.add(f)
            return ToolResult('\n'.join(sorted(dirs)), 0, 1.0)
        # List specific directory
        prefix = path.rstrip('/') + '/'
        entries = set()
        for f in sorted(files):
            if f.startswith(prefix):
                rest = f[len(prefix):]
                parts = rest.split('/')
                if len(parts) > 1:
                    entries.add(parts[0] + '/')
                else:
                    entries.add(rest)
        return ToolResult('\n'.join(sorted(entries)) if entries else f"Directory empty: {path}", 0, 1.0)
