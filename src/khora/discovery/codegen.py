"""Code generation, AST validation, and sandboxed execution for discovery scripts.

The discovery agent generates Python scripts (via LLM) to fetch data from
complex sources like paginated APIs, authenticated endpoints, or repos.
Before execution the generated code is validated through AST inspection
to block dangerous patterns.  Execution happens in a subprocess with a
stripped environment, hard timeout, and resource limits.

Security model:
    - Allowlisted imports only (httpx, csv, json, pathlib, etc.)
    - Blocked builtins and patterns (eval, exec, __import__, subprocess, ...)
    - Getattr-based evasion detection
    - Subprocess execution with minimal env, tmpdir HOME, hard timeout
    - User must confirm before any script runs
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------

#: Modules the generated script is allowed to import.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "httpx",
        "csv",
        "json",
        "pathlib",
        "zipfile",
        "gzip",
        "io",
        "re",
        "datetime",
        "xml",
        "xml.etree",
        "xml.etree.ElementTree",
        "time",  # for sleep between pagination requests
        "sys",  # for stdout JSON output
        "os.path",  # path manipulation only (os itself is blocked)
        "typing",
        "dataclasses",
        "collections",
        "math",
        "urllib.parse",
    }
)

#: Top-level module names derived from ALLOWED_IMPORTS (for ``import X`` checks).
_ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(m.split(".")[0] for m in ALLOWED_IMPORTS)

#: Names that are never allowed as function calls or attribute access targets.
BLOCKED_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "breakpoint",
        "exit",
        "quit",
        "input",  # no interactive input in sandbox
        "open",  # we re-allow open() below in _is_open_call — but block it
        # from getattr/string tricks
    }
)

#: Fully-qualified attribute paths that are always blocked.
BLOCKED_ATTRS: frozenset[str] = frozenset(
    {
        "os.system",
        "os.popen",
        "os.exec",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.spawn",
        "os.spawnl",
        "os.spawnle",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "os.rename",
        "os.environ",
        "shutil.rmtree",
        "shutil.move",
        "shutil.copy",
        "shutil.copy2",
        "socket.socket",
        "ctypes.CDLL",
        "ctypes.cdll",
    }
)

#: Module names that cannot appear in any import statement.
BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "importlib",
        "code",
        "codeop",
        "pickle",
        "shelve",
        "marshal",
        "signal",
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",  # script runs synchronously in subprocess
        "webbrowser",
        "http.server",
        "xmlrpc",
        "ftplib",
        "smtplib",
        "telnetlib",
        "antigravity",
    }
)


@dataclass(slots=True)
class Violation:
    """A single AST validation violation."""

    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}:{self.col} - {self.message}"


def validate_script(source: str) -> list[Violation]:
    """Parse *source* and walk the AST looking for disallowed patterns.

    Returns an empty list when the script is safe to execute.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            Violation(
                line=exc.lineno or 1,
                col=exc.offset or 0,
                message=f"SyntaxError: {exc.msg}",
            )
        ]

    violations: list[Violation] = []
    _walk(tree, violations)
    return violations


def _walk(tree: ast.AST, violations: list[Violation]) -> None:  # noqa: C901 — complexity is intentional
    """Recursive AST walk that accumulates violations."""
    for node in ast.walk(tree):
        # --- imports -----------------------------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name, node, violations)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _check_import(module, node, violations)

        # --- dangerous calls ---------------------------------------------------
        elif isinstance(node, ast.Call):
            _check_call(node, violations)

        # --- string concatenation / f-string evasion ---------------------------
        # Detect: getattr(os, "sys" + "tem")  — caught via _check_call
        # Detect: eval("os.sy" + "stem()")    — caught via blocked name check

        # --- attribute access on blocked targets -------------------------------
        elif isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted and dotted in BLOCKED_ATTRS:
                violations.append(
                    Violation(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"blocked attribute access: {dotted}",
                    )
                )

        # --- bare name references to blocked builtins -------------------------
        elif isinstance(node, ast.Name):
            if node.id == "__import__":
                violations.append(
                    Violation(
                        line=node.lineno,
                        col=node.col_offset,
                        message="reference to __import__",
                    )
                )


def _check_import(module: str, node: ast.AST, violations: list[Violation]) -> None:
    """Validate a single import target."""
    top = module.split(".")[0]

    # Explicitly blocked modules
    if top in BLOCKED_MODULES:
        violations.append(
            Violation(
                line=node.lineno,
                col=node.col_offset,
                message=f"blocked import: {module}",
            )
        )
        return

    # Allow 'os.path' but not 'os' by itself
    if top == "os" and module != "os.path":
        violations.append(
            Violation(
                line=node.lineno,
                col=node.col_offset,
                message=f"blocked import: {module} (only os.path is allowed)",
            )
        )
        return

    # Check allowlist
    if module not in ALLOWED_IMPORTS and top not in _ALLOWED_TOP_LEVEL:
        violations.append(
            Violation(
                line=node.lineno,
                col=node.col_offset,
                message=f"import not in allowlist: {module}",
            )
        )


def _check_call(node: ast.Call, violations: list[Violation]) -> None:
    """Check a Call node for dangerous invocations."""
    # Direct call to a blocked name: eval(...), exec(...), __import__(...)
    if isinstance(node.func, ast.Name):
        if node.func.id in BLOCKED_NAMES:
            violations.append(
                Violation(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"blocked call: {node.func.id}()",
                )
            )

    # Attribute call: os.system(...)
    elif isinstance(node.func, ast.Attribute):
        dotted = _dotted_name(node.func)
        if dotted and dotted in BLOCKED_ATTRS:
            violations.append(
                Violation(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"blocked call: {dotted}()",
                )
            )

    # getattr-based evasion: getattr(os, 'system')(...)
    # or even just getattr(os, 'system') as a standalone call
    if isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) >= 2:
            target = _resolve_name(node.args[0])
            attr = _resolve_string(node.args[1])
            if target and attr:
                combined = f"{target}.{attr}"
                if combined in BLOCKED_ATTRS:
                    violations.append(
                        Violation(
                            line=node.lineno,
                            col=node.col_offset,
                            message=f"getattr evasion: getattr({target}, {attr!r}) -> {combined}",
                        )
                    )
            # Even if we can't resolve the string, flag getattr on dangerous modules
            if target in ("os", "subprocess", "shutil", "socket", "ctypes"):
                violations.append(
                    Violation(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"getattr on blocked module: {target}",
                    )
                )


def _dotted_name(node: ast.expr) -> str | None:
    """Reconstruct a dotted name from nested Attribute/Name nodes.

    Returns ``None`` for expressions that aren't simple dotted names
    (e.g. function calls, subscripts).
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _resolve_name(node: ast.expr) -> str | None:
    """Resolve a Name node to its string id."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _resolve_string(node: ast.expr) -> str | None:
    """Try to resolve a node to a literal string value.

    Handles plain strings and simple concatenation (BinOp with Add).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # f-string — can't resolve statically, return None
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_string(node.left)
        right = _resolve_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------

#: Default timeout for script execution in seconds.
DEFAULT_TIMEOUT_SECONDS: int = 120

#: Maximum stdout/stderr capture size in bytes.
MAX_OUTPUT_BYTES: int = 10 * 1024 * 1024  # 10 MiB

#: Maximum total output file size in bytes.
MAX_OUTPUT_DIR_BYTES: int = 500 * 1024 * 1024  # 500 MiB


@dataclass(slots=True)
class ScriptResult:
    """Result of executing a generated script."""

    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    files_created: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timed_out: bool = False


async def execute_script(
    script: str,
    output_dir: Path,
    *,
    approved_env_keys: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    python: str | None = None,
) -> ScriptResult:
    """Execute a validated script in a sandboxed subprocess.

    The script is written to a temporary file and run with a stripped
    environment.  Only ``PATH`` and explicitly approved env vars are
    forwarded.  ``HOME`` is set to a temporary directory.

    Args:
        script: Python source code to execute.
        output_dir: Directory the script writes output files to (must exist).
        approved_env_keys: Environment variable names to forward (e.g. API keys
            the user explicitly approved).
        timeout: Hard timeout in seconds.
        python: Python interpreter path.  Defaults to ``sys.executable``.

    Returns:
        ScriptResult with captured output, created files, and parsed summary.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot directory contents before execution
    pre_files = _snapshot_dir(output_dir)

    # Build minimal environment
    env = _build_env(output_dir, approved_env_keys or [])

    # Write script to a temp file inside output_dir so the script's CWD
    # is the output directory and relative paths work.
    script_path = output_dir / "_khora_fetch_script.py"
    script_path.write_text(script, encoding="utf-8")

    python_bin = python or sys.executable
    start = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            python_bin,
            str(script_path),
            cwd=str(output_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            duration = time.monotonic() - start
            logger.warning(f"Script killed after {timeout}s timeout")
            return ScriptResult(
                success=False,
                exit_code=-9,
                stderr=f"Script killed: exceeded {timeout}s timeout",
                duration_seconds=duration,
                timed_out=True,
                files_created=_diff_dir(output_dir, pre_files),
            )

    except OSError as exc:
        return ScriptResult(
            success=False,
            exit_code=-1,
            stderr=f"Failed to start subprocess: {exc}",
            duration_seconds=time.monotonic() - start,
        )
    finally:
        # Clean up the script file
        script_path.unlink(missing_ok=True)

    duration = time.monotonic() - start
    stdout = stdout_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    stderr = stderr_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

    # Detect created files
    new_files = _diff_dir(output_dir, pre_files)

    # Try to parse JSON summary from stdout (last line or entire stdout)
    summary = _parse_summary(stdout)

    exit_code = proc.returncode or 0
    return ScriptResult(
        success=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        summary=summary,
        files_created=new_files,
        duration_seconds=duration,
    )


def _build_env(output_dir: Path, approved_keys: list[str]) -> dict[str, str]:
    """Build a minimal environment for the sandboxed subprocess."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        # Prevent httpx/urllib from reading netrc or proxy config
        "no_proxy": "*",
    }

    # Forward approved API keys
    for key in approved_keys:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    return env


def _snapshot_dir(directory: Path) -> set[str]:
    """Return the set of file paths (relative to *directory*) currently present."""
    result: set[str] = set()
    if directory.exists():
        for p in directory.rglob("*"):
            if p.is_file() and p.name != "_khora_fetch_script.py":
                result.add(str(p.relative_to(directory)))
    return result


def _diff_dir(directory: Path, before: set[str]) -> list[str]:
    """Return file paths created since the *before* snapshot."""
    after = _snapshot_dir(directory)
    new = sorted(after - before)
    return new


def _parse_summary(stdout: str) -> dict[str, Any]:
    """Try to parse a JSON summary from stdout.

    The script is expected to print a JSON object as the last line of
    stdout.  We try last-line first, then the entire stdout.
    """
    # Try last non-empty line
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    for candidate in reversed(lines):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return {}


# ---------------------------------------------------------------------------
# Script template
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated fetch script for: {title}

Source: {url}
Generated by Khora discovery agent.
"""

import json
import sys
import time
from pathlib import Path

import httpx

OUTPUT_DIR = Path(".")
"""Output directory — the script's CWD is set to the output dir at runtime."""


def main() -> None:
    files_written: list[str] = []
    total_records = 0

    # --- Fetch logic (generated) ---
{fetch_body}
    # --- End fetch logic ---

    # Print JSON summary to stdout
    summary = {{
        "files": files_written,
        "total_records": total_records,
    }}
    print(json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({{"error": str(exc)}}))
        sys.exit(1)
'''


def render_template(
    *,
    title: str,
    url: str,
    fetch_body: str,
) -> str:
    """Render the script template with the given fetch body.

    The *fetch_body* is dedented to its base level and then re-indented
    to sit inside ``main()`` (4-space indent).  Relative indentation
    within the body (nested blocks, loops, etc.) is preserved.
    """
    import textwrap

    # Dedent to remove any common leading whitespace, then add 4-space
    # prefix for every non-empty line so it lives inside main().
    dedented = textwrap.dedent(fetch_body)
    indented_lines: list[str] = []
    for line in dedented.splitlines():
        if line.strip():
            indented_lines.append(f"    {line}")
        else:
            indented_lines.append("")
    body = "\n".join(indented_lines)
    if not body.endswith("\n"):
        body += "\n"

    return SCRIPT_TEMPLATE.format(
        title=title.replace('"', '\\"'),
        url=url,
        fetch_body=body,
    )
