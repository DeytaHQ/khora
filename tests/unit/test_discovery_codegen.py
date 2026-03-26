"""Tests for discovery code generation, AST validation, and sandboxed execution.

Covers:
    - AST validation: allowed imports, blocked imports/calls/attrs, evasion
    - Script execution: happy path, timeout, env stripping, file detection
    - Script template rendering
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from khora.discovery.codegen import (
    Violation,
    execute_script,
    render_template,
    validate_script,
)

# ===================================================================
# AST validation — allowed patterns
# ===================================================================


class TestValidateAllowed:
    """Scripts using only allowed constructs should produce zero violations."""

    def test_empty_script(self) -> None:
        assert validate_script("") == []

    def test_allowed_imports(self) -> None:
        code = textwrap.dedent("""\
            import httpx
            import csv
            import json
            import pathlib
            import zipfile
            import gzip
            import io
            import re
            import datetime
            import xml.etree.ElementTree
            import time
            import sys
            import math
            from pathlib import Path
            from urllib.parse import urljoin
            from collections import defaultdict
        """)
        assert validate_script(code) == []

    def test_os_path_allowed(self) -> None:
        code = "import os.path\nresult = os.path.join('a', 'b')\n"
        assert validate_script(code) == []

    def test_httpx_usage(self) -> None:
        code = textwrap.dedent("""\
            import httpx
            import json
            from pathlib import Path

            resp = httpx.get("https://example.com/data.json")
            data = resp.json()
            Path("output.json").write_text(json.dumps(data))
        """)
        assert validate_script(code) == []

    def test_open_in_normal_code(self) -> None:
        # open() for file writing is allowed in the actual script;
        # it is only blocked via getattr/string tricks.
        # The validator blocks `open` as a name in BLOCKED_NAMES,
        # but scripts should use pathlib instead.  This test documents
        # that open() IS flagged — scripts should use Path.write_text().
        code = "f = open('test.txt', 'w')\n"
        violations = validate_script(code)
        assert len(violations) == 1
        assert "open()" in violations[0].message


# ===================================================================
# AST validation — blocked patterns
# ===================================================================


class TestValidateBlocked:
    """Dangerous patterns must produce violations."""

    def test_import_subprocess(self) -> None:
        violations = validate_script("import subprocess")
        assert len(violations) == 1
        assert "subprocess" in violations[0].message

    def test_from_subprocess(self) -> None:
        violations = validate_script("from subprocess import run")
        assert len(violations) == 1
        assert "subprocess" in violations[0].message

    def test_import_shutil(self) -> None:
        violations = validate_script("import shutil")
        assert len(violations) == 1
        assert "shutil" in violations[0].message

    def test_import_socket(self) -> None:
        violations = validate_script("import socket")
        assert len(violations) == 1
        assert "socket" in violations[0].message

    def test_import_os_bare(self) -> None:
        violations = validate_script("import os")
        assert len(violations) == 1
        assert "os" in violations[0].message

    def test_from_os_import(self) -> None:
        violations = validate_script("from os import system")
        assert len(violations) == 1
        assert "os" in violations[0].message

    def test_import_pickle(self) -> None:
        violations = validate_script("import pickle")
        assert len(violations) == 1
        assert "pickle" in violations[0].message

    def test_import_ctypes(self) -> None:
        violations = validate_script("import ctypes")
        assert len(violations) == 1
        assert "ctypes" in violations[0].message

    def test_import_not_in_allowlist(self) -> None:
        violations = validate_script("import requests")
        assert len(violations) == 1
        assert "allowlist" in violations[0].message

    def test_call_eval(self) -> None:
        violations = validate_script("eval('1+1')")
        assert len(violations) == 1
        assert "eval()" in violations[0].message

    def test_call_exec(self) -> None:
        violations = validate_script("exec('print(1)')")
        assert len(violations) == 1
        assert "exec()" in violations[0].message

    def test_call_compile(self) -> None:
        violations = validate_script("compile('x=1', '<>', 'exec')")
        assert len(violations) == 1
        assert "compile()" in violations[0].message

    def test_dunder_import(self) -> None:
        violations = validate_script("__import__('os')")
        assert len(violations) >= 1
        messages = " ".join(v.message for v in violations)
        assert "__import__" in messages

    def test_os_system_attribute(self) -> None:
        code = "import os.path\nos.system('ls')\n"
        violations = validate_script(code)
        # Should flag os.system call even if os.path is imported
        assert any("os.system" in v.message for v in violations)

    def test_blocked_attr_access_no_call(self) -> None:
        # Even accessing os.environ without calling it
        code = "import os.path\nx = os.environ\n"
        violations = validate_script(code)
        assert any("os.environ" in v.message for v in violations)

    def test_syntax_error(self) -> None:
        violations = validate_script("def foo(:\n")
        assert len(violations) == 1
        assert "SyntaxError" in violations[0].message

    def test_multiple_violations(self) -> None:
        code = textwrap.dedent("""\
            import subprocess
            import shutil
            eval("bad")
        """)
        violations = validate_script(code)
        assert len(violations) == 3


# ===================================================================
# AST validation — evasion detection
# ===================================================================


class TestValidateEvasion:
    """Attempts to circumvent the validator must be caught."""

    def test_getattr_os_system(self) -> None:
        code = "import os.path\ngetattr(os, 'system')('ls')\n"
        violations = validate_script(code)
        assert len(violations) >= 1
        messages = " ".join(v.message for v in violations)
        assert "getattr" in messages

    def test_getattr_string_concat(self) -> None:
        code = "import os.path\ngetattr(os, 'sys' + 'tem')('ls')\n"
        violations = validate_script(code)
        assert len(violations) >= 1
        messages = " ".join(v.message for v in violations)
        assert "getattr" in messages or "os.system" in messages

    def test_getattr_on_blocked_module(self) -> None:
        # Even if we can't resolve the attr string, getattr on a
        # dangerous module name is flagged.
        code = "import os.path\nf = getattr(os, some_var)\n"
        violations = validate_script(code)
        assert len(violations) >= 1
        messages = " ".join(v.message for v in violations)
        assert "getattr" in messages

    def test_getattr_on_safe_module_ok(self) -> None:
        # getattr on non-dangerous modules should NOT be flagged
        code = textwrap.dedent("""\
            import json
            fn = getattr(json, 'dumps')
        """)
        violations = validate_script(code)
        assert len(violations) == 0

    def test_import_importlib(self) -> None:
        violations = validate_script("import importlib")
        assert len(violations) == 1
        assert "importlib" in violations[0].message


# ===================================================================
# Script execution
# ===================================================================


class TestExecuteScript:
    @pytest.mark.asyncio
    async def test_simple_script(self, tmp_path: Path) -> None:
        """A trivial script that writes a file and prints JSON summary."""
        script = textwrap.dedent("""\
            import json
            from pathlib import Path

            Path("output.json").write_text(json.dumps({"key": "value"}))
            print(json.dumps({"files": ["output.json"], "total_records": 1}))
        """)

        result = await execute_script(script, tmp_path)

        assert result.success
        assert result.exit_code == 0
        assert not result.timed_out
        assert "output.json" in result.files_created
        assert result.summary.get("total_records") == 1
        assert (tmp_path / "output.json").exists()

    @pytest.mark.asyncio
    async def test_script_error(self, tmp_path: Path) -> None:
        """A script that raises an exception."""
        script = "raise RuntimeError('boom')\n"
        result = await execute_script(script, tmp_path)

        assert not result.success
        assert result.exit_code != 0
        assert "boom" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path) -> None:
        """A script that exceeds the timeout is killed."""
        script = textwrap.dedent("""\
            import time
            time.sleep(60)
        """)

        result = await execute_script(script, tmp_path, timeout=1)

        assert not result.success
        assert result.timed_out
        assert result.duration_seconds < 5  # should not wait full 60s

    @pytest.mark.asyncio
    async def test_env_stripping(self, tmp_path: Path) -> None:
        """The subprocess should not see the parent's full environment."""
        script = textwrap.dedent("""\
            import os
            import json
            # os import will be in the env but the script CAN use it at runtime
            # (validation is separate from execution)
            env_keys = list(os.environ.keys())
            print(json.dumps({"env_keys": env_keys}))
        """)

        result = await execute_script(script, tmp_path)

        assert result.success
        env_keys = result.summary.get("env_keys", [])
        # Should have minimal env, not the full parent env
        assert "PATH" in env_keys
        assert "HOME" in env_keys
        # Common dev env vars should NOT be present
        assert "USER" not in env_keys
        assert "SHELL" not in env_keys

    @pytest.mark.asyncio
    async def test_approved_env_forwarded(self, tmp_path: Path) -> None:
        """Approved env keys are forwarded to the subprocess."""
        import os

        os.environ["_KHORA_TEST_KEY"] = "secret123"
        try:
            script = textwrap.dedent("""\
                import os, json
                val = os.environ.get("_KHORA_TEST_KEY", "")
                print(json.dumps({"key_value": val}))
            """)

            result = await execute_script(script, tmp_path, approved_env_keys=["_KHORA_TEST_KEY"])

            assert result.success
            assert result.summary.get("key_value") == "secret123"
        finally:
            del os.environ["_KHORA_TEST_KEY"]

    @pytest.mark.asyncio
    async def test_multiple_files_detected(self, tmp_path: Path) -> None:
        """All files created by the script are detected."""
        script = textwrap.dedent("""\
            import json
            from pathlib import Path

            Path("a.csv").write_text("h1,h2\\n1,2\\n")
            Path("b.json").write_text('{"x": 1}')
            sub = Path("subdir")
            sub.mkdir(exist_ok=True)
            (sub / "c.txt").write_text("hello")

            print(json.dumps({"files": ["a.csv", "b.json", "subdir/c.txt"]}))
        """)

        result = await execute_script(script, tmp_path)

        assert result.success
        assert "a.csv" in result.files_created
        assert "b.json" in result.files_created
        assert "subdir/c.txt" in result.files_created

    @pytest.mark.asyncio
    async def test_script_file_cleaned_up(self, tmp_path: Path) -> None:
        """The temporary script file is removed after execution."""
        script = "print('hello')\n"
        await execute_script(script, tmp_path)
        assert not (tmp_path / "_khora_fetch_script.py").exists()

    @pytest.mark.asyncio
    async def test_json_summary_last_line(self, tmp_path: Path) -> None:
        """JSON summary is parsed from the last line even with other output."""
        script = textwrap.dedent("""\
            import json
            print("Fetching page 1...")
            print("Fetching page 2...")
            print("Done.")
            print(json.dumps({"files": ["data.csv"], "total_records": 42}))
        """)

        result = await execute_script(script, tmp_path)
        assert result.success
        assert result.summary.get("total_records") == 42

    @pytest.mark.asyncio
    async def test_no_json_in_stdout(self, tmp_path: Path) -> None:
        """When stdout has no JSON, summary is an empty dict."""
        script = "print('just text, no json')\n"
        result = await execute_script(script, tmp_path)
        assert result.success
        assert result.summary == {}


# ===================================================================
# Script template
# ===================================================================


class TestScriptTemplate:
    def test_render_basic(self) -> None:
        body = textwrap.dedent("""\
            resp = httpx.get("https://example.com/data.csv")
            Path("data.csv").write_bytes(resp.content)
            files_written.append("data.csv")
            total_records = 1
        """)

        script = render_template(
            title="Example Dataset",
            url="https://example.com/data.csv",
            fetch_body=body,
        )

        # Should be valid Python
        assert validate_script(script) == []
        assert "Example Dataset" in script
        assert "https://example.com/data.csv" in script
        assert "httpx.get" in script
        assert "json.dumps(summary)" in script

    def test_render_compiles(self) -> None:
        """The rendered template must be valid Python (compilable)."""
        body = "pass\n"
        script = render_template(title="Test", url="http://x", fetch_body=body)
        compile(script, "<test>", "exec")  # raises SyntaxError if invalid

    def test_render_preserves_logic(self) -> None:
        body = textwrap.dedent("""\
            for page in range(5):
                resp = httpx.get(f"https://api.example.com/data?page={page}")
                data = resp.json()
                if not data:
                    break
                fname = f"page_{page}.json"
                Path(fname).write_text(json.dumps(data))
                files_written.append(fname)
                total_records += len(data)
                time.sleep(1)
        """)

        script = render_template(
            title="Paginated API",
            url="https://api.example.com/data",
            fetch_body=body,
        )

        assert validate_script(script) == []
        assert "for page in range(5):" in script
        assert "time.sleep(1)" in script

    def test_title_with_quotes(self) -> None:
        script = render_template(
            title='He said "hello"',
            url="http://x",
            fetch_body="pass\n",
        )
        compile(script, "<test>", "exec")


# ===================================================================
# Violation dataclass
# ===================================================================


class TestViolation:
    def test_str_representation(self) -> None:
        v = Violation(line=10, col=4, message="blocked call: eval()")
        assert str(v) == "line 10:4 - blocked call: eval()"
