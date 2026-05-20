# Contributing to Khora

Thanks for your interest in contributing. Khora is an open-source knowledge
memory library - knowledge graphs + vector search + PostgreSQL (with an
optional SQLite + LanceDB embedded path).

## Local setup

```bash
git clone https://github.com/DeytaHQ/khora.git
cd khora
uv sync --all-extras
```

You need Python 3.13 and [`uv`](https://github.com/astral-sh/uv). For
integration tests against Postgres/Neo4j you also need Docker.

## Running tests

The full suite runs unit tests in parallel and integration tests serially,
appending into a single coverage report:

```bash
make test
```

For tighter feedback loops, pick a subset:

| Command | What it runs | Needs Docker? |
| --- | --- | --- |
| `make test-unit` | `tests/unit/` in parallel (`-n auto`) | No |
| `make test-embedded` | Tests marked `embedded` - the SQLite+LanceDB stack | No |
| `make test-integration` | `tests/integration/` against Postgres+Neo4j | Yes (`make dev` first) |
| `make test-soak` | Long-running soak/burn-in tests (`-m soak`) | No |

`make dev` starts the local Postgres + Neo4j containers. `make dev-down`
stops them.

## Coverage gates

Two gates run in CI:

1. **Aggregate gate** - `--cov-fail-under=53` in `pyproject.toml`. Protects
   the global percentage; tripped by broad regressions.
2. **Per-path floors** - `scripts/check_coverage_floors.py`. Reads
   `coverage.json` and enforces a per-file minimum for the hot paths the
   project has invested in (currently the embedded SQLite+LanceDB stack,
   FTS5 escaping, the chronicle engine, and the `_accel` module).

The per-path floors catch the case where aggregate coverage is fine but a
specific file silently loses test coverage. To re-check locally after a
test run:

```bash
uv run coverage json -o coverage.json
uv run python scripts/check_coverage_floors.py
```

When you legitimately need to lower a floor (e.g. you intentionally removed
test surface), do it in the same commit as the corresponding code change
and explain why in the commit message.

## Pre-commit

```bash
uv run prek install
```

This installs the `prek` (a `pre-commit` rewrite in Rust) hooks declared in
`.pre-commit-config.yaml`. They run `ruff format`, `ruff check`, and `ty`
locally before each commit. The same checks run again in CI.

## PR workflow

1. Find or open an issue at
   https://github.com/DeytaHQ/khora/issues (`gh issue list` from the CLI).
2. Branch off `main`. Use `<initials>/<short-desc>` as the branch name.
3. Make focused commits - explain *why* in the message, not just *what*.
4. Run `make format && make test` before pushing.
5. Open a PR against `main`. Reference the issue with `Fixes #<n>` in the
   body so the issue auto-closes on merge.
6. CI must be green. Squash-merge is the default.

## Finding work

```bash
gh issue list --label good-first-issue
gh issue list --state open
```

If you're unsure where to start, look at issues labelled `good-first-issue`
or `help-wanted`.

## Reporting bugs

Open a GitHub issue with:

- What you ran (command + minimal repro).
- What you expected vs. what happened.
- Khora version (`pip show khora`), Python version, and OS.
- Relevant traceback or log output.

## License

By contributing, you agree your contributions will be licensed under the
project's Apache 2.0 license (see `LICENSE`).
