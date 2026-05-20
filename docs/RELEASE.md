# Release Process

## Tag Naming

Tags follow semantic versioning: `v{major}.{minor}.{patch}` (e.g., `v0.10.2`).

The `v` prefix is required - it triggers the publish workflow.

## Versioning

`khora` and `khora-accel` are released at **identical versions**, always. This is enforced mechanically at release time - see "Version lockstep" below.

| Package | How version is set |
|---------|-------------------|
| `khora` | `hatch-vcs` reads the most recent git tag at build time |
| `khora-accel` | `release.yml` extracts the tag and stamps it into `rust/khora-accel/Cargo.toml` before maturin builds |

At runtime, `khora.__version__` reads the installed package version via `importlib.metadata`.

In development (no tag on current commit), the version will be something like `0.10.3.dev3`.

## Where Packages Are Published

Both packages publish to **public PyPI** under the **Deyta** organization:

- https://pypi.org/project/khora/
- https://pypi.org/project/khora-accel/

Install: `pip install khora` (or `uv pip install khora`).

`khora-accel` ships as an **sdist only** (no platform wheels) - installers compile the Rust extension at install time via maturin's PEP 517 backend. **Requires a Rust toolchain** (`rustup` with stable `cargo` on PATH) on the install host.

## Authentication

PyPI Trusted Publishing via GitHub OIDC - no API tokens, no secrets in the repo. Each publish job runs under the `pypi` GitHub deployment environment, which is bound to the trusted-publisher configuration on pypi.org for both projects.

## Releasing

1. Make sure `main` is green in CI.
2. Create and push a tag:
   ```bash
   git tag v0.10.2
   git push origin v0.10.2
   ```
3. The `release.yml` workflow triggers automatically and serializes:

   | Step | Package | What it does |
   |------|---------|-------------|
   | `verify-ci-green` | - | Confirms ci.yml passed for the tagged SHA |
   | `publish-accel` | `khora-accel` | `maturin sdist` → publish to PyPI |
   | `publish-khora` | `khora` | Pins `khora-accel == ${tag}` in pyproject.toml, then `python -m build` (wheel + sdist) → publish to PyPI |

   Publish order is **accel first, then khora** so that the moment `khora==X.Y.Z` appears on PyPI, its `khora-accel==X.Y.Z` dependency is already resolvable.

### Manual Publish

The workflow supports `workflow_dispatch` for manual re-runs from the GitHub Actions UI.

## Version Lockstep

khora's `pyproject.toml` declares `khora-accel >= X.Y.Z` (a loose floor) in the `rust` extra. At release time, the `Pin khora-accel to release version` step in `release.yml` rewrites this to `khora-accel == ${tag}` before building the khora wheel. Consequence: the published khora wheel always hard-pins khora-accel to the exact same version.

If you ever need to break this lockstep (e.g. ship a khora hotfix that uses an older khora-accel), edit pyproject.toml on the release branch and remove or override the sed step in release.yml for that release.

## Verification

After the workflow completes, confirm packages are visible:

```bash
curl -s https://pypi.org/pypi/khora/json | jq '.releases | keys[-3:]'
curl -s https://pypi.org/pypi/khora-accel/json | jq '.releases | keys[-3:]'
```

Or open https://pypi.org/project/khora/${VERSION}/ in a browser.

## Dev Releases

There are no automatic dev/pre-release publishes. Only tag pushes publish to PyPI. To consume in-flight changes from `main` before a tag, install from git:

```bash
pip install git+https://github.com/DeytaHQ/khora.git@main
```

This requires repo access for as long as the repository is private.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Trusted publishing exchange failure` | Environment name, workflow filename, or repo owner doesn't match the pypi.org trusted-publisher config | Check the per-project Publishing settings on pypi.org match `release.yml`, `pypi`, `DeytaHQ`, `khora` |
| `File already exists` | Re-running release for a version that already published | Bump to the next patch; PyPI does not allow overwriting |
| Version shows `0.0.0` or `dev` | No git tags reachable from HEAD | `git push origin --tags`; verify `hatch-vcs` sees the tag (`uv run python -c "import khora; print(khora.__version__)"`) |
| khora-accel sdist install fails for a user | No Rust toolchain on the install host | Install `rustup`, ensure `cargo` is on `PATH`, retry; long-term we may add a prebuilt wheel matrix |
