# Release Process

## Tag Naming

Tags follow semantic versioning: `v{major}.{minor}.{patch}` (e.g., `v0.5.0`).

The `v` prefix is required — it triggers the publish workflows.

## Versioning

Versions are derived automatically from git tags — no files need manual version bumps.

| Package | How version is set |
|---------|-------------------|
| `khora` | `hatch-vcs` reads the most recent git tag at build time |
| `khora-accel` | `release.yml` extracts the tag and stamps it into `Cargo.toml` before maturin builds |

At runtime, `khora.__version__` reads the installed package version via `importlib.metadata`.

In development (no tag on current commit), the version will be something like `0.5.5.dev3`.

## Dev Releases (automatic)

Every merge to `main` publishes a dev release to CodeArtifact via `ci.yml`:

| Package | Version example | How |
|---------|----------------|-----|
| `khora` | `0.5.5.dev14` | `hatch-vcs` with `local_scheme = "no-local-version"` in pyproject.toml |
| `khora-accel` | `0.5.5-dev.14` (semver) → `0.5.5.dev14` (wheel) | `git describe` → stamp `Cargo.toml` → maturin |

Dev versions are PEP 440 pre-releases. Downstream projects that pin `>=X.Y.Z.dev0` will pick them up.

## Stable Releases (tag-triggered)

1. Create and push a tag:
   ```bash
   git tag v0.6.0
   git push origin v0.6.0
   ```
2. The `release.yml` workflow triggers automatically:

| Job | Package | What it does |
|----------|---------|-------------|
| `publish-khora` | `khora` | Builds a pure Python wheel via `hatch-vcs` and publishes to CodeArtifact |
| `publish-accel` | `khora-accel` | Builds native wheels for 3 platforms (Linux x64/ARM64, macOS ARM64) and publishes to CodeArtifact |

The workflow uses OIDC authentication — no secrets required.

### Manual Publish

The workflow supports `workflow_dispatch` for manual re-runs from the GitHub Actions UI.

## Verification

After the workflows complete, confirm packages are available:

```bash
aws codeartifact list-package-versions \
  --domain deyta \
  --repository packages \
  --format pypi \
  --package khora

aws codeartifact list-package-versions \
  --domain deyta \
  --repository packages \
  --format pypi \
  --package khora-accel
```

## Downstream Projects (always-latest)

For internal projects like `khora-benchmarks` that should always track the latest dev release, add this to their `pyproject.toml`:

```toml
[project]
dependencies = [
    "khora>=0.5.0.dev0",        # .dev0 opts into pre-releases
    "khora-accel>=0.5.0.dev0",
]

[tool.uv]
upgrade-package = ["khora", "khora-accel"]
```

The `>=X.Y.Z.dev0` specifier tells uv to accept dev/pre-release versions for these packages (under the default `if-necessary-or-explicit` prerelease strategy).

The `upgrade-package` setting makes every `uv sync` re-resolve these packages to the latest available version, ignoring the lockfile pin. No CLI flags needed — plain `uv sync` does the right thing.

> **Note:** This will update `uv.lock` on every `uv sync` when a newer dev version is available. With frequent merges to main, expect lockfile changes daily. This is intentional — commit the updates as part of normal workflow. If lockfile churn is disruptive, pin to a specific dev version instead (e.g., `khora==0.5.5.dev14`).

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| OIDC credential error | IAM role trust policy misconfigured | Check `github-actions-codeartifact-publish` role in AWS |
| Twine upload 403 | Expired or invalid CodeArtifact token | Re-run the workflow; tokens are generated per-run |
| Version shows `0.0.0` or `dev` | No git tags reachable from HEAD | Ensure you've pushed tags: `git push origin --tags` |
| khora-accel build fails on one platform | Platform-specific compilation issue | Check build logs; other platforms still complete |
