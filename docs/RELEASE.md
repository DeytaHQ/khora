# Release Process

## Tag Naming

Tags follow semantic versioning: `v{major}.{minor}.{patch}` (e.g., `v0.5.0`).

The `v` prefix is required -- it triggers the publish workflows.

## Version Bumps

Before tagging a release, update the version in **all four files** and regenerate lockfiles:

1. `pyproject.toml` -- khora version
2. `src/khora/__init__.py` -- `__version__`
3. `rust/khora-accel/Cargo.toml` -- khora-accel version
4. `rust/khora-accel/pyproject.toml` -- khora-accel version
5. Run `uv lock` to regenerate the Python lockfile
6. Run `cargo generate-lockfile` in `rust/khora-accel/` to regenerate the Cargo lockfile

Commit the version bump and lockfile changes before tagging.

## Publishing

Pushing a `v*` tag triggers two GitHub Actions workflows:

| Workflow | Package | What it does |
|----------|---------|-------------|
| `publish.yml` | `khora` | Builds a pure Python wheel and publishes to CodeArtifact |
| `publish-accel.yml` | `khora-accel` | Builds native wheels for 4 platforms (Linux x64/ARM64, macOS x64/ARM64) and publishes to CodeArtifact |

Both workflows use OIDC authentication -- no secrets required.

### Steps

1. Bump versions in all four files (see above).
2. Commit: `git commit -m "Bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main --tags`
5. Both workflows trigger automatically.

### Manual Publish

Both workflows support `workflow_dispatch` for manual re-runs from the GitHub Actions UI.

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

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| OIDC credential error | IAM role trust policy misconfigured | Check `github-actions-codeartifact-publish` role in AWS |
| Twine upload 403 | Expired or invalid CodeArtifact token | Re-run the workflow; tokens are generated per-run |
| Version mismatch | Tag doesn't match pyproject.toml version | Ensure all 4 version files match the tag |
| khora-accel build fails on one platform | Platform-specific compilation issue | Check build logs; other platforms still complete |
