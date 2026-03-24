# Release Process

## Tag Naming

Tags follow semantic versioning: `v{major}.{minor}.{patch}` (e.g., `v0.5.0`).

The `v` prefix is required — it triggers the publish workflows.

## Versioning

Versions are derived automatically from git tags — no files need manual version bumps.

| Package | How version is set |
|---------|-------------------|
| `khora` | `hatch-vcs` reads the most recent git tag at build time |
| `khora-accel` | `publish-accel.yml` extracts the tag and stamps it into `Cargo.toml` before maturin builds |

At runtime, `khora.__version__` reads the installed package version via `importlib.metadata`.

In development (no tag on current commit), the version will be something like `0.5.0.dev3+gabc1234`.

## Publishing

1. Create and push a tag:
   ```bash
   git tag v0.6.0
   git push origin v0.6.0
   ```
2. Two workflows trigger automatically:

| Workflow | Package | What it does |
|----------|---------|-------------|
| `publish.yml` | `khora` | Builds a pure Python wheel via `hatch-vcs` and publishes to CodeArtifact |
| `publish-accel.yml` | `khora-accel` | Builds native wheels for 4 platforms (Linux x64/ARM64, macOS x64/ARM64) and publishes to CodeArtifact |

Both workflows use OIDC authentication — no secrets required.

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
| Version shows `0.0.0` or `dev` | No git tags reachable from HEAD | Ensure you've pushed tags: `git push origin --tags` |
| khora-accel build fails on one platform | Platform-specific compilation issue | Check build logs; other platforms still complete |
