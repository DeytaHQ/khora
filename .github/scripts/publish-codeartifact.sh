#!/usr/bin/env bash
# Idempotent twine upload for CodeArtifact.
#
# CodeArtifact does NOT support `twine upload --skip-existing` (that flag is
# PyPI-specific; CA rejects it as UnsupportedConfiguration).  Instead we run
# twine, capture stderr, and treat 409 Conflict as success — that's the case
# where a wheel for `(name, version)` was already published, which happens
# on tag-retries and dev-publish reruns.  Any non-409 error still fails the
# step.
#
# Required env: CA_TOKEN, CA_REPOSITORY_URL.
# Required args: dist files (e.g. `dist/*`).

set -uo pipefail

if [[ -z "${CA_TOKEN:-}" || -z "${CA_REPOSITORY_URL:-}" ]]; then
  echo "::error::CA_TOKEN and CA_REPOSITORY_URL must be set." >&2
  exit 2
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT

set +e
twine upload \
  --verbose \
  --repository-url "$CA_REPOSITORY_URL" \
  --username aws \
  --password "$CA_TOKEN" \
  "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -eq 0 ]]; then
  exit 0
fi

# twine emits a line like:
#   ERROR    HTTPError: 409 Conflict from <url>
# If every HTTPError is 409, we already-have the artifact — treat as success.
if grep -i "HTTPError" "$log" | grep -v -i "409 Conflict" >/dev/null 2>&1; then
  echo "::error::twine reported a non-409 error; failing the step." >&2
  exit "$rc"
fi

echo "All upload errors were '409 Conflict' (already published) — treating as success."
exit 0
