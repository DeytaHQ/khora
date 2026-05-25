"""Allow running as python -m khora.

Khora is a library. The CLI tooling lives in the separate khora-cli package.
"""

if __name__ == "__main__":
    raise SystemExit("Khora is a library and does not provide a CLI. Use khora-cli instead.")