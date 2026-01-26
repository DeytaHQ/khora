#!/usr/bin/env python3
"""
Pre-commit hook to update REPOMIX.md before commits.

This script runs repomix to generate an up-to-date REPOMIX.md file
containing the repository documentation for AI-assisted development.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Generate REPOMIX.md using repomix."""
    project_root = Path(__file__).parent.parent

    # Check if repomix.config.json exists
    config_file = project_root / "repomix.config.json"
    if not config_file.exists():
        print("repomix.config.json not found, skipping REPOMIX.md generation")
        return 0

    print("Generating REPOMIX.md...")

    try:
        # Run repomix using npx
        result = subprocess.run(
            ["npx", "-y", "repomix@latest", "--config", "repomix.config.json"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            print(f"repomix failed with exit code {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            # Don't fail the commit if repomix fails - it's a nice-to-have
            return 0

        repomix_file = project_root / "REPOMIX.md"
        if repomix_file.exists():
            print("REPOMIX.md generated successfully")

            # Check if file has changes and stage it
            check_result = subprocess.run(
                ["git", "diff", "--quiet", "REPOMIX.md"],
                cwd=project_root,
                capture_output=True,
            )

            if check_result.returncode != 0:
                # File has changes, stage it
                subprocess.run(
                    ["git", "add", "REPOMIX.md"],
                    cwd=project_root,
                    check=True,
                )
                print("REPOMIX.md staged for commit")
        else:
            print("REPOMIX.md was not created")

    except subprocess.TimeoutExpired:
        print("repomix timed out, skipping")
        return 0
    except FileNotFoundError:
        print("npx not found. Install Node.js to enable REPOMIX.md generation.")
        return 0
    except subprocess.SubprocessError as e:
        print(f"Error running repomix: {e}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
