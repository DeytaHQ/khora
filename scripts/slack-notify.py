#!/usr/bin/env python3
"""Deterministic Slack notification script for PR lifecycle events.

Posts messages to the #pull-requests channel via the Slack API.
Uses only stdlib — no pip dependencies required.

Required env: SLACK_BOT_TOKEN (bot token; scopes: chat:write, reactions:write)

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Add Bot Token Scopes: chat:write, reactions:write
  3. Install to workspace and copy the Bot User OAuth Token (xoxb-...)
  4. export SLACK_BOT_TOKEN=xoxb-...

Dedup strategy: After posting a top-level message, the script writes the Slack
message `ts` as a hidden HTML comment on the GitHub PR. On subsequent runs it
reads that comment to find the existing thread. This avoids needing
search:read scope (which requires a user token).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

CHANNEL_ID = "C0ALZNY9UBF"  # #pull-requests
SLACK_API = "https://slack.com/api"
MARKER_RE = re.compile(r"<!-- slack-notify:([A-Z0-9]+):(\d+\.\d+) -->")


# ── Slack helpers ────────────────────────────────────────────────────────────


def slack_request(method: str, payload: dict, token: str) -> dict:
    """Call a Slack Web API method and return the parsed JSON response."""
    url = f"{SLACK_API}/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def post_message(text: str, token: str, *, thread_ts: str | None = None) -> dict:
    """Post a message to #pull-requests, optionally as a thread reply."""
    payload: dict = {"channel": CHANNEL_ID, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    result = slack_request("chat.postMessage", payload, token)
    if not result.get("ok"):
        print(f"Slack API error: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return result


def add_reaction(name: str, ts: str, token: str) -> None:
    """Add a reaction to a message. Silently ignores already_reacted errors."""
    result = slack_request(
        "reactions.add",
        {"channel": CHANNEL_ID, "timestamp": ts, "name": name},
        token,
    )
    if not result.get("ok") and result.get("error") != "already_reacted":
        print(f"Slack reactions.add error: {result.get('error', 'unknown')}", file=sys.stderr)


# ── PR comment marker (dedup) ───────────────────────────────────────────────


def find_marker(pr_url: str) -> tuple[str, str] | None:
    """Read PR comments and return (channel_id, ts) if a marker exists."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", pr_url, "--json", "comments", "--jq", ".comments[].body"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None

    for line in result.stdout.splitlines():
        m = MARKER_RE.search(line)
        if m:
            return m.group(1), m.group(2)
    return None


def write_marker(pr_url: str, channel_id: str, ts: str) -> None:
    """Write the Slack message ts as a hidden HTML comment on the PR."""
    body = f"<!-- slack-notify:{channel_id}:{ts} -->"
    try:
        subprocess.run(
            ["gh", "pr", "comment", pr_url, "--body", body],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Failed to write PR marker comment: {exc}", file=sys.stderr)


# ── Message builders ─────────────────────────────────────────────────────────


def build_pr_message(args: argparse.Namespace) -> str:
    """Build the top-level PR notification message."""
    linear_url = f"https://linear.app/issue/{args.ticket_id}"
    draft_prefix = ":construction: Draft — " if args.draft else ""
    lines = [
        f"{draft_prefix}{args.repo} PR <{args.pr_url}|{args.title}>",
        f"Linear: <{linear_url}|{args.ticket_id}>",
    ]
    if args.author:
        lines.append(f"Author: {args.author}")
    lines += ["", args.summary]
    return "\n".join(lines)


def build_pr_thread(args: argparse.Namespace) -> str | None:
    """Build the optional PR thread reply with diff stats and commits."""
    parts: list[str] = []
    if args.diff_stat:
        parts.append(f"*Changes:* {args.diff_stat}")
    if args.commits:
        lines = [line.strip() for line in args.commits.strip().splitlines() if line.strip()]
        commit_bullets = []
        for line in lines:
            sha, _, msg = line.partition(" ")
            commit_bullets.append(f"\u2022 `{sha}` {msg}")
        parts.append("*Commits:*\n" + "\n".join(commit_bullets))
    if args.reviewers:
        parts.append(f"*Reviewers:* {args.reviewers}")
    else:
        parts.append("*Reviewers:* None requested")
    if args.target_branch:
        parts.append(f"*Target:* `{args.target_branch}`")
    if args.labels:
        parts.append(f"*Labels:* {args.labels}")
    return "\n\n".join(parts) if parts else None


def build_pr_update_thread(args: argparse.Namespace) -> str:
    """Build the dedup update thread reply when a PR message already exists."""
    parts: list[str] = ["*PR updated*"]
    if args.diff_stat:
        parts.append(f"*Changes:* {args.diff_stat}")
    if args.commits:
        lines = [line.strip() for line in args.commits.strip().splitlines() if line.strip()]
        commit_bullets = []
        for line in lines:
            sha, _, msg = line.partition(" ")
            commit_bullets.append(f"\u2022 `{sha}` {msg}")
        parts.append("*New commits:*\n" + "\n".join(commit_bullets))
    if args.summary:
        parts.append(f"*Summary:*\n{args.summary}")
    return "\n\n".join(parts)


def build_merge_thread() -> str:
    """Build the short merge thread reply."""
    return "*Merged* :white_check_mark:"


def build_ci_failure_thread(failed_checks: str) -> str:
    """Build the CI failure thread reply."""
    return f"*CI failed* :x:\nFailed checks: {failed_checks}"


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post Slack notifications for PR events")
    parser.add_argument("--mode", required=True, choices=["pr", "merge", "ci-failure", "approved"])
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--pr-url", required=True, help="Pull request URL")
    parser.add_argument("--title", required=True, help="PR title")
    parser.add_argument("--ticket-id", required=True, help="Linear ticket ID")
    parser.add_argument("--summary", help="One-line summary (required for pr mode)")
    parser.add_argument("--diff-stat", help='e.g. "5 files changed, +120 / -30"')
    parser.add_argument("--commits", help="Newline-separated 'sha message' list")
    parser.add_argument("--target-branch", help="Base branch name")
    parser.add_argument("--failed-checks", help="Comma-separated check names (required for ci-failure mode)")
    parser.add_argument("--reviewers", help="Comma-separated list of requested reviewers")
    parser.add_argument("--labels", help="Comma-separated PR labels")
    parser.add_argument("--draft", action="store_true", help="PR is a draft")
    parser.add_argument("--author", help="PR author (GitHub username)")

    args = parser.parse_args(argv)

    if args.mode == "pr" and not args.summary:
        parser.error("--summary is required for pr mode")
    if args.mode == "ci-failure" and not args.failed_checks:
        parser.error("--failed-checks is required for ci-failure mode")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Check token
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        print("SLACK_BOT_TOKEN not set — skipping Slack notification")
        sys.exit(0)

    # Look up existing thread via PR comment marker
    marker = find_marker(args.pr_url)
    thread_ts: str | None = marker[1] if marker else None

    if args.mode == "pr":
        if thread_ts:
            # PR already posted — post compact update in thread
            post_message(build_pr_update_thread(args), token, thread_ts=thread_ts)
        else:
            # New PR — post top-level + thread detail
            result = post_message(build_pr_message(args), token)
            new_ts = result.get("ts")
            if new_ts:
                write_marker(args.pr_url, CHANNEL_ID, new_ts)
                thread_text = build_pr_thread(args)
                if thread_text:
                    post_message(thread_text, token, thread_ts=new_ts)

    elif args.mode == "merge":
        if thread_ts:
            post_message(build_merge_thread(), token, thread_ts=thread_ts)
            add_reaction("git-merged", thread_ts, token)
        else:
            # No prior message — post top-level merge notification
            linear_url = f"https://linear.app/issue/{args.ticket_id}"
            text = (
                f"{args.repo} PR <{args.pr_url}|{args.title}> :white_check_mark:\n"
                f"Linear: <{linear_url}|{args.ticket_id}>\n"
                f"\n"
                f"Merged and branch cleaned up."
            )
            result = post_message(text, token)
            new_ts = result.get("ts")
            if new_ts:
                write_marker(args.pr_url, CHANNEL_ID, new_ts)
                add_reaction("git-merged", new_ts, token)

    elif args.mode == "approved":
        if thread_ts:
            add_reaction("git-approved", thread_ts, token)
        else:
            # No prior message — skip silently
            print("No existing Slack message found — skipping approved reaction")

    elif args.mode == "ci-failure":
        if thread_ts:
            post_message(build_ci_failure_thread(args.failed_checks), token, thread_ts=thread_ts)
        else:
            # No prior message — post top-level CI failure
            linear_url = f"https://linear.app/issue/{args.ticket_id}"
            text = (
                f"{args.repo} PR <{args.pr_url}|{args.title}> :x:\n"
                f"Linear: <{linear_url}|{args.ticket_id}>\n"
                f"\n"
                f"CI failed: {args.failed_checks}"
            )
            result = post_message(text, token)
            new_ts = result.get("ts")
            if new_ts:
                write_marker(args.pr_url, CHANNEL_ID, new_ts)

    print(f"Slack notification sent ({args.mode})")


if __name__ == "__main__":
    main()
