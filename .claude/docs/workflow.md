# Workflow Reference

<!-- Shared workflow doc installed by TTOJ at .claude/docs/workflow.md.      -->
<!-- This file is auto-deployed; edits will be overwritten on ttoj update.   -->
<!-- Project-specific overrides belong in CLAUDE.md.                         -->

## Default Branch Detection

Different repos use different default branches (`main`, `staging`, etc.). **Always detect the repo's default branch dynamically** rather than assuming a specific branch name:

```bash
DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)
```

Use `$DEFAULT_BRANCH` wherever this document refers to "the default branch." This ensures the workflow works regardless of whether the repo targets `main`, `staging`, or any other branch.

## Workflow Lifecycle

Every feature, bugfix, or task follows: **Linear ticket → branch → implement → PR → done**.

If your project uses Linear (has `project.linear_team` in `.ttoj.toml`):

### Starting work
1. Search Linear for an existing ticket matching the task. If none exists, create one.
2. Assign yourself (or the requesting user) to the ticket.
3. Move the ticket to **In Progress**.
4. Create a branch from the repo's default branch following the branch naming convention below. For hotfixes, branch from `main`.
5. Begin implementation.

### During work
- Commit early and often with descriptive messages.
- Keep the Linear ticket updated if scope changes or blockers arise.
- If you discover sub-tasks, create child issues in Linear.

### Finishing work
1. Run any project-defined lint/format/test commands.
2. Commit all changes with a clear message.
3. Push the branch and create a PR following PR standards below. PRs target the default branch; only hotfixes target `main`.
4. Add the PR link as a comment on the Linear ticket.
5. The ticket stays **In Progress** until the PR is merged, then move to **Done**.

## Git Conventions

### Branch naming
Format: `{initials}/{TICKET-ID}-{kebab-description}`
Examples: `nn/DYT-42-add-auth`, `ms/DYT-108-fix-pagination`

The `initials` and `TICKET-ID` are required. Every branch must trace back to a Linear ticket. TTOJ stores the engineer-specific branch prefix in `.ttoj.toml`.

### Commit messages
- Use imperative mood: "Add feature", not "Added feature"
- First line: concise summary under 72 characters
- Format: `{TICKET-ID}: {description}` (e.g., `DYT-42: Add auth middleware`)
- Body (optional): explain *why*, not *what*
- Reference the Linear ticket ID when relevant

### Branching strategy
- `main` = production. The repo's default branch is the PR target (detect via `gh repo view`).
- Feature branches are created from the default branch. For hotfixes, branch from `main`.
- PRs **always** target the default branch, unless it's a hotfix.
- Hotfixes: branch from `main`, PR targets `main` (production deploy).

### Rules
- Never force-push to `main`, the default branch, or shared branches.
- Rebase feature branches on the PR target branch (the default branch, or `main` for hotfixes) before opening a PR when possible.
- Delete branches after merge.

## Linear Integration

Use the Linear MCP tools (`list_issues`, `save_issue`, `save_comment`, etc.) for all ticket operations.

### Status transitions
- **Backlog** → **Todo** → **In Progress** → **Done**
- Other states: **Canceled**, **Duplicate**
- Move to In Progress when you start working.
- Move to Done only after the PR is merged.

### Linking
- Always add the PR URL as a comment on the Linear ticket.
- Include the ticket ID (e.g., `DYT-42`) in the PR title.

## PR Standards

### Title format
`TICKET-ID: Short imperative description` (e.g., `DYT-42: Add JWT authentication`)

### Body structure
```
## Summary
- Bullet points describing what changed and why

## Test plan
- How to verify the changes work
```

- PRs target the repo's default branch; hotfix PRs target `main`.
- Keep it concise. The code should speak for itself.

## Multi-Agent Coordination

When working as part of an agent team:

- Claim tasks explicitly via TaskUpdate before starting work.
- Never modify files owned by another agent without coordinating first.
- Avoid editing the same file concurrently. If unavoidable, coordinate line ranges.
- If two agents disagree on approach, escalate to the team lead.

### Agent Identity

All Claude Code agents share the same Linear account, so assignment alone cannot distinguish which agent is working on which issue. **Every agent session MUST generate a unique session ID at startup** and include it in all Linear comments.

**At the start of every session**, generate a 6-character hex ID (e.g., the first 6 characters of a UUID). Use this ID consistently in every Linear comment you post.

Format: `` `agent-<6-char-hex>` `` (e.g., `agent-a1b2c3`)

### Claim Before Coding

If your project uses Linear (has `project.linear_team` in `.ttoj.toml`), post a Linear comment with your session ID before writing any code:

```
🤖 Agent `<session-id>` starting work. Branch: `<branch-name>`
```

This is the only reliable way to signal ownership when all agents share the same Linear account. Moving the ticket to In Progress alone is not sufficient — another agent may do the same concurrently.

### Worktree Usage

**Never use `git checkout` to switch branches in the main repo.** Always create a worktree:

```bash
DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)
git worktree add .claude/worktrees/{TICKET-ID}-short-desc -b {initials}/{TICKET-ID}-short-desc "$DEFAULT_BRANCH"
cd .claude/worktrees/{TICKET-ID}-short-desc
# Run project-specific install (e.g., bun install, npm install)
# Re-deploy TTOJ surface (.ttoj/ is not shared across worktrees):
uv run ttoj install -p .
```

Rules:
- Each agent gets its own worktree directory under `.claude/worktrees/`
- Run dependency install and `ttoj install` in a new worktree before building or testing
- Clean up worktrees after PRs are merged: `git worktree remove .claude/worktrees/{TICKET-ID}-short-desc`

## AI Changelog

After every completed task, append an entry to `docs/AI_CHANGELOG.md`. Create the file if it doesn't exist.

Format: `- YYYY-MM-DD: TICKET-ID: Brief description of change` (one line, max 72 chars). Append at bottom; never edit existing entries.

## Pre-Submission Checklist

Before creating a PR, verify:

- [ ] Lint/format passes with zero errors
- [ ] Tests pass (all relevant tests)
- [ ] No `any` types in changed files (TypeScript projects)
- [ ] No secrets, API keys, or credentials in code (use env vars)
- [ ] No `console.log` left in production code (use structured logger)
- [ ] PR created with description and Linear issue link
- [ ] Linear issue updated (status, comment with PR link)

Projects may define additional checklist items in their CLAUDE.md.

## PRD & ADR

Major features require a PRD in `docs/prds/`. Significant architectural decisions require an ADR in `docs/adrs/`. Use `/plan:create` and `/plan:adr` to scaffold these documents. See the TTOJ repo for templates.

## Agent Teams

Use `/team:feature`, `/team:bugfix`, or `/team:review` for purpose-built teams. Default to single-agent work unless the task clearly benefits from parallelism. Role definitions are in the TTOJ repo under `content/templates/team-profiles/` and are available via the `/team:*` commands.

## Slack Integration

Commands post to team-wide default channels (deployed via TTOJ):
- `/done` → `#pull-requests` — PR opened notification with detailed context in thread.
- `/automerge` → `#pull-requests` — merge success or CI failure.
- `/plan:create` → `#engineering` — optionally shares a PRD summary with goals and requirements in thread.
- `/slack-notify` — sends an ad-hoc message to any channel or user.

Default channels are configured per-project via TTOJ. Commands never fail due to Slack errors.

Codex uses the same `slack-notify.py` backend through its own installed skills (`slack-pr-notify`, `slack-merge-notify`, `slack-prd-notify`, `slack-notify`). The notification behavior is identical across both hosts.

## Orchestrated Workflow

The recommended command sequence for feature development:

1. `/plan:create` — scaffold a PRD (optionally share on Slack)
2. `/plan:adr` — document key decisions (if needed)
3. `/plan:expand` — break PRD into Linear tickets
4. `/workflow` — pick a ticket, create branch, start work
5. Implement (solo or `/team:feature`)
6. `/done` — commit, PR, update Linear, notify Slack
7. `/audit` — review the PR

## Tool Preferences

- **Documentation lookup**: Use Context7 MCP for up-to-date library docs.
- **Linear operations**: Use Linear MCP tools (list_issues, save_issue, save_comment, etc.).
- **GitHub operations**: Use `gh` CLI for PRs, issues, and repo operations.
- **File operations**: Prefer dedicated Claude Code tools (Read, Write, Edit, Grep, Glob) over shell equivalents.
- **Web research**: Use WebSearch/WebFetch for current information beyond training data.

## Team Configuration

Default team profiles are deployed to `.ttoj/templates/team-profiles/` and referenced by the `/team:*` commands. To customize team composition for this project, edit the deployed profiles or create project-specific overrides in `.claude/commands/team/`.
