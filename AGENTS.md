# AGENTS.md — khora

<!-- This is the project-specific AGENTS.md. Copy it into your repo root as AGENTS.md -->
<!-- and replace each placeholder section with project-specific guidance.              -->
<!-- Machine-readable workflow config lives in .ttoj.toml. Keep this file human-facing. -->

## Project Overview

<!-- Replace with 2-3 sentences describing what this project does, who it serves, -->
<!-- and the context an engineer or agent needs before making changes.            -->

## Tech Stack

<!-- Replace with the actual stack. Add or remove lines as needed. -->

- **Language:** {e.g., TypeScript, Python, Go}
- **Framework:** {e.g., Next.js, FastAPI, Gin}
- **Database:** {e.g., PostgreSQL, DynamoDB}
- **ORM / Query layer:** {e.g., Prisma, SQLAlchemy, raw SQL}
- **Package manager:** {e.g., pnpm, uv, go modules}
- **Infra / Hosting:** {e.g., Vercel, AWS, Fly.io}

## Architecture

<!-- Replace with a brief map of key directories and modules. Example: -->
<!-- src/api/    — route handlers                                      -->
<!-- src/lib/    — shared utilities and domain logic                    -->
<!-- src/db/     — database schema, migrations, queries                -->
<!-- tests/      — unit and integration tests                          -->

## Commands

<!-- Replace each placeholder with the actual command for this project. -->
<!-- Remove any that don't apply.                                       -->

| Task       | Command              |
|------------|----------------------|
| Lint       | `{lint command}`     |
| Format     | `{format command}`   |
| Test       | `{test command}`     |
| Build      | `{build command}`    |
| Dev server | `{dev command}`      |
| Typecheck  | `{typecheck command}`|

## Codex Settings

Run `uv run ttoj install -p /path/to/your-project` to deploy the repo-shared Codex surface: `AGENTS.md` plus the shared `.agents/skills/` set. TTOJ installs this alongside the Claude repo surface by default, but this v1 adapter does not generate `.codex/config.toml`, approvals, or other machine-local Codex rules.

- Add any project-specific Codex configuration manually if your team needs it.
- Commit `AGENTS.md` and the installed `.agents/skills/` directories to version control if your repo standard is to share agent instructions.
- Use `uv run ttoj update -p /path/to/your-project` to refresh the shared Codex assets later.
- Advanced team-workflow skills are installed too, but they stay opt-in. Codex should use them only when a user explicitly asks for delegation, subagents, or parallel agent work.

## Workflow Lifecycle

Every feature, bugfix, or task follows: **Linear ticket → branch → implement → PR → done**.

### Starting work
1. Search Linear for an existing ticket matching the task. If none exists, create one.
2. Assign yourself (or the requesting user) to the ticket.
3. Move the ticket to **In Progress**.
4. Create a branch from `staging` (default) following the branch naming convention below. For hotfixes, branch from `main`.
5. Begin implementation.

### Finishing work
1. Run the project-defined lint, format, and test commands.
2. Commit all changes with a clear message.
3. Push the branch and open a PR to `staging` by default. Only hotfixes target `main`.
4. Add the PR link as a comment on the Linear ticket.
5. Move the ticket to **Done** only after the PR is merged.

## Git Conventions

### Branch naming
Format: `{initials}/{TICKET-ID}-{kebab-description}`
Examples: `nn/DYT-42-add-auth`, `ms/DYT-108-fix-pagination`

TTOJ stores the engineer-specific branch prefix in `.ttoj.toml`.

### Commit messages
- Use imperative mood: "Add feature", not "Added feature"
- First line: concise summary under 72 characters
- Format: `{TICKET-ID}: {description}` (e.g., `DYT-42: Add auth middleware`)

## Linear Integration

Use the Linear MCP tools for all ticket operations. The Codex workflow skills under `.agents/skills/` are the project-level extension points for host-specific Linear flows.

## PR Standards

### Title format
`TICKET-ID: Short imperative description`

### Body structure
```md
## Summary
- Bullet points describing what changed and why

## Test plan
- How to verify the changes work
```

## Tool Preferences

- **Documentation lookup**: Prefer primary docs for libraries and frameworks.
- **Linear operations**: Use Linear MCP tools for ticket state, comments, and links.
- **GitHub operations**: Use `gh` CLI for PRs, issues, and repo operations.
- **File operations**: Prefer precise file edits over broad destructive rewrites.

## Codex Skills

TTOJ installs shared Codex skills under `.agents/skills/`.

- `plan-create` drafts PRDs from `.ttoj/templates/prd.md`
- `plan-expand` turns PRDs into reviewable Linear task plans
- `plan-adr` drafts ADRs from `.ttoj/templates/adr.md`
- `review`, `audit`, `claude-audit`, and `security-audit` contain concrete review and audit guidance
- `team-feature`, `team-bugfix`, and `team-review` orchestrate opt-in subagent workflows by reusing `.ttoj/templates/team-profiles/*.md`

## Codex Team Workflows

Codex team workflows are an explicit escalation path for larger tasks, not the default way to work.

- Only use `team-feature`, `team-bugfix`, or `team-review` when the user explicitly asks for delegation, subagents, or parallel work.
- Reuse the deployed team profiles in `.ttoj/templates/team-profiles/` when selecting roles and defining ownership.
- Treat Codex subagents as isolated workspaces with disjoint write scopes. Do not assume Claude-style per-worker git branches unless the user specifically wants that branch topology.

Current non-parity with Claude:

- Codex has no `/team:*` slash commands; the equivalent entry points are the installed Codex skills.
- Codex team coordination is manual through subagent tools rather than a built-in team runtime.
- Worker isolation is typically via forked workspaces, not a shared subtask-branch tree.
- `claude-audit` is the Codex-side bridge for a Claude second opinion. It shells out to the local `claude` CLI with pre-fetched PR context instead of assuming Claude slash-command parity inside Codex.

## AI Changelog

After every completed task, append an entry to `docs/AI_CHANGELOG.md`. Create the file if it doesn't exist.

### Format

```md
- YYYY-MM-DD: TICKET-ID: Brief description of change
```
