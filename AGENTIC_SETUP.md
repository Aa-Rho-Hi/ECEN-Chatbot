# Agentic Issue → Fix → PR Workflow

This repo uses the official [Claude Code GitHub Action](https://github.com/anthropics/claude-code-action)
to turn bug reports into reviewed fixes. **No fix is ever merged automatically —
a human approves and merges every PR.**

## The loop

```
Tester finds a bug
      │
      ▼
Opens a GitHub Issue (Bug report template)  ──or──  comments "@claude ..."
      │
      ▼  (GitHub Actions: .github/workflows/claude.yml)
Claude VERIFIES the issue against the code
      │
      ├─ can't confirm ─▶ comments on the issue, asks for info, stops (no code change)
      │
      └─ confirmed ─▶ creates branch  claude/fix-issue-<n>
                         │
                         ▼
                     implements smallest correct fix
                         │
                         ▼
                     opens a Pull Request
                         │
                         ▼  (.github/workflows/claude-code-review.yml)
                     Claude posts an automated code review (read-only)
                         │
                         ▼
                 ┌─────────────────────────────┐
                 │  HUMAN reviews + clicks Merge │  ◀── the only merge gate
                 └─────────────────────────────┘
```

## One-time setup

1. **Install the Claude GitHub App** on the `Aa-Rho-Hi/ECEN-Chatbot` repo.
   Easiest path: from a terminal with Claude Code installed, run `/install-github-app`,
   or install it from https://github.com/apps/claude and grant it this repo.

2. **Add the API key secret.** Repo → Settings → Secrets and variables → Actions →
   New repository secret:
   - Name: `ANTHROPIC_API_KEY`
   - Value: your Anthropic API key (from https://console.anthropic.com)

3. **Protect `main` so nothing merges without you.** Repo → Settings → Branches →
   Add branch ruleset / protection rule for `main`:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Do not allow bypassing the above settings
   - (Optional) ✅ Require status checks to pass
   This guarantees Claude's `claude/fix-issue-*` branches can only land via a PR you approve.

## Files in this setup
| File | Role |
|---|---|
| `.github/workflows/claude.yml` | Triggers on issues / `@claude` mentions; verifies + opens fix PRs |
| `.github/workflows/claude-code-review.yml` | Auto-review on every PR (read-only) |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured bug report that feeds the loop |
| `CLAUDE.md` | Project context + verify commands the agent reads |

## Customizing
- **Trigger phrase:** add `trigger_phrase: "/claude"` under the action's `with:`.
- **Model / turns / allowed tools:** edit `claude_args` in `claude.yml`.
- **Scope reviews:** uncomment the `paths:` filter in `claude-code-review.yml`.
- **Cost control:** the action runs on GitHub-hosted runners and bills your
  Anthropic API key per run; `--max-turns` caps work per invocation.

## Security notes
- `.env` and all secrets are git-ignored — never commit them.
- The action only has the GitHub permissions declared in each workflow's
  `permissions:` block. `claude.yml` can write branches/PRs but branch protection
  stops it from merging to `main`.
