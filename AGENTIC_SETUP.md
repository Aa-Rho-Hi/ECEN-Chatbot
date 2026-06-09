# Agentic Report → Plan → Approve → Fix (OpenAI Codex)

Testers report bugs **from inside the app** (no GitHub account needed). Codex
verifies and proposes a fix plan for you to approve. Only after you approve does
it write the fix and open a PR — and only if the backend still compiles. **You
merge every PR.**

## The loop

```
Tester clicks "Report a problem" in the chat UI, types what's wrong
        │
        ▼  POST /api/report → backend POST /report-issue  (GH_ISSUE_TOKEN)
   Backend files a GitHub issue, labeled `user-report`
        │
        ▼  .github/workflows/codex-triage.yml  (label: user-report)
   Codex VERIFIES + posts a "proposed fix plan" comment   ← read-only, no code
        │
        ▼  You read the plan. To approve, comment:  @codex implement
        │
        ▼  .github/workflows/codex.yml  (phase 2)
   Codex writes the fix → py_compile gate → opens a PR     ← never if it won't compile
        │
        ▼
   You review + merge.   ← the only merge gate
```

## What's in the repo
| Piece | File |
|---|---|
| UI report button + modal | `frontend/components/ChatUI.tsx` |
| Frontend proxy | `frontend/app/api/report/route.ts` |
| Backend endpoint | `POST /report-issue` in `backend/main.py` |
| GitHub issue creation | `backend/github_issues.py` |
| Phase 1 — triage/plan | `.github/workflows/codex-triage.yml` |
| Phase 2 — implement (gated) | `.github/workflows/codex.yml` |
| PR review (optional) | `.github/workflows/codex-review.yml` |
| Agent context | `AGENTS.md` |

## One-time setup

1. **Codex GitHub app + API key** (powers phases 1–2):
   - Install the Codex GitHub app on `Aa-Rho-Hi/ECEN-Chatbot` (github.com/apps/codex).
   - Add repo secret `OPENAI_API_KEY` (Settings → Secrets and variables → Actions).
     The OpenAI account needs billing/credit.

2. **Token for the app to file issues** (powers the UI report button):
   - Create a **fine-grained PAT**: github.com/settings/tokens → Fine-grained →
     repo `ECEN-Chatbot` → Repository permissions → **Issues: Read and write**.
   - Put it in the backend env as `GH_ISSUE_TOKEN`, and set
     `GH_REPO=Aa-Rho-Hi/ECEN-Chatbot`. Locally that's `.env`; on Cloud Run it's a
     Secret Manager secret. **Never commit it.**

3. **The `user-report` label** must exist (the backend applies it). Create it once
   in the repo's Labels page, or it's auto-created on first issue.

4. **Protect `main`** so nothing merges without you: Settings → Branches →
   require a PR + 1 approval.

## How you drive it day to day
- A report comes in → you get a GitHub notification with Codex's **fix plan**.
- Happy with it? Comment **`@codex implement`** (you can add tweaks in the same
  comment, e.g. `@codex implement — also add a test`).
- Codex opens a PR **only if it compiles**. If its change fails the `py_compile`
  gate, it posts the reasoning instead of a broken PR, and you can refine and
  re-approve.
- Review the PR and merge.

## Notes
- **Compile gate** (`python -m py_compile backend/*.py crawler/*.py`) is a syntax/
  import-safety backstop so a fix can't ship a backend that won't start. It's not
  a full test suite — review still matters.
- **Job split:** the Codex job holds `OPENAI_API_KEY` with read-only repo scope;
  a separate job opens the PR with write scope but no key (OpenAI's recommended
  pattern against prompt-injection from issue text).
- **Cost:** every report triggers one triage run; every `@codex implement`
  triggers one implement run. Both bill your OpenAI key.
- Reporting is **disabled gracefully** if `GH_ISSUE_TOKEN`/`GH_REPO` are unset —
  the endpoint returns 503 and the UI shows a friendly error.
