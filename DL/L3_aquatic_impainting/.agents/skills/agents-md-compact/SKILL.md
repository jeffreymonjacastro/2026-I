---
name: agents-md-compact
description: Create or revise repository AGENTS.md files using compact, evidence-based instructions. Use when the user asks Codex to create, audit, trim, rewrite, or improve AGENTS.md, CLAUDE.md-style repository agent instructions, or coding-agent context files while avoiding context bloat, lint leakage, skill leakage, and conflicting instructions.
---

# agents-md-compact

Create the smallest useful `AGENTS.md`: commands and constraints an agent needs to work in this repo, not a second README.

## Workflow

1. Inspect existing repo evidence before writing:
   - `README*`, package/project files, lockfiles, CI config, Makefiles, task runners.
   - Existing `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, or project skill files.
2. Include only instructions grounded in repo files or explicit user preferences.
3. Prefer exact commands over prose. Mark unverified commands as unverified or omit them.
4. Keep root `AGENTS.md` short. Use nested `AGENTS.md` only for true subproject differences.
5. Run one cheap validation: markdown sanity plus any listed command that is safe and fast.

## Must Include

- Project purpose in one sentence, only if not obvious from file names.
- Layout map with only directories an agent must know.
- Setup/test/build commands that actually exist.
- Repo-specific safety rules: secrets, generated artifacts, remote execution boundaries, data-loss risks.
- One short "do not" list for known costly mistakes.

## Must Omit

- Generic coding advice.
- Long architecture essays.
- Lint/format rules already enforced by config files.
- Full skill workflows or tool manuals; link to skills instead.
- Unverified deploy steps, credentials, or environment assumptions.
- "Always add tests" unless the repo or user explicitly requires it.

## Paper-Guided Check

Before finalizing, read `references/paper-guided-checklist.md` and remove anything that fails it.
