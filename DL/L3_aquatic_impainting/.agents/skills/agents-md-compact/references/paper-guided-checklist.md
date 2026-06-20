# Paper-guided checklist

Use this as a deletion checklist, not a section template.

## Findings to preserve

- Good `AGENTS.md` files can reduce runtime and output tokens when they give agents direct repo navigation and runnable commands.
- Bad `AGENTS.md` files can reduce success and increase cost when they add unnecessary requirements.
- Common smells:
  - Context bloat: long background, duplicated README content, tutorials.
  - Lint leakage: style rules copied from formatter/linter configs.
  - Skill leakage: full workflows that belong in skills/tools.
  - Conflicting instructions: incompatible rules across root and nested files.
  - Stale commands: commands not present in repo config or CI.
  - Over-broad mandates: rules like "always test everything" without repo-specific reason.

## Final AGENTS.md target

- Aim for 40-120 lines unless the repo is a real monorepo.
- Prefer bullets and commands.
- Each line should answer: "Will this prevent a likely agent mistake in this repo?"
- If not, delete it.

## Compact structure

```markdown
# AGENTS.md

## Project
- One-sentence purpose.

## Layout
- `path/`: why it matters.

## Commands
- Install: `...`
- Test: `...`
- Build: `...`

## Working Rules
- Repo-specific constraints only.

## Avoid
- Known costly mistakes only.
```
