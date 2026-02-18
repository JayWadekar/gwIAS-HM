## Branch Safety (Hard Requirement)
- Allowed working branches for this repository are:
  - `dev-codex`
  - `codex/tejaswi-exp1`
  - `codex/jay-exp1`
- Before making any edits or running project-changing commands, run:
  - `git branch --show-current`
- If current branch is not one of the allowed branches, stop and ask the user which allowed branch to use.
- Do not switch branches automatically unless the user explicitly asks.
- Do not create or use any other branch unless the user explicitly asks.

## Knowledge Anchoring (Hard Requirement)
- Before substantial analysis, design, or code changes, read:
  - `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_SPEC.md`
  - `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_TODO.md`
- Treat `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_SPEC.md` as the canonical source of pipeline aims, architecture, and constraints.
- If the spec and code context are insufficient, consult references in this order:
  - `/Users/tejaswi/Work/gwIAS-HM/references/REFERENCES.md`
  - relevant PDFs in `/Users/tejaswi/Work/gwIAS-HM/references/`

## Spec/TODO Workflow (Hard Requirement)
- Scope: apply this workflow only to changes that modify actual pipeline
  behavior/code, primarily under:
  - `/Users/tejaswi/Work/gwIAS-HM/Pipeline/`
  - and core pipeline behavior docs/configs that define how the pipeline runs.
- Exclusion: do not use spec/TODO workflow for auxiliary tasks unless the user
  explicitly asks. Auxiliary tasks include:
  - repo/branch management,
  - reference collection/indexing,
  - housekeeping/documentation not changing pipeline behavior,
  - operational/admin tasks.
- Track new goals and planned work in `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_TODO.md`.
- When work is completed:
  - remove or mark the corresponding TODO item as completed in the TODO file, and
  - add the new canonical behavior/aim to `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_SPEC.md`.
- Every completed addition to the spec must include:
  - a spec version bump, and
  - a changelog entry describing what was added/changed.
- Spec version format: `MAJOR.MINOR.PATCH`
  - `MAJOR`: incompatible conceptual restructuring
  - `MINOR`: new capabilities, aims, or workflow sections
  - `PATCH`: clarifications, wording fixes, or non-semantic edits

## Cross-Branch Sync Rule (Hard Requirement)
- If synchronizing work from `codex/tejaswi-exp1` to `dev-codex`, do not
  synchronize agent-only files unless the user explicitly asks:
  - `/Users/tejaswi/Work/gwIAS-HM/AGENTS.md`
  - `/Users/tejaswi/Work/gwIAS-HM/.agent/**`
- Use selective sync (for example, cherry-pick specific commits or apply
  path-limited changes) to keep agent-only files out of shared branch syncs.
