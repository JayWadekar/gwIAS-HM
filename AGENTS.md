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
