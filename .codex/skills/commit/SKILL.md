---
name: commit
description: Create a focused git commit from the current Symphony worktree changes.
---

# Commit

Use this skill when a coherent implementation slice is complete.

## Workflow

1. Inspect `git status --short`, `git diff`, and `git diff --staged`.
2. Exclude unrelated changes, local data, benchmark outputs, caches, and logs.
3. Stage only intended files.
4. Re-check the staged diff.
5. Write a conventional commit message with:
   - imperative subject under 72 characters
   - summary
   - rationale
   - validation run or explicit reason validation was not run
   - `Co-authored-by: Codex <codex@openai.com>`
6. Commit with `git commit -F <message-file>`.
7. Update the Linear workpad with the commit SHA and validation status.
