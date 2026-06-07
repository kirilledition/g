---
name: pull
description: Merge latest origin/main into the current Symphony worktree branch and resolve conflicts.
---

# Pull

Use this skill before implementation starts and whenever the branch needs to be
updated with latest `origin/main`.

## Workflow

1. Confirm the current branch is `symphony/<issue-key>`.
2. Ensure `git status --short` is clean or commit intended changes first.
3. Enable rerere:
   - `git config rerere.enabled true`
   - `git config rerere.autoupdate true`
4. Fetch:
   - `git fetch origin`
5. Merge:
   - `git -c merge.conflictstyle=zdiff3 merge origin/main`
6. If conflicts occur, inspect both sides, resolve one logical batch at a time,
   run `git diff --check`, then continue the merge.
7. Run focused validation after conflict resolution.
8. Record the merge source, result, resulting short SHA, and validation in the
   Linear workpad.

Do not rebase unless the Linear issue explicitly asks for it.
