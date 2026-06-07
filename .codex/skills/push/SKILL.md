---
name: push
description: Push the current Symphony branch and integrate it directly into main.
---

# Push

Use this skill after local validation passes and the branch is ready for direct
integration into `main`.

## Workflow

1. Confirm the branch is `symphony/<issue-key>`.
2. Run validation appropriate for the change:
   - `just check-local`
   - `just test-local`
   - focused `uv run pytest ...`
   - relevant `just slurm-*` recipes for GPU-sensitive changes
3. Push with upstream tracking:
   - `git push -u origin HEAD`
4. If push is rejected because the remote moved, use the `pull` skill, re-run
   validation, then push again.
5. Move the Linear issue to `Merging` before touching `main`.
6. Fetch `origin/main` and merge it into the task branch if it has advanced:
   - `git fetch origin main`
   - `git merge origin/main`
7. Resolve any conflicts in the task worktree, then re-run validation.
8. Push the validated branch head directly to `main`:
   - `git push origin HEAD:main`
9. If the `HEAD:main` push is rejected because `main` advanced, repeat fetch,
   merge, validation, and push.
10. If GitHub branch protection rejects direct pushes to `main`, record the
   blocker in the Linear workpad and leave the issue in `Merging` or `Blocked`.
11. Attach or link the pushed branch and main commit on the Linear issue using
   the `linear` skill.
12. Update the workpad with the branch URL, main commit SHA, and validation
   evidence.
13. Move the Linear issue to `Done` after `origin/main` contains the task branch
   head.

Never use `git push --force`; use `--force-with-lease` only after an intentional
history rewrite and after confirming it will not discard remote work.

Do not create GitHub pull requests unless a human explicitly asks for one on the
issue.
