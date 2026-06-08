# Documentation Operations

This repository keeps documentation in three directories:

- `docs/public/` contains user-facing guidance for installation, quick starts, CLI usage, configuration, inputs, outputs, GPU/SLURM use, and troubleshooting.
- `docs/development/` contains development-team guidance, architecture notes, style rules, tooling references, and automation workflows.
- `docs/scratchpad/` contains internal work products created while developing, such as learning notes, review-derived notes, profiling summaries, and historical investigation output.

All three directories can be published by Zensical. The site navigation labels each section so a reader can distinguish public product guidance from development-team material and scratchpad history.

## Local Build

Documentation dependencies live in the `docs` dependency group.

```bash
uv sync --group docs
just docs-serve
just docs-build
```

`just docs-serve` runs `uv run --group docs zensical serve` for a local preview. `just docs-build` runs `uv run --group docs zensical build --clean` and writes generated HTML to `site/`. Do not commit `site/`.

## Zensical Configuration

The site is configured in `zensical.toml`.

- `docs_dir = "docs"` keeps all three documentation sections publishable.
- `site_dir = "site"` keeps generated output in the ignored local build directory.
- `nav` controls section names and ordering. Keep page order explicit because it is the source of truth for previous/next links and same-section continuous scrolling.
- `extra_css` and `extra_javascript` load the repository docs assets used for same-section continuous scrolling.
- `[project.extra]` contains Zensical extras, including `generator = false` to suppress the generator notice in the footer.
- `[project.theme]` controls the Zensical theme. Change `variant` for the base visual treatment, update `features` for navigation, search, and content actions, and keep `logo` and `favicon` pointed at the sparkle assets under `docs/assets/images/`.
- `[project.plugins.mkdocstrings.handlers.python]` points API documentation rendering at `src/` and keeps Google-style docstrings aligned with the project style guide.

When moving a page between sections, update `zensical.toml`, local Markdown links, and any repository references to the old path.

## GitHub Pages Setup

The repository publishes the built `site/` directory through the `Documentation` GitHub Actions workflow.

To enable or verify Pages in the GitHub UI:

1. Open the repository on GitHub.
2. Go to `Settings`.
3. In the left sidebar, open `Pages`.
4. Under `Build and deployment`, set `Source` to `GitHub Actions`.
5. Save the setting if GitHub shows a save button.
6. Run or re-run the `Documentation` workflow from the `Actions` tab, or push a qualifying docs change to `main`.
7. Return to `Settings` -> `Pages` and confirm the latest deployment URL.

GitHub Pages does not bind the Pages setting to one workflow file. The Pages settings page links to the workflow run that most recently deployed the site.

## Documentation Workflow

The documentation workflow lives in `.github/workflows/docs.yml`.

On pull requests to `main`, it installs the docs dependency group and builds the Zensical site. It does not deploy previews.

On pushes to `main`, it:

1. checks out the repository;
2. installs Python and `uv`;
3. installs documentation dependencies;
4. builds the Zensical site into `site/`;
5. runs `actions/configure-pages`;
6. uploads `site/` with `actions/upload-pages-artifact`;
7. deploys with `actions/deploy-pages`.

The workflow has `contents: read`, `pages: write`, and `id-token: write` permissions because Pages deployment uses GitHub's Pages artifact and OIDC flow.

The workflow is path-filtered. It runs for docs infrastructure and publishable documentation changes, including:

- `docs/**`
- `zensical.toml`
- `pyproject.toml`
- `uv.lock`
- `Justfile`
- `.github/workflows/docs.yml`

It excludes `docs/scratchpad/**`, so scratchpad-only updates do not rebuild or deploy the site. If a scratchpad update should be published immediately, include a related change outside `docs/scratchpad/` in the same commit or run the workflow manually after confirming the workflow supports that event.

## Action Failure Triage

If the `Documentation` workflow fails:

- Run `just docs-build` locally first. Most failures are broken links, moved files missing from `zensical.toml`, or dependency lock mismatches.
- Check that the GitHub repository Pages source is set to `GitHub Actions`.
- Check that the workflow still grants `pages: write` and `id-token: write`.
- Check that the build output path in `.github/workflows/docs.yml` matches `site_dir` in `zensical.toml`.
- Check action version changes in the workflow only when the local docs build succeeds but remote deployment setup fails.
