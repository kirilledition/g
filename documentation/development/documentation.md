# Documentation Operations

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-01 documentation build and publishing | Documentation maintainers |

This repository keeps documentation in three directories:

- `documentation/public/` contains user-facing guidance for installation, quick starts, CLI usage, configuration, inputs, outputs, GPU/SLURM use, and troubleshooting.
- `documentation/development/` contains development-team guidance, architecture notes, style rules, tooling references, and automation workflows.
- `documentation/scratchpad/` contains internal work products created while developing, such as learning notes, review-derived notes, profiling summaries, and historical investigation output.

Public and development pages are in the primary Zensical navigation. Scratchpad
pages remain under `documentation/scratchpad/` for direct-path local reference
and may be stale.

## Local Build

Documentation dependencies live in the `docs` dependency group.

```bash
uv sync --group docs
just docs-serve
just docs-build
just docs-check
```

`just docs-serve` runs `uv run --group docs zensical serve` for a local preview. `just docs-build` runs `uv run --group docs zensical build --clean` and writes generated HTML to `documentation_rendered_website/`. Do not commit `documentation_rendered_website/`.
`just docs-check` builds the site and runs repository-specific rendering
guardrails for dynamic documentation assets such as Mermaid diagrams.

## Zensical Configuration

The site is configured in `zensical.toml`.

- `docs_dir = "documentation"` keeps all three documentation sections publishable.
- `site_dir = "documentation_rendered_website"` keeps generated output in the ignored local build directory.
- `nav` controls section names and ordering. Keep page order explicit because it is the source of truth for previous/next links and same-section continuous scrolling. Do not add scratchpad pages to the primary nav unless the page has been promoted or clearly marked internal.
- `extra_css` and `extra_javascript` load the repository documentation assets used for same-section continuous scrolling.
- `[project.extra]` contains Zensical extras, including `generator = false` to suppress the generator notice in the footer.
- `[project.theme]` controls the Zensical theme. Change `variant` for the base visual treatment, update `features` for navigation, search, and content actions, and keep `logo` and `favicon` pointed at the sparkle assets under `documentation/assets/images/`.
- `[project.plugins.mkdocstrings.handlers.python]` points API documentation rendering at `src/` and keeps Google-style docstrings aligned with the project style guide.

When moving a page between sections, update `zensical.toml`, local Markdown links, and any repository references to the old path.

## Page Contracts

Public pages describe user-visible behavior. Development pages describe
implementation contracts and maintainer workflows. Scratchpad pages preserve
historical work products and are not authoritative unless promoted into public
or development docs.

Every maintained public or development page should declare freshness near the
top with:

```md
| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of YYYY-MM-DD topic scope | Public user docs |
```

When changing user-facing CLI behavior, configuration, input/output contracts,
runtime behavior, performance assumptions, or deployment workflow, update the
relevant page under `documentation/public/` in the same branch. When changing
implementation architecture, maintainer tooling, or validation workflows, update
the relevant page under `documentation/development/`.

## GitHub Pages Setup

The repository publishes the built `documentation_rendered_website/` directory through the `Documentation` GitHub Actions workflow.

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
4. builds the Zensical site into `documentation_rendered_website/`;
5. checks repository-specific rendering hooks;
6. runs `actions/configure-pages`;
7. uploads `documentation_rendered_website/` with `actions/upload-pages-artifact`;
8. deploys with `actions/deploy-pages`.

Permissions are scoped by job. The build job has `contents: read` and `pages: read` so it can configure and upload the Pages artifact without deployment credentials. The deploy job alone has `pages: write` and `id-token: write` for the Pages deployment and OIDC flow.

The workflow is path-filtered. It runs for docs infrastructure and publishable documentation changes, including:

- `documentation/**`
- `zensical.toml`
- `pyproject.toml`
- `uv.lock`
- `Justfile`
- `.github/workflows/docs.yml`

It excludes `documentation/scratchpad/**`, so scratchpad-only updates do not rebuild or deploy the site. If a scratchpad update should be published immediately, include a related change outside `documentation/scratchpad/` in the same commit or run the workflow manually after confirming the workflow supports that event.

## Action Failure Triage

If the `Documentation` workflow fails:

- Run `just docs-check` locally first. Most failures are broken links, moved files missing from `zensical.toml`, rendering-hook regressions, or dependency lock mismatches.
- Check that the GitHub repository Pages source is set to `GitHub Actions`.
- Check that the build job grants `contents: read` and `pages: read`, and that only the deploy job grants `pages: write` and `id-token: write`.
- Check that the build output path in `.github/workflows/docs.yml` matches `site_dir` in `zensical.toml`.
- Check action version changes in the workflow only when the local docs build succeeds but remote deployment setup fails.
