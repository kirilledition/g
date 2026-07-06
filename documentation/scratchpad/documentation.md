# Documentation Notes

> Internal scratchpad. Canonical docs under `documentation/public/`
> and `documentation/development/`.

## Goal

Docs answer fast:

```text
Can I install it?
Can I run a real or tutorial analysis?
Can I trust and interpret the results?
Can I debug, tune, and resume it safely?
```

## Information Architecture

```text
README.md
  short project portal

documentation/public/
  user guide, tutorials, references, troubleshooting

documentation/development/
  architecture, migration, testing, tooling, benchmarking, maintainer policy

documentation/scratchpad/
  non-authoritative future-development notes only
```

## Rules

- README links, no manual duplicate.
- Public docs = behavior.
- Development docs = implementation contracts/invariants.
- Scratchpad = non-authoritative, stale possible.
- No hand-maintained defaults in many places.
- Parser-check user commands where practical.
- Public docs avoid `just`; development docs may use it.

## Topic Owners

| Topic | Canonical owner |
| --- | --- |
| Product summary | `README.md`, `documentation/public/index.md` |
| Installation | `documentation/public/installation.md` |
| First run / examples | `documentation/public/quickstart.md` |
| CLI grammar | `documentation/public/cli.md` |
| TOML config | `documentation/public/configuration.md` |
| Input contracts | `documentation/public/input-files.md` |
| Output contracts | `documentation/public/output-files.md` |
| Resume/manifests | `documentation/public/resume-and-manifest.md` |
| Statistics | `documentation/public/algorithm.md` |
| Compatibility | `documentation/public/compatibility.md` |
| Python API | `documentation/public/api-python.md` |
| GPU/cluster use | `documentation/public/gpu-and-clusters.md` |
| Performance | `documentation/public/performance-guide.md` |
| Troubleshooting | `documentation/public/troubleshooting.md` |
| Architecture | `documentation/development/architecture.md` |
| Rust migration | `documentation/development/rust-migration.md` |
| Config frontend | `documentation/development/configuration-frontend.md` |
| Tooling/Justfile | `documentation/development/tooling.md`, `development/justfile.md` |

## Missing Or Weak Public Docs

- first-run tutorial with tiny fixtures;
- REGENIE Step 1 predictions -> `g` Step 2 guide;
- port existing REGENIE command guide;
- output analysis guide;
- FAQ;
- glossary.

## Guardrails To Add

- Docs build pass.
- Nav pages exist.
- Public docs no stale config paths or removed commands.
- Generated option/schema docs fail if Rust metadata changes.
