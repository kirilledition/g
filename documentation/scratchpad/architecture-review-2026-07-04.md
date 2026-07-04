# Architecture Review - 2026-07-04

## Scope

Reviewed the current dirty worktree as-is. This is a high-level architecture audit, not
a behavior change. No production code was edited for this report.

Focus areas:

- how Rust workspace crates communicate;
- how much Rust/Python contact surface exists;
- whether the current split is overcomplicated or over-entangled;
- where wrappers, payload objects, policy objects, or DTOs look unnecessary;
- how functional design, SOLID, and clean-code heuristics can guide the next cleanup.

## Executive Verdict

Cargo-level architecture is mostly healthy. The internal crate dependency graph is
small, acyclic, and directionally sensible. Core domain crates do not depend on PyO3
or Python. `g-plan`, `g-genotype`, `g-input`, and `g-runtime` have no internal crate
dependencies. `g-interface` depends only on `g-plan`; `g-output` depends only on
`g-plan`; `g-engine` depends on `g-genotype` and `g-output`. This is not over-entangled
at the crate graph level.

The overcomplication is at the Rust/Python boundary and inside the transitional
Python orchestration path. The PyO3 surface exposes 139 classes and 15 registered
functions across 24 modules, with especially large `schedule` and
`callback_runtime_resources` modules. Many of those objects are precise and testable,
but the aggregate surface is too granular: Python sees many tiny `Native*Plan`,
`Native*Result`, and `Native*Policy` objects instead of a few coarse engine sessions
or request/result contracts.

The architecture direction is right: pure Rust planning/policy, Rust-owned I/O and
runtime state, Python-owned JAX kernels. The current implementation still leaks too
many internal planning atoms across PyO3. That raises maintenance cost, makes stubs
large, and forces Python to assemble lifecycle behavior that Rust is increasingly
well-positioned to own.

## Current Crate Communication

Observed internal dependencies from root and crate manifests:

| Package | Internal dependencies | Interpretation |
| --- | --- | --- |
| root `g` | `g-cli`, `g-engine`, `g-genotype`, `g-input`, `g-interface`, `g-output`, `g-plan`, `g-runtime` | Composition/PyO3 adapter. Expected to be broad during migration. |
| `g-cli` | `g-interface` | Thin CLI bridge. Healthy. |
| `g-interface` | `g-plan` | Config/frontend compiles toward plan contracts. Healthy. |
| `g-output` | `g-plan` | Output manifests consume plan identities. Healthy, but contact is high. |
| `g-engine` | `g-genotype`, `g-output` | Engine has BGEN/output ownership. Reasonable, but not yet full application coordinator. |
| `g-plan` | none | Pure contract crate. Healthy. |
| `g-genotype` | none | Isolated BGEN/genotype crate. Healthy. |
| `g-input` | none | Isolated sample/prediction alignment crate. Healthy. |
| `g-runtime` | none | Isolated runtime/telemetry crate. Healthy at graph level. |

Reference counts from source grep:

| Source | Target | Direct refs |
| --- | --- | ---: |
| `cli` | `interface` | 13 |
| `engine` | `genotype` | 4 |
| `engine` | `output` | 5 |
| `output` | `plan` | 74 |
| root PyO3 (`src/python`) | `engine` | 21 |
| root PyO3 (`src/python`) | `genotype` | 9 |
| root PyO3 (`src/python`) | `input` | 5 |
| root PyO3 (`src/python`) | `interface` | 2 |
| root PyO3 (`src/python`) | `output` | 4 |
| root PyO3 (`src/python`) | `runtime` | 16 |

The high `g-output` to `g-plan` contact is expected because manifests persist prepared
run-plan identity. It is not a cycle or ownership violation, but it should stay
one-way: output should continue consuming prepared plan identity, not discovering
analysis semantics itself.

## Contact Surface Metrics

Rust public item counts by crate:

| Crate | Rust source files | Public items |
| --- | ---: | ---: |
| `g-cli` | 2 | 13 |
| `g-engine` | 15 | 350 |
| `g-genotype` | 17 | 94 |
| `g-input` | 7 | 30 |
| `g-interface` | 14 | 72 |
| `g-output` | 9 | 89 |
| `g-plan` | 4 | 42 |
| `g-runtime` | 16 | 397 |

PyO3 surface counts:

| PyO3 module | Registered classes | Registered functions | Lines | Note |
| --- | ---: | ---: | ---: | --- |
| `src/python/schedule.rs` | 50 | 0 | 3295 | Largest exposed micro-plan surface. |
| `src/python/callback_runtime_resources.rs` | 15 | 0 | 3305 | Large runtime/resource facade. |
| `src/python/config/mod.rs` | 8 | 11 | 898 | Necessary config bridge. |
| `src/python/run_events.rs` | 7 | 0 | 2480 | Telemetry/event payload bridge. |
| `src/python/output.rs` | 7 | 0 | 1093 | Output lifecycle bridge. |
| `src/python/runtime_state.rs` | 6 | 0 | 642 | Process-global runtime bridge. |
| `src/python/sample_alignment.rs` | 5 | 3 | 305 | Input bridge. |
| Other PyO3 modules | 41 | 1 | mixed | Smaller bridges. |
| Total | 139 | 15 | - | Large boundary. |

Python orchestration shape:

| Package area | Files | Functions | Classes | Dataclasses |
| --- | ---: | ---: | ---: | ---: |
| `src/g/runner` | 8 | 99 | 16 | 13 |
| `src/g/engine/native_dispatch` | 10 | 66 | 13 | 5 |
| `src/g/engine/regenie2_pipeline` | 19 | 104 | 11 | 10 |
| `src/g/engine/callbacks` | 11 | 259 | 29 | 11 |

This is the real contact cost. The crate graph is clean, but the Python/Rust bridge
has many small objects and many call sites that need to stay synchronized with
`src/g/_core.pyi`.

## Findings

| Area | State | Design read | Recommendation |
| --- | --- | --- | --- |
| Cargo crate graph | Clean, acyclic, low direct coupling. | Good dependency direction. Single-responsibility split is mostly working. | Keep this graph. Do not add `g-common`, `g-utils`, or root-level shared buckets. |
| Root `g` crate | Depends on every internal crate and registers all PyO3 symbols. | Acceptable as composition layer during migration. Risk only if policy lives here. | Keep root broad but dumb: PyO3 conversion and registration only. Move policy to domain crates. |
| `g-plan` | Pure contracts and deterministic planning helpers. | Good functional core. Immutable values and pure builders fit current needs. | Keep as central language between crates. Prefer adding stable contract types here over ad hoc dict payloads. |
| `g-interface` | Rust owns CLI/TOML/config/defaults and emits plan requests. | Good separation. Uses Serde/TOML where it belongs. | Keep frontend-only. Do not let it open BGEN/output/JAX. |
| `g-genotype` and `g-input` | Independent, no internal deps. | Strong boundaries. BGEN and input alignment are isolated. | Keep independent. Let engine compose them, not each other. |
| `g-output` | Consumes `g-plan` heavily for manifests/resume. | Cohesive side-effect boundary. Coupling to `g-plan` is justified. | Keep output as side-effect shell. Avoid adding compute/input knowledge. |
| `g-runtime` | No internal deps, but 397 public Rust items and huge run-event export list. | Graph is clean, API surface is broad. Telemetry payload builders are many small pure functions. | Group event builders behind fewer domain facades or typed event enums when changing this area. |
| `g-engine` | Good backend/effects traits and phase coordinator, plus large schedule planner. | Functional-core / imperative-shell pattern is visible and valuable. Surface is too broad. | Keep backend/effects traits. Collapse exported schedule atoms where Python no longer needs each item. |
| PyO3 schedule bridge | 50 exposed `Native*Plan` classes, plus `NativeSchedulePolicy`. | Interface segregation is inverted: Python depends on many internals. | Replace clusters with coarse methods/results when touched. Prefer one native scheduler/session object returning coarse lifecycle outcomes. |
| PyO3 callback runtime resources | 15 exposed result/resource classes over 3305 lines. | Some state belongs in Rust, but Python sees too much internal machinery. | Move queue/buffer/backpressure lifecycle toward Rust-owned session. Python should provide JAX callback functions and arrays. |
| Python runner -> pipeline -> native dispatch | Many dataclasses and forwarding functions. | Clear but deep hop chain. Clean-code smell: procedural wrappers around transport. | Consolidate dispatch request into one typed request object and reduce pass-through wrappers. |
| Python `_core` usage | `_core`/`Native*` refs spread across runner, engine, callbacks, output, JAX runtime. | Boundary is not localized enough. | Keep `_core` imports concentrated in bridge modules; domain Python modules should depend on Python protocols/contracts, not PyO3 details. |
| Policy objects | Many `Native*Policy` classes are stateless or thin over pure Rust functions. | Useful during migration, but object-per-policy can become ceremony. | For stateless pure functions, prefer module-level PyO3 functions or coarse service facade. Keep stateful policies only where lifecycle/cache exists. |
| Payload dict conversion | Repeated Rust payload -> `PyDict` -> Python dataclass mapping. | Transport noise. Weak typed boundary. | Prefer exposing PyO3 classes/dataclasses directly or returning JSON only at persistence boundaries. Avoid dict payloads for internal calls. |

## Overcomplication Assessment

Not over-entangled:

- Crate dependencies are one-way and acyclic.
- Domain crates do not depend on PyO3.
- BGEN decode, input alignment, output, config, runtime, and plan contracts are separated.
- `g-engine` already has useful abstractions: `AssociationBackend`, `EngineRunEffects`,
  `EngineCoordinator`, and explicit phase transitions.

Overcomplicated:

- PyO3 exports too many internal schedule/callback concepts.
- Python orchestration still assembles lifecycle by calling many native micro-plans.
- `Native*Policy` objects sometimes wrap pure functions without carrying state.
- `payload` dictionaries are used as transport between typed Rust and typed Python.
- `src/g/_core.pyi` is large enough that stubs are now an architecture maintenance
  surface, not just typing support.

Over-entangled at boundary:

- Python callbacks know native queue/backpressure/result-resource details.
- Runner/pipeline/native-dispatch layers all know `_core` names.
- Output preparation and manifest lifecycle cross Python/Rust several times, even
  though Rust owns output semantics.

The issue is not too many crates. The issue is too many cross-boundary concepts.

## Design Guidance

### Functional design

Keep pure logic in Rust as plain functions over immutable inputs:

- plan resolution;
- GPU genotype format selection;
- resume compatibility decisions;
- telemetry/event payload construction;
- queue/backpressure scheduling decisions;
- manifest identity construction.

Then keep side effects at explicit edges:

- BGEN reads;
- file writes/finalization;
- JAX import/runtime setup;
- JAX kernel execution;
- telemetry sink writes;
- signal handling.

This pattern is already present. The next improvement is to stop exporting every pure
decision as its own Python-visible plan object. Compose pure Rust decisions inside
coarser Rust-owned lifecycle operations.

### SOLID heuristics

- Single responsibility: crates mostly pass. PyO3 modules sometimes fail because
  they combine conversion, policy, lifecycle, and telemetry emission.
- Open/closed: backend trait direction is good. `AssociationBackend` gives space for
  JAX, Rust-score, or future CUDA backends without changing coordinator logic.
- Liskov/interface segregation: Python-facing `NativeSchedulePolicy` exposes too much.
  Callers depend on methods and result objects they should not need.
- Dependency inversion: engine owns backend/effects traits, good. Python should
  implement a coarse association backend adapter, not manipulate engine internals.

Use SOLID as pressure, not dogma. The useful target is fewer, stronger boundaries.

### Clean-code heuristics

- Remove wrappers that only rename and forward to `_core` when they have one caller.
- Keep wrappers that enforce domain naming, validation, lazy JAX import, or public API
  stability.
- Avoid names like `payload` for internal domain values. Use `RunStartedEvent`,
  `PreparedRunPlan`, `OutputManifestHeader`, etc.
- Prefer one typed request object over a function with 20 keyword arguments.
- Prefer one lifecycle owner per phase. If Rust owns output lifecycle, Python should
  not reconstruct output lifecycle state except to pass JAX arrays/results.

## Recommended Cleanup Direction

Priority 1: keep current crate graph and shrink boundary surface.

- Do not split more crates now.
- Do not introduce generic shared crates.
- Keep root `g` as broad PyO3 composition, but prevent application policy from living
  there.
- Track PyO3 surface count as an architecture metric.

Priority 2: collapse schedule/callback micro-objects.

- Replace clusters of `Native*Plan`/`Native*Result` classes with coarse native
  session methods where Python only needs an action outcome.
- Keep detailed Rust structs internally for unit tests.
- Expose detailed DTOs only when Python truly branches on their fields.

Priority 3: make Python engine requests typed and narrow.

- Replace large kwargs fanout between runner and pipeline with one immutable dispatch
  request per execution shape.
- Keep JAX-heavy imports lazy.
- Keep `_core` imports inside bridge modules, not broad domain modules.

Priority 4: reduce dict payload bridges.

- Convert stable internal payloads to typed PyO3 classes or typed Python dataclasses at
  one boundary.
- Keep JSON/dict conversion for persisted manifests, telemetry JSONL, and user-facing
  serialization only.

Priority 5: continue toward Rust-owned lifecycle.

- Rust should own queue, buffer, backpressure, output session, resume, cleanup, and
  phase state.
- Python should own JAX runtime setup, JAX kernels, and numeric callback execution.
- The long-term Python adapter should look like "prepare backend, compute batch,
  return arrays/results", not "orchestrate native queue internals".

## Specific Wrapper/Object Suspects

These are not deletion instructions. They are high-value areas for focused cleanup.

| Suspect | Why it looks heavy | Safer replacement direction |
| --- | --- | --- |
| `src/python/schedule.rs` `Native*Plan` classes | 50 PyO3 classes mirror internal schedule atoms. | Keep Rust structs internal; expose fewer lifecycle outcomes. |
| `src/python/callback_runtime_resources.rs` result classes | 15 PyO3 result/resource classes over 3305 lines. | Move queue/buffer lifecycle behind Rust session methods. |
| `Native*Policy` stateless classes | Several are object wrappers over pure functions. | Use module functions or one coarse service object unless state/cache exists. |
| `payload` dict builders in PyO3 | Type erasure between typed Rust and typed Python. | Typed PyO3 classes/dataclasses or direct JSON only at persistence boundaries. |
| Python `native_dispatch` wrappers | Many short functions wrap `_core` while adding only diagnostics. | Keep diagnostics, but group engine open/validation/delivery into coarser bridge. |
| Runner common dispatch kwargs | Wide request passed through multiple layers. | One typed dispatch request object per single/multi execution path. |

## What To Keep

- Current crate dependency direction.
- `g-plan` as pure contract crate.
- `g-interface` as Rust-owned config/TOML/CLI frontend.
- `g-genotype` and `g-input` independence.
- `g-output` as output/resume/manifest owner.
- `AssociationBackend` and `EngineRunEffects` traits.
- Lazy JAX import boundary in runner/runtime.
- Public Python API stability while internals move.

## Suggested Architecture Target

Near-term target:

```text
Python public API / CLI bridge
        |
Rust config + plan (`g-interface`, `g-plan`)
        |
Python runner configures JAX lazily
        |
Rust engine session owns lifecycle
        |
Python/JAX backend callback computes batches
        |
Rust output/runtime finalize side effects
```

Desired contact surface:

- Python calls a small number of native entrypoints per phase.
- Rust returns coarse typed outcomes.
- Python does not see queue slot, buffer reuse, worker stop-poll, or per-observation
  plan objects unless a callback must branch on them.
- Manifest and telemetry serialization stay Rust-owned and stable.

## Evidence Commands

Commands run from repository root:

```bash
git status --short --branch
python3 - <<'PY'
from pathlib import Path
import re, tomllib
# parsed Cargo.toml files, internal deps, public Rust counts, PyO3 counts, and _core usage counts
PY
python3 - <<'PY'
from pathlib import Path
import re
# counted direct g_* crate references from crates/* and src/python
PY
find src/g -maxdepth 3 -type f -name '*.py' | sort
sed -n '1,220p' crates/engine/src/lib.rs
sed -n '1,260p' crates/engine/src/coordinator.rs
sed -n '1,240p' crates/engine/src/backend.rs
sed -n '1,240p' crates/engine/src/effects.rs
sed -n '1,180p' src/python/mod.rs
sed -n '1,200p' src/lib.rs
sed -n '1,220p' crates/plan/src/lib.rs
sed -n '1,220p' crates/interface/src/lib.rs
sed -n '1,220p' crates/output/src/lib.rs
sed -n '1,220p' crates/runtime/src/lib.rs
sed -n '1,220p' src/g/runner/execution.py
sed -n '1,240p' src/g/engine/native_dispatch/engine.py
sed -n '1,260p' src/g/engine/regenie2_pipeline/single_trait.py
sed -n '1,260p' src/g/engine/regenie2_pipeline/context.py
sed -n '1,220p' src/g/execution_plan.py
sed -n '1,240p' src/g/interface/config.py
rg -n "Native.*Policy\\(|Native.*Plan|_core\\.Native|g\\._core|_core\\." src/g
rg -n "class .*Policy|Policy\\)|Plan\\)|payload|Payload|Result\\)|State\\)|Context\\)" src/python crates/runtime/src crates/engine/src crates/plan/src
wc -l src/g/_core.pyi src/python/schedule.rs src/python/callback_runtime_resources.rs crates/engine/src/schedule.rs crates/runtime/src/run_events.rs
```

## Limitations

This report used static inspection, grep, manifest parsing, and selected file reads.
It did not run performance workloads, cargo diagnostics, or integration tests. Static
counts do not prove code is bad or removable; they identify coupling and maintenance
pressure.

The worktree is dirty. Existing edits, deleted files, and scratchpad reports were left
untouched. Findings should be rechecked after the current cleanup branch settles.
