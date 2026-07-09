Yes. Renaming `src/binding` to `src/binding` is a good idea. The name `python` makes it sound like this directory owns Python-side application behavior. What it should actually be is a **PyO3 binding layer**.

The current code still makes `src/binding` look like an application subsystem: `src/lib.rs` imports `mod binding` and delegates `_core` module registration to `binding::register_module`.  Inside `src/binding/mod.rs`, the directory declares many modules: callback diagnostics/progress/queue/runtime resources, config, genotype, host policy, JAX runtime, logging, output, prediction sources, preflight, run engine, run events, lifecycle, schedule, shutdown, telemetry, timing, and more.  It also registers callback/schedule/preflight internals as part of the engine domain.

That is too much for a binding layer. The cleanup should make the distinction explicit:

```text
crates/*      = application/domain logic
src/binding   = PyO3 adaptation only
src/g         = public Python API + JAX backend/kernels
```

Below is a concrete implementation plan you can give to agents.

---

# Target principle

## New rule

```text
src/binding may contain PyO3 adaptation only.

If a function can be written using only Rust crate types, it belongs in crates/*.

If a function needs Python, PyO3, PyAny, PyErr, PyModule, NumPy/Python buffers,
or a Python callback, it may live in src/binding.
```

## Allowed in `src/binding`

```text
#[pyclass] wrappers
#[pymethods] getters/setters
#[pyfunction] wrappers
PyErr conversion
PyAny extraction
Python callback invocation
NumPy/Python buffer adapters
Python module/submodule registration
temporary compatibility aliases
```

## Forbidden in `src/binding`

```text
scheduling policy
callback worker lifecycle policy
BGEN delivery cleanup logic
manifest/header construction
output resume/repair planning
preflight validation logic
run-event/diagnostic payload construction
telemetry rendering policy
genotype preprocessing policy
sample alignment logic
domain-level config validation
```

Those belong in `g-engine`, `g-runtime`, `g-output`, `g-genotype`, `g-input`, `g-interface`, or `g-plan`.

---

# Desired final layout

## Rust crate tree

```text
src/
  lib.rs
  binding/
    mod.rs
    errors.rs

    convert/
      mod.rs
      path.rs
      arrays.rs
      json.rs

    cli/
      mod.rs

    config/
      mod.rs

    runtime/
      mod.rs
      jax.rs
      shutdown.rs
      logging.rs

    telemetry/
      mod.rs

    engine/
      mod.rs
      session.rs
      backend_bridge.rs

    genotype/
      mod.rs

    input/
      mod.rs

    output/
      mod.rs

    debug/
      mod.rs        # feature-gated / test-support only
```

## `_core` Python-visible namespace

```text
g._core.cli
g._core.config
g._core.runtime
g._core.telemetry
g._core.engine
g._core.genotype
g._core.input
g._core.output
g._core.debug      # not production
```

Root `g._core` can keep compatibility aliases temporarily, but new code should use submodules.

---

# Phase 0 — Stabilize and document the boundary

## Agent task

Create:

```text
documentation/development/binding-layer-policy.md
```

Content should define:

```text
1. `src/binding` is PyO3 only.
2. Pure Rust logic belongs in crates.
3. Production binding modules expose domain handles, not implementation internals.
4. Debug/test internals go under `_core.debug` or `test-support`.
5. New binding files must state which crate/domain they adapt.
```

Add a required module header convention:

```rust
//! PyO3 bindings for `_core.engine`.
//!
//! Adapts:
//! - g-engine high-level session/backend APIs
//!
//! Allowed:
//! - PyO3 wrappers
//! - Python callback bridge
//! - error conversion
//!
//! Forbidden:
//! - scheduling policy
//! - output manifest construction
//! - BGEN delivery cleanup
```

## Acceptance criteria

```text
No code behavior change.
Policy doc exists.
Agents have a written rule for what belongs in binding.
```

---

# Phase 1 — Rename `src/binding` to `src/binding`

## Why

The current name encourages the wrong ownership model. This is not “Python application code”; it is native binding code.

## Mechanical changes

Rename:

```text
src/binding/ -> src/binding/
```

Change `src/lib.rs` from:

```rust
mod binding;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    binding::register_module(module)
}
```

to:

```rust
mod binding;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    binding::register_module(module)
}
```

Current `src/lib.rs` uses the binding module name.

Confirm textual references:

```bash
rg "src/binding|mod binding|binding::register_module" .
```

Expected current state:

```text
Rust binding source lives under src/binding.
src/lib.rs declares mod binding.
_core delegates registration to binding::register_module.
```

Do not rename the Python package `src/g`. Do not rename `g._core`. This is only the Rust source directory/module name.

## Acceptance criteria

```text
cargo fmt --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features
maturin develop / import g._core still works
```

---

# Phase 2 — Introduce binding domain submodules

## Current issue

`src/binding/mod.rs` currently registers everything into one root module and groups domains only internally.  It should create real `_core.<domain>` submodules.

## Add helper

In `src/binding/mod.rs`:

```rust
fn add_submodule(
    root: &Bound<'_, PyModule>,
    name: &str,
    register: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = root.py();
    let submodule = PyModule::new(py, name)?;
    register(&submodule)?;
    root.add_submodule(&submodule)?;
    root.add(name, &submodule)?;

    let full_name = format!("{}.{}", root.name()?, name);
    py.import("sys")?
        .getattr("modules")?
        .set_item(full_name, &submodule)?;

    Ok(())
}
```

Adjust exact PyO3 API names as needed.

## New registration

Replace flat registration with:

```rust
pub(crate) fn register_module(root: &Bound<'_, PyModule>) -> PyResult<()> {
    add_submodule(root, "cli", cli::register_module)?;
    add_submodule(root, "config", config::register_module)?;
    add_submodule(root, "runtime", runtime::register_module)?;
    add_submodule(root, "telemetry", telemetry::register_module)?;
    add_submodule(root, "engine", engine::register_module)?;
    add_submodule(root, "genotype", genotype::register_module)?;
    add_submodule(root, "input", input::register_module)?;
    add_submodule(root, "output", output::register_module)?;

    #[cfg(any(test, feature = "test-support"))]
    add_submodule(root, "debug", debug::register_module)?;

    register_compatibility_aliases(root)?;
    Ok(())
}
```

## Compatibility aliases

Keep old root symbols temporarily:

```text
_core.run_cli_with_python_backend
_core.RegenieConfig
_core.NativeRunArtifacts
_core.OutputWriterSession
...
```

But document them as compatibility-only.

## Tests

Add:

```python
def test_core_domain_submodules_importable():
    import g._core
    import g._core.cli
    import g._core.config
    import g._core.runtime
    import g._core.engine
    import g._core.output
```

Also test root aliases still work until removed.

## Acceptance criteria

```text
import g._core.cli works
import g._core.engine works
current internal Python still passes
root aliases are temporary and documented
```

---

# Phase 3 — Reorganize binding modules by domain

## Current modules

Current `src/binding/mod.rs` has many files at one level.  Move them into domain folders.

## Mapping

### `src/binding/cli`

Move:

```text
cli_driver.rs -> cli/mod.rs
```

Keep only:

```text
NativeCliRunResult
NativeCliRunContext
NativeCliTelemetrySessionView
run_with_python_backend
```

Eventually rename Python-visible function:

```text
_core.run_cli_with_python_backend
  -> _core.cli.run_with_python_backend
```

The current Python CLI is already small and calls `_core.run_cli_with_python_backend`.  Migrate it to `_core.cli.run_with_python_backend` after submodules exist.

### `src/binding/config`

Move:

```text
config/*
host_policy.rs   # if still config-facing
```

Expose:

```text
RegenieConfig
InputConfig
TraitConfig
GComputeConfig
GOutputConfig
GDiagnosticsConfig
from_options
validate_for_run
TOML helpers
```

### `src/binding/runtime`

Move:

```text
runtime.rs
runtime_state.rs
jax_runtime.rs
shutdown.rs
logging.rs
timing.rs if process/runtime-level
```

Expose:

```text
ProcessRuntimeState
RuntimePolicy
RunRuntime
JaxRuntimeSetupSession
ShutdownController
logging init
```

### `src/binding/telemetry`

Move:

```text
telemetry_policy.rs
telemetry session wrappers from logging.rs if currently there
```

Expose:

```text
NativeTelemetryRunSession
TelemetryPaths
```

Avoid exposing every event builder.

### `src/binding/engine`

Move:

```text
run_engine.rs
run_lifecycle.rs
backend bridge code when added
```

Expose:

```text
NativeRunEngineSession
NativeRunArtifacts
NativeRunLifecycleSession temporarily
```

Do **not** expose callback queues/schedule/preflight in production engine.

### `src/binding/genotype`

Move:

```text
genotype.rs
```

Expose only required genotype handles.

### `src/binding/input`

Move:

```text
sample_alignment.rs
prediction_sources.rs
```

Temporary. Long-term, most of this should disappear when Rust engine owns input prep.

### `src/binding/output`

Move:

```text
output.rs
profile.rs if output profile-specific
```

Expose:

```text
OutputWriterSession
NativeChunkHandle
```

### `src/binding/debug`

Move or create wrappers for:

```text
schedule.rs
callback_queue.rs
callback_runtime_resources.rs
callback_summary.rs
callback_progress.rs
callback_diagnostics.rs
preflight.rs
```

These should be debug/test-support only.

Current production registration exposes those internals.  Move them out of production first, then delete as Rust ownership improves.

## Acceptance criteria

```text
src/binding root has domain folders, not 25 flat files.
Production `_core.engine` no longer registers callback/schedule internals.
Debug/test imports require `_core.debug`.
```

---

# Phase 4 — Move pure binding logic into crates

This is the core cleanup. For every big binding function, ask:

```text
Does this need PyO3?
```

If no, move it.

## Move to `g-runtime`

From binding files, move pure logic for:

```text
terminal result construction
CLI failed/interrupted/completed rendering
telemetry close planning
run diagnostic payload construction
runtime policy construction
shutdown decision policy
timing file write planning
```

`src/binding/cli` should only call Python backend and convert errors/results. The current `cli_driver` already does a lot of policy; keep shrinking it.

## Move to `g-engine`

Move pure logic for:

```text
scheduler plans
callback queue operation planning
callback worker lifecycle planning
callback resource state transitions
BGEN delivery cleanup plans
preflight validation
output-preparation orchestration
run-engine orchestration
```

## Move to `g-output`

Move pure logic for:

```text
manifest header construction
manifest JSON hashing
resume compatibility
writer session preparation
finalization decisions
```

## Move to `g-genotype`

Move pure logic for:

```text
BGEN reader config
preprocess stats helpers
variant/chromosome chunk planning
buffer layout validation
```

## Move to `g-input`

Move pure logic for:

```text
sample alignment
prediction source loading
phenotype group planning
grouped union sample planning
```

## Acceptance criteria

```text
No binding function named plan_* unless it is a direct PyO3 wrapper.
No binding function named validate_* unless it converts Python input and calls crate validation.
No binding function named build_* unless it builds a Python wrapper object.
```

---

# Phase 5 — Shrink or delete specific oversized binding modules

## 5A. `callback_runtime_resources`

Goal:

```text
delete or debug-gate
```

Reason: callback resources/queues/buffers/result slots should be owned by `g-engine`, not PyO3.

Interim:

```text
move to src/binding/debug/engine/callback_runtime_resources.rs
feature-gate with test-support/debug-api
remove from production _core
```

Final:

```text
delete after NativeRunEngineSession owns callback runtime
```

## 5B. `schedule`

Goal:

```text
move to debug/test-support, then delete most of it
```

Scheduling is `g-engine` logic. Production Python should not call schedule planners.

## 5C. `run_events`

Goal:

```text
keep only artifact/event PyO3 wrappers
move payload builder exposure into g-runtime or debug
```

Do not expose every diagnostic builder to Python.

## 5D. `run_engine`

Goal:

```text
replace with NativeRunEngineSession
```

It should not open BGEN, prepare input, build output, or schedule delivery directly in PyO3. It should call `g-engine`.

## 5E. `output`

Goal:

```text
keep writer/session/chunk handle wrappers only
move manifest/resume construction to g-output
```

## 5F. `config`

`config/mod.rs` may remain longer because PyO3 config classes are verbose. But the config logic should be in `g-interface`.

Acceptance criteria:

```text
No binding file over 1000 lines.
No production binding file over 600 lines without allowlist.
callback_runtime_resources and schedule are not production bindings.
```

---

# Phase 6 — Clean `src/g` Python after binding cleanup

The Python package should not depend on binding internals.

## Keep in Python

```text
src/g/api.py
src/g/runner/cli.py
src/g/runner/runtime.py       # JAX import/setup boundary
src/g/jax_runtime/*
src/g/compute/*
future src/g/jax_backend/*
```

## Delete or demote after Rust ownership

```text
src/g/execution_plan.py
src/g/engine/regenie2_pipeline/*
src/g/engine/native_dispatch/*
src/g/engine/callbacks/runtime.py
src/g/io.py if only manifest/output adapter
```

The CLI file is already close to the target: small adapter, Python backend callback only.  Apply that same pattern to the rest.

## Acceptance criteria

```text
Python owns JAX backend and public API.
Rust owns execution lifecycle and native host orchestration.
```

---

# Phase 7 — Crate facade cleanup in parallel

The binding cleanup will be blocked if crates keep exporting too many internals.

## `g-engine::api.rs`

Currently still exports many scheduler internals.  Shrink production facade to:

```text
EngineCoordinator / NativeRunEngineSession
EngineRunInput / EngineRunReport
AssociationBackend
BackendError
EngineError
RunPhase
temporary Regenie2RunEngineCore
```

Move schedule/callback internals to:

```text
pub(crate)
test_support
debug-api feature
```

## `g-runtime::api.rs`

Stop re-exporting every diagnostic/event builder from the main facade. Create deliberate submodules:

```text
runtime::events
runtime::diagnostics
runtime::telemetry
runtime::timing
```

Only high-level runtime handles should be in the top facade.

## `g-output::api.rs`

Expose output operations, not manifest JSON toolkit.

## `g-genotype::api.rs`

Expose genotype source contracts, not preprocess/planner internals.

---

# Phase 8 — Enforcement tooling

Add or extend architecture checks.

## Binding-layer checks

Create:

```text
tooling/debug/check_binding_architecture.py
```

Rules:

```text
1. src/binding must not contain pure planning functions unless allowlisted.
2. production binding modules cannot register schedule/callback_runtime_resources.
3. debug binding modules cannot be imported by production Python.
4. no binding module may exceed 600 LOC unless allowlisted.
5. no binding function may exceed 80 LOC unless allowlisted.
6. root _core aliases must be listed in compatibility allowlist.
7. new PyO3 symbols must be registered in a domain submodule.
```

## Crate facade checks

```text
1. crates/*/src/api.rs public re-export additions require PUBLIC_API.md update.
2. no public fake/test types outside test_support.
3. no public schedule/callback internals in g-engine api.rs unless allowlisted.
```

## Python package checks

```text
1. production src/g cannot import _core.debug.
2. production src/g cannot import g.engine.callbacks.runtime after migration.
3. production src/g cannot import g.engine.regenie2_pipeline after migration.
```

---

# Detailed PR sequence

## PR 1 — Rename `src/binding` to `src/binding`

```text
Pure mechanical rename.
Update src/lib.rs.
Update docs/scripts/tests.
No behavior change.
```

Validation:

```bash
cargo fmt --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features
python -c "import g._core"
```

## PR 2 — Add binding policy and architecture check

```text
Add docs.
Add check_binding_architecture.py.
No behavior change except checks.
```

## PR 3 — Add `_core` submodules with compatibility aliases

```text
_core.cli/config/runtime/telemetry/engine/genotype/input/output/debug
Keep root aliases.
Update _core.pyi.
```

## PR 4 — Reorganize binding files into domain folders

```text
Move files, keep same registration behavior.
No logic movement yet.
```

## PR 5 — Move debug internals out of production registration

```text
schedule/callback/preflight bindings move to _core.debug or test-support.
Production _core.engine becomes smaller.
```

## PR 6 — Move pure CLI/runtime policy to crates

```text
src/binding/cli becomes callback/error/result adapter only.
g-runtime owns terminal/lifecycle policy.
```

## PR 7 — Move event builder exposure out of binding

```text
run_events binding shrinks.
g-runtime owns payload builders and renderers.
```

## PR 8 — Move output manifest/prep logic to `g-output`

```text
output binding shrinks to handles.
Python output adapter shrinks or disappears.
```

## PR 9 — Introduce `NativeRunEngineSession`

```text
binding/engine exposes high-level session.
PyO3 no longer exposes pipeline/schedule internals.
```

## PR 10 — Move callback runtime to Rust engine

```text
delete/debug-gate callback_runtime_resources binding.
Python JAX backend only compute_batch.
```

## PR 11 — Delete old Python orchestration

```text
execution_plan.py production use removed
regenie2_pipeline removed or debug-only
native_dispatch removed or debug-only
callbacks/runtime.py removed or replaced by jax_backend
```

## PR 12 — Remove root `_core` compatibility aliases

```text
Only after all internal Python and tests use submodules.
```

---

# Success metrics

Track these after each phase:

```text
src/binding total LOC
largest src/binding file LOC
number of root _core symbols
number of _core.debug symbols
number of production _core.engine symbols
number of src/g imports from engine internals
number of public re-exports in g-engine::api.rs
```

Targets:

```text
src/binding total LOC: -40% to -60%
no production binding file > 600 LOC
root _core symbols near zero except submodules
_core.engine production surface < 10 high-level classes/functions
callback_runtime_resources.rs gone or debug-only
schedule.rs gone or debug-only
```

---

# Bottom line

Yes: `src/binding` is too large, too smart, and confusingly named.

The better architecture is:

```text
src/binding = PyO3 adapters
crates/*    = all pure Rust logic
src/g       = public Python API + JAX backend/kernels
```

Renaming to `src/binding` is the right first step, but the real win comes from enforcing the boundary: PyO3 code adapts; crates decide.
