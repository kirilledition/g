# Active Code Review Task Plan

This file is the human-readable source for `docs/code-review.tasks.json`.
Keep it limited to live work that Codex task-farm workers should be allowed to
pick up. Dated audits and superseded reviews belong under `docs/archive/`.

# P0 Release Blockers

## 1. JIT the binary variant-major score path

Binary score-only variant-major chunks should execute as one fused JAX program.
The current wrapper in `src/g/compute/regenie2_binary/api.py` calls the
variant-major score and correction helpers without a top-level JIT boundary,
and `src/g/engine/callbacks.py` reaches that wrapper directly.

Target files: `src/g/compute/regenie2_binary/api.py`,
`src/g/compute/regenie2_binary/score.py`,
`src/g/engine/callbacks.py`, and `tests/test_regenie2_binary.py`.

**Guidance**

Add an explicitly jitted variant-major score entry point. Keep the Firth path
as score JIT, host candidate-count decision when correction is enabled, then
jitted fixed-shape correction. Score-only binary chunks should not pay for a
host candidate synchronization.

---

## 2. Remove the multi-binary traits-by-variants-by-samples intermediate

The multi-binary score path in `src/g/compute/regenie2_binary/score.py` risks
materializing a conceptual `traits x variants x samples` tensor. That is not a
safe design for biobank-scale runs.

Target files: `src/g/compute/regenie2_binary/score.py` and
`tests/test_regenie2_binary.py`.

**Guidance**

Rewrite the contractions so weighted sums and scores use `variants x samples`
and `traits x samples` inputs directly. The projection result may be
`traits x variants x covariates`, but no computation should require a
`traits x variants x samples` buffer.

---

## 3. Fix O(T x N^2) complete-case multi-phenotype alignment

The complete-case intersection in `src/sample.rs` builds vectors of keys and
checks membership with repeated `Vec::contains`. For T traits and N samples,
that becomes O(T x N^2).

Target files: `src/sample.rs` and `tests/test_tabular.py`.

**Guidance**

Use the existing `HashMap<AlignedSampleKey, usize>` values or explicit
`HashSet` membership for the complete-case intersection. Preserve the current
complete-case semantics and add a regression test that would have exercised the
slow membership path.

---

## 4. Fail by default on binary null logistic non-convergence

Binary null logistic fitting records convergence in chromosome state, but the
pipeline can continue and emit masked rows instead of failing the run. For the
default scientific path, non-convergence should be a clear error.

Target files: `src/g/types.py`, `src/g/interface/options.py`,
`src/g/interface/config.py`, `src/g/compute/regenie2_binary/state.py`,
`src/g/engine/callbacks.py`, and `tests/test_regenie2_binary.py`.

**Guidance**

Add a policy such as `--g-null-logistic-nonconvergence fail|warn`, defaulting
to fail. Check the scalar or trait-vector convergence state once per binary
chromosome after state preparation. A host scalar/vector sync per chromosome is
acceptable here.

---

## 5. Emit NaN for invalid binary score statistics

When score variance is invalid but the null logistic model converged,
`src/g/compute/regenie2_binary/score.py` can emit `CHISQ = 0` and
`LOG10P` near zero while `BETA` and `SE` are NaN. Invalid statistic rows should
have invalid statistics.

Target files: `src/g/compute/regenie2_binary/score.py` and
`tests/test_regenie2_binary.py`.

**Guidance**

Use the same statistic-validity mask for `BETA`, `SE`, `CHISQ`, and `LOG10P`.
The extra-code path can still mark failed rows as `TEST_FAIL`.

# P1 High Priority

## 6. Add strict hard-crash resume scan and repair

Graceful resume uses manifest chunk commits, but a hard crash after Arrow
rename and before manifest finalization can leave durable chunk files that are
not listed in the manifest.

Target files: `src/g/io/output.py`, `src/output/resume.rs`,
`src/output/manifest.rs`, `src/output/session.rs`, and
`tests/test_io_output.py`.

**Guidance**

Implement one deterministic strict repair path: scan Arrow chunk metadata,
validate schema and execution identity, repair the manifest, then resume. Fast
resume may keep trusting manifest commits only, but filename-only inference
must not be used for strict repair.

---

## 7. Avoid sparse Firth mask H2D transfer for score-only paths

The sample-major binary callback still transfers
`chunk_stats.is_rare_sparse_firth_candidate` to device even when the correction
plan is score-only. The variant-major path already avoids this transfer.

Target files: `src/g/engine/callbacks.py` and
`tests/test_regenie2_pipeline.py`.

**Guidance**

Apply the score-only conditional to single and multi-binary sample-major paths.
Preserve Firth behavior by transferring the mask only when approximate Firth
can use it.

---

## 8. Use native dosage sums and square sums in kernels

Rust chunk stats already carry dosage sums and square sums. Linear kernels
currently reread genotype matrices to compute normalized sum of squares, and
binary genotype flipping recomputes allele counts on device.

Target files: `src/g/compute/regenie2_linear/score.py`,
`src/g/compute/regenie2_binary/score.py`,
`src/g/compute/common/genotype.py`, `src/g/engine/callbacks.py`,
`src/genotype/preprocess.rs`, and `tests/test_regenie2_binary.py`.

**Guidance**

Pass native `dosage_sum` and square-sum arrays where available. Keep fallback
device reductions for non-native tests and direct compute callers.

---

## 9. Group phenotypes by identical sample and covariate alignment

Default multi-phenotype mode is semantically honest but performance-limited
because per-phenotype semantics can force repeated BGEN passes.

Target files: `src/sample.rs`, `src/g/engine/native_dispatch.py`,
`src/g/engine/regenie2_pipeline.py`, `src/g/execution_plan.py`, and
`tests/test_regenie2_pipeline.py`.

**Guidance**

Fingerprint each phenotype alignment from sample indices, covariate matrix,
and prediction alignment. Run one BGEN pass per identical group. Keep
complete-case mode explicit and non-default because it changes sample
inclusion semantics.

---

## 10. Improve telemetry buffering and stream ownership

Python telemetry currently writes `events.jsonl` and `progress.jsonl`; Rust
tracing writes `log_file` and `trace_file`. The distinction is useful but easy
to misread, and Python opens the JSONL file for each event.

Target files: `src/g/engine/telemetry.py`, `src/g/runner.py`,
`tests/test_telemetry.py`, and `docs/logging-setup.md`.

**Guidance**

Keep progress mode low overhead. Add buffered or session-owned file handles for
profile and trace events, and document which process writes each stream. Do not
make Python and Rust append to the same file unless there is one shared writer.

# P2 Follow-Up

## 11. Reduce writer metadata and result-array copies

The Rust writer path clones chunk metadata and stats and copies result arrays
before queuing native chunk writes. This is not a correctness bug, but it is
visible memory traffic at scale.

Target files: `src/output/session.rs`, `src/output/writer.rs`,
`src/g/engine/callbacks.py`, and `tests/test_io_output.py`.

**Guidance**

Consider passing a Rust-owned native chunk handle through the Python callback
and accepting ownership of contiguous NumPy buffers where safe. Measure before
and after; keep the existing path as a correctness baseline until the ownership
contract is clear.

---

## 12. Harden Oxford sample whitespace parsing

Oxford `.sample` files may contain tabs or mixed whitespace. A single-space CSV
delimiter plus empty-field filtering is less robust than splitting sample-file
lines by arbitrary whitespace.

Target files: `src/sample.rs` and `tests/test_tabular.py`.

**Guidance**

Use `split_whitespace` or an equivalent strict parser for Oxford sample files.
Keep phenotype and covariate TSV parsing documented separately.

---

## 13. Preserve p-value dtype policy

`src/g/compute/common/pvalue.py` downcasts chi-square values to float32 before
tail conversion. That may be acceptable for score-only output, but it discards
precision from float64 Firth internals.

Target files: `src/g/compute/common/pvalue.py` and
`tests/test_regenie2_binary.py`.

**Guidance**

Make dtype behavior explicit. Preserve float64 inputs where they occur, or add
separate float32 and float64 conversion helpers with parity tests for extreme
GWAS tails.
