# **Phase 2 GPU Arm: JAX-on-GPU Execution Plan**

## **1. Mission**

The GPU arm exists to make the current Phase 1 math run efficiently on GPU before the project invests in custom kernels.

The immediate objective is not to redesign the statistical methods. The immediate objective is to remove avoidable orchestration overhead, transfer overhead, and synchronization overhead from the existing JAX implementation.

Writing custom kernels is explicitly the last step of the GPU arm, not the first one. The first serious GPU question is: what is the maximum performance available from a carefully optimized JAX implementation with disciplined device residency, stable shapes, minimized synchronization, and reduced host orchestration?

## **1.1 Current Understanding After Rust and Host-Side Work**

Recent work on the Rust and host orchestration lanes changed the GPU plan in important ways.

What we now know:

* The biggest Phase 2 gains are unlikely to come from rewriting BED decode or forcing more logic into Rust.
* `bed-reader` is already a strong native ingestion baseline, so the GPU arm should assume host chunks arrive through that path unless later evidence says otherwise.
* Direct host-side import into JAX is not automatically better than `jax.device_put(...)`; on the current CPU-backed setup, `device_put(...)` was slightly faster than `jnp.from_dlpack(...)` for representative chunk sizes.
* Hybrid logistic fallback orchestration does contain measurable host overhead, but not every attempted host/device refactor helps. GPU work must stay benchmark-driven and be willing to revert regressions.

This means the GPU arm should focus first on maximizing JAX execution quality and minimizing orchestration waste before entertaining custom-kernel work.

## **2. Current GPU-Relevant Code Map**

### **Runtime Configuration**

* `pyproject.toml` currently pins `jax[cpu]`.
* `src/g/jax_setup.py` enables float64 globally and configures the persistent JAX compilation cache.
* The current machine has no CUDA-enabled `jaxlib` installed, so all present measurements remain CPU-backed JAX measurements.

### **Orchestration**

* `src/g/engine.py` coordinates sample loading, genotype chunk iteration, compute dispatch, and result formatting.
* Linear chunk output no longer recomputes p-values after host transfer.
* Logistic chunk output still transfers chunk results to host for formatting.
* Hybrid logistic fallback still contains host-side batching and merge behavior, though some Python overhead has already been reduced.

### **Compute Kernels**

* `src/g/compute/linear.py` is compact and structurally simple.
* `src/g/compute/logistic.py` is the primary hot path and contains the highest-value GPU cleanup work.

### **I/O Dependency**

* `src/g/io/plink.py` still creates host NumPy arrays and then calls `jax.device_put` on each chunk.
* This is acceptable for early GPU bring-up and should remain the default ingestion path until a GPU-backed measurement proves otherwise.

## **3. Problems the GPU Arm Is Expected to Solve**

### **Problem 1: The Runtime Is Still CPU-Oriented**

Today the default dependency setup is CPU JAX. The GPU arm must create a reproducible and documented GPU execution path.

### **Problem 2: Too Many Host-Device Round-Trips**

The current pipeline contains explicit `jax.device_get(...)` calls in result formatting and in the hybrid logistic fallback path. Those synchronizations are especially harmful once compute moves to GPU.

This remains the primary GPU engineering problem.

### **Problem 3: The Logistic Path Is Not Yet GPU-Friendly in Practice**

The logistic implementation is mathematically structured for JAX, but the hybrid dispatch path still mixes device work with host branching and host result patch-up.

Recent cleanup improved this path, but it is still the least GPU-clean part of the engine.

### **Problem 4: Phase 1 Precision Choices May Not Be Phase 2 Throughput Choices**

Phase 1 correctly optimized for float64 parity. The GPU arm must determine what remains in float64 and what can safely move to a performance-oriented mode.

This question should be answered only after the JAX/device residency path is cleaned up enough that we can attribute wins correctly.

## **4. GPU Arm Non-Goals**

The GPU arm should not:

* introduce custom Triton or CUDA kernels before optimized-JAX limits are understood
* change the regression math without explicit approval
* assume the Rust arm must finish first
* let benchmarking devolve into a single end-to-end wall clock number with no attribution
* treat DLPack, direct-to-JAX imports, or lower precision as wins without measurement

## **5. Recommended Execution Order**

### **Step 0: Establish a GPU Baseline**

Before changing kernel code, confirm and document:

* which GPU hardware is being targeted
* the installed CUDA-enabled JAX path
* whether JAX sees the device
* compile time versus steady-state runtime
* current memory use and transfer behavior

At this stage, the goal is visibility, not speed.

Status:

* Not done yet because the repository does not currently have a CUDA-enabled `jaxlib` path configured.
* CPU-backed JAX measurements already suggest which transfer/orchestration boundaries matter, so GPU bring-up should start with those known issues in mind rather than from a blank slate.

### **Step 1: Make Device Policy Explicit**

Agents should introduce a clear Phase 2 device configuration story so it is obvious whether a run is:

* CPU parity mode
* GPU parity mode
* GPU performance mode

This can be implemented via configuration, environment variables, CLI flags, or a combination. The exact interface is less important than making the behavior explicit and testable.

Recommended policy model now:

* **CPU parity mode:** current conservative path
* **GPU parity mode:** float64-first, correctness-oriented GPU execution
* **GPU tuned-JAX mode:** still pure JAX, but allowed to retune chunk size, batching, and synchronization boundaries
* **GPU kernel mode:** future, explicitly post-JAX-optimization phase

### **Step 2: Clean Up Linear Execution First**

Linear regression is the easier proving ground because the kernel in `src/g/compute/linear.py` is compact and easier to reason about.

Recommended improvements:

* keep p-value computation on device until the final transfer
* reduce chunk-level host synchronization in `src/g/engine.py`
* verify stable compilation behavior across chunk sizes
* measure how much speedup is available before touching logistic

This step gives the team a simpler GPU success path and validates the benchmarking harness.

Status:

* Partially done already: linear p-value computation now stays in the compute path rather than being recomputed after host transfer.
* Remaining work is mainly GPU measurement and shape/compilation behavior once CUDA-backed JAX is available.

### **Step 3: Refactor Standard Logistic for Device Residency**

Before attacking Firth fallback behavior, improve the standard logistic path:

* reduce unnecessary transfers of chunk outputs
* keep per-variant masks and result assembly on device where practical
* revisit places where explicit matrix inversion can be replaced by solves if profiling supports it
* ensure the implementation remains shape-stable for JIT caching

This step should be prioritized ahead of deeper Firth work if measurements show the standard path dominates overall wall time.

### **Step 4: Refactor Hybrid Logistic Dispatch**

This is the highest-priority GPU engineering task.

The path centered around `compute_logistic_association_chunk_with_mask` currently:

* computes a heuristic Firth dispatch mask on device and transfers it to host
* transfers standard logistic results and coefficients to host
* uses Python control flow to batch fallback variants
* patches fallback initial coefficients on host
* copies fallback outputs back into host arrays before rebuilding a device result

The GPU arm should aim to remove as much of that host orchestration as possible.

Recommended direction:

* keep fallback masks on device until a host decision is strictly unavoidable
* minimize Python loops over fallback batches
* keep result merging on device
* avoid host patch-up of coefficient tensors unless absolutely necessary

Agents are free to choose the exact refactor strategy as long as the parity contract remains intact.

Status:

* Some host cleanup is already done: less Python list handling, less fallback-only overhead, and an early return for chunks with no fallback variants.
* A focused benchmark now exists in `scripts/benchmark_logistic_fallback.py`.
* The path is still host-influenced enough that it should remain the main GPU cleanup target before any kernel discussion.

### **Step 5: Introduce a Controlled Precision Strategy**

The GPU arm should not blindly flip the entire engine from float64 to float32. Instead, define explicit modes.

Suggested model:

* **Parity mode:** preserve current float64-first behavior
* **Performance mode:** allow selective lower precision for throughput-sensitive paths

If mixed precision is introduced, document exactly which arrays and operations stay in float64 and why.

This step is explicitly downstream of GPU bring-up and optimized-JAX cleanup. It should not be used to mask synchronization or orchestration problems.

### **Step 6: Tune Throughput Parameters**

Once transfers and device residency improve, retune:

* chunk size
* fallback batch size
* data layout assumptions
* any final-chunk padding or shape-stabilization strategy

The current CLI default chunk size of `512` variants is a safe Phase 1 default, not necessarily a good GPU default.

This step should be done only after compile behavior and transfer boundaries are stable enough that throughput tuning is interpretable.

### **Step 7: Only Then Decide Whether Custom Kernels Are Worth It**

Custom kernels should come after all of the following are true:

* a reproducible CUDA-backed JAX path exists
* compile and steady-state costs are separated clearly
* transfer and synchronization boundaries have been cleaned up as far as practical in JAX
* chunk sizing and batching have been retuned for the GPU path
* the remaining bottleneck is clearly inside an optimized JAX compute region rather than around it

If those conditions are not met, custom kernels are premature.

## **6. Exact Code Hotspots to Inspect First**

These files deserve immediate profiling attention.

### **`src/g/engine.py`**

Inspect:

* `build_linear_output_frame`
* `build_logistic_output_frame`
* `compute_logistic_association_with_missing_exclusion`
* the chunk iteration loops in `iter_linear_output_frames` and `iter_logistic_output_frames`

Why:

* this file controls transfer boundaries
* this file decides where formatting happens
* this file determines whether chunk processing is compute-bound or orchestration-bound

This file should also be used to identify where a future GPU path can defer formatting and final host transfer until larger result blocks are ready.

### **`src/g/compute/logistic.py`**

Inspect:

* standard logistic chunk computation
* pre-dispatch Firth heuristic
* host transfer helper functions
* fallback batch construction and merging
* Firth initialization and update flow

Why:

* this file likely dominates runtime on the current benchmark dataset
* this file is also the least GPU-clean part of the current implementation

This file should now be treated as two separate optimization zones:

* standard logistic device-residency cleanup
* hybrid fallback orchestration cleanup

### **`src/g/io/plink.py`**

Inspect:

* chunk shape stability
* dtype conversion behavior
* cost of `jax.device_put`

Why:

* even before the Rust arm lands, this file determines how much time the GPU spends waiting for new chunks

At present, this file should be optimized for predictable host staging rather than ambitious direct-to-JAX import tricks.

## **7. GPU-Specific Deliverables**

The GPU arm should aim to produce the following concrete outputs.

### **Deliverable 1: Reproducible GPU Bring-Up**

Documented installation and runtime behavior for the chosen CUDA-enabled JAX path.

### **Deliverable 2: GPU-Aware Benchmark Harness**

A benchmark path that separates:

* compilation cost
* read cost
* transfer cost
* compute cost
* output formatting cost

It should also distinguish first-run compilation cost from warmed steady-state chunk execution.

### **Deliverable 3: Device-Resident Linear Path**

Linear association should have a clean, measurable GPU path with minimized host synchronization.

### **Deliverable 4: Improved Hybrid Logistic Path**

The current host-heavy fallback orchestration should be substantially cleaner and measurably faster.

This should be demonstrated first with pure JAX plus orchestration cleanup, before any custom kernel proposal is written.

### **Deliverable 5: Precision and Runtime Policy**

The repository should clearly express how parity-oriented and performance-oriented runs differ.

## **8. Suggested Agent Workflow**

This is the recommended way for implementation agents to attack the GPU arm.

1. Measure first and write down the bottleneck split.
2. Make one transfer-boundary improvement at a time.
3. Re-run parity checks after each substantial refactor.
4. Only retune chunk sizes after transfer boundaries stabilize.
5. Keep changes additive when possible so CPU fallback remains available.

Add two explicit rules:

6. Revert any GPU-cleanup idea that regresses the focused benchmark, even if it looks architecturally elegant.
7. Do not propose custom kernels until the optimized-JAX ceiling is documented.

If the measurements contradict this sequence, agents should adapt. The plan should guide the work, not trap it.

## **9. Acceptance Criteria**

The GPU arm is successful when all of the following are true:

* A documented GPU execution path exists and is reproducible.
* The project has a measured optimized-JAX baseline on GPU before any custom-kernel phase begins.
* The linear path runs with fewer avoidable host-device synchronizations.
* The logistic path shows measurable speed improvements, especially around hybrid fallback behavior.
* Phase 1 correctness gates still pass.
* The remaining GPU bottlenecks are understood well enough to justify or reject the next custom-kernel phase.

## **10. Immediate Next Steps**

The next GPU work should be practical and sequential.

1. Install and verify a CUDA-enabled `jaxlib` path on the target GPU machine.
2. Add a GPU-oriented benchmark harness that records compile time, warmed runtime, and chunk-level transfer/formatting overhead.
3. Run the current linear and logistic paths unchanged on GPU to establish a truthful baseline.
4. Profile `src/g/engine.py` and `src/g/compute/logistic.py` to quantify where device synchronization still happens.
5. Optimize standard JAX device residency and fallback orchestration before discussing precision changes.
6. Only after those steps, decide whether there is still a compute-core bottleneck that could justify custom kernels.
