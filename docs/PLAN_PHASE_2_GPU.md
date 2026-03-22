# **Phase 2 GPU Arm: JAX-on-GPU Execution Plan**

## **1. Mission**

The GPU arm exists to make the current Phase 1 math run efficiently on GPU before the project invests in custom kernels.

The immediate objective is not to redesign the statistical methods. The immediate objective is to remove avoidable orchestration overhead, transfer overhead, and synchronization overhead from the existing JAX implementation.

## **2. Current GPU-Relevant Code Map**

### **Runtime Configuration**

* `pyproject.toml` currently pins `jax[cpu]`.
* `src/g/jax_setup.py` enables float64 globally and configures the persistent JAX compilation cache.

### **Orchestration**

* `src/g/engine.py` coordinates sample loading, genotype chunk iteration, compute dispatch, and result formatting.
* Linear chunk output currently computes p-values after a host transfer.
* Logistic chunk output transfers chunk results to host for formatting.

### **Compute Kernels**

* `src/g/compute/linear.py` is compact and structurally simple.
* `src/g/compute/logistic.py` is the primary hot path and contains the highest-value GPU cleanup work.

### **I/O Dependency**

* `src/g/io/plink.py` still creates host NumPy arrays and then calls `jax.device_put` on each chunk.
* This is acceptable for early GPU bring-up, but it is not the intended end state.

## **3. Problems the GPU Arm Is Expected to Solve**

### **Problem 1: The Runtime Is Still CPU-Oriented**

Today the default dependency setup is CPU JAX. The GPU arm must create a reproducible and documented GPU execution path.

### **Problem 2: Too Many Host-Device Round-Trips**

The current pipeline contains explicit `jax.device_get(...)` calls in result formatting and in the hybrid logistic fallback path. Those synchronizations are especially harmful once compute moves to GPU.

### **Problem 3: The Logistic Path Is Not Yet GPU-Friendly in Practice**

The logistic implementation is mathematically structured for JAX, but the hybrid dispatch path still mixes device work with host branching and host result patch-up.

### **Problem 4: Phase 1 Precision Choices May Not Be Phase 2 Throughput Choices**

Phase 1 correctly optimized for float64 parity. The GPU arm must determine what remains in float64 and what can safely move to a performance-oriented mode.

## **4. GPU Arm Non-Goals**

The GPU arm should not:

* introduce custom Triton or CUDA kernels in this phase
* change the regression math without explicit approval
* assume the Rust arm must finish first
* let benchmarking devolve into a single end-to-end wall clock number with no attribution

## **5. Recommended Execution Order**

### **Step 0: Establish a GPU Baseline**

Before changing kernel code, confirm and document:

* which GPU hardware is being targeted
* the installed CUDA-enabled JAX path
* whether JAX sees the device
* compile time versus steady-state runtime
* current memory use and transfer behavior

At this stage, the goal is visibility, not speed.

### **Step 1: Make Device Policy Explicit**

Agents should introduce a clear Phase 2 device configuration story so it is obvious whether a run is:

* CPU parity mode
* GPU parity mode
* GPU performance mode

This can be implemented via configuration, environment variables, CLI flags, or a combination. The exact interface is less important than making the behavior explicit and testable.

### **Step 2: Clean Up Linear Execution First**

Linear regression is the easier proving ground because the kernel in `src/g/compute/linear.py` is compact and easier to reason about.

Recommended improvements:

* keep p-value computation on device until the final transfer
* reduce chunk-level host synchronization in `src/g/engine.py`
* verify stable compilation behavior across chunk sizes
* measure how much speedup is available before touching logistic

This step gives the team a simpler GPU success path and validates the benchmarking harness.

### **Step 3: Refactor Standard Logistic for Device Residency**

Before attacking Firth fallback behavior, improve the standard logistic path:

* reduce unnecessary transfers of chunk outputs
* keep per-variant masks and result assembly on device where practical
* revisit places where explicit matrix inversion can be replaced by solves if profiling supports it
* ensure the implementation remains shape-stable for JIT caching

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

### **Step 5: Introduce a Controlled Precision Strategy**

The GPU arm should not blindly flip the entire engine from float64 to float32. Instead, define explicit modes.

Suggested model:

* **Parity mode:** preserve current float64-first behavior
* **Performance mode:** allow selective lower precision for throughput-sensitive paths

If mixed precision is introduced, document exactly which arrays and operations stay in float64 and why.

### **Step 6: Tune Throughput Parameters**

Once transfers and device residency improve, retune:

* chunk size
* fallback batch size
* data layout assumptions
* any final-chunk padding or shape-stabilization strategy

The current CLI default chunk size of `512` variants is a safe Phase 1 default, not necessarily a good GPU default.

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

### **`src/g/io/plink.py`**

Inspect:

* chunk shape stability
* dtype conversion behavior
* cost of `jax.device_put`

Why:

* even before the Rust arm lands, this file determines how much time the GPU spends waiting for new chunks

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

### **Deliverable 3: Device-Resident Linear Path**

Linear association should have a clean, measurable GPU path with minimized host synchronization.

### **Deliverable 4: Improved Hybrid Logistic Path**

The current host-heavy fallback orchestration should be substantially cleaner and measurably faster.

### **Deliverable 5: Precision and Runtime Policy**

The repository should clearly express how parity-oriented and performance-oriented runs differ.

## **8. Suggested Agent Workflow**

This is the recommended way for implementation agents to attack the GPU arm.

1. Measure first and write down the bottleneck split.
2. Make one transfer-boundary improvement at a time.
3. Re-run parity checks after each substantial refactor.
4. Only retune chunk sizes after transfer boundaries stabilize.
5. Keep changes additive when possible so CPU fallback remains available.

If the measurements contradict this sequence, agents should adapt. The plan should guide the work, not trap it.

## **9. Acceptance Criteria**

The GPU arm is successful when all of the following are true:

* A documented GPU execution path exists and is reproducible.
* The linear path runs with fewer avoidable host-device synchronizations.
* The logistic path shows measurable speed improvements, especially around hybrid fallback behavior.
* Phase 1 correctness gates still pass.
* The remaining GPU bottlenecks are understood well enough to justify or reject the next custom-kernel phase.
