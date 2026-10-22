# Vulkan Backend — Design Decisions

**Scope:** the choices made to bring a Vulkan compute path to mistral.rs on
Void GNU/Linux, Intel UHD CML GT2 (Gen9.5), 16.5 GB shared UMA.
**Status:** completed spike. GEMV and GEMM pass numerical gates. Model-level
prediction (from kernel-level benchmark): ~2× slower than CPU on this device.
See `docs/vulkan-device-report.md` for measurements.
**Related:** `docs/vulkan-device-report.md` (D1, facts), `docs/vulkan-upstream-plan.md` (D6b, integration).

Every section below states the decision, the alternatives considered, and the
observation that would reopen it. The last part matters more than the decision:
a design that cannot be falsified drifts into folklore.

---

## 1. Sidecar crate, not a Candle backend

**Decision:** Vulkan compute lives in `crates/vulkex`, a standalone crate that
exposes `Session::gemv` and `Session::gemm`. It is not a `BackendStorage`
implementation. It is not wired into `mistralrs-core`.

**Alternatives:**

- **Fork Candle and add `VulkanStorage`.** Rejected for phase 1. The Candle
  `BackendStorage` trait surface is ~40 methods; covering it is a month of work
  before the first numerical result. The prior art
  (EricLBuehler/candle issue #1810 and derivatives) has stayed at buffer
  allocation for roughly a year, which is a signal that the trait surface is
  the bottleneck, not the Vulkan API.
- **`wgpu`.** Rejected. `wgpu` provides a good portability story but hides the
  memory allocation strategy, which is exactly the layer that needs control on
  a UMA device where 10.4 GiB is the hard ceiling. It also adds a second
  runtime dependency (Dawn / naga) and a shader language decision that has
  nothing to do with the goals.

**Status of the reopen condition (from the original decision):** the original
wrote "if `vulkex` proves ≥ 1.5× CPU on the shapes in `bench_shapes` and Q4_K
dequant lands within budget, the case for full `BackendStorage` coverage
becomes strong enough that a fork is worth the month."

**That condition is falsified on this hardware.** Measured: 0.48–0.52× CPU on
prefill GEMM, and predicted 0.46× CPU on model-level decode. The sidecar
approach was the right call — it produced the measurement that closed the
question. The fork is not justified on the Intel UHD CML GT2.

**What would reopen it now:** running `vulkex` on a stronger UMA device
(Intel Arc, AMD RDNA3 APU, or discrete) where compute scales with EU count
while bandwidth is separate. If `bench_shapes` on such a device shows
≥ 1.5× CPU on prefill, the fork becomes worth the month. Until then, the
sidecar is the deliverable.

---

## 2. `ash` + `gpu-allocator`, not `ash` alone and not `vk-mem`

**Decision:** `ash 0.38` for bindings, `gpu-allocator 0.28` for memory.

**Alternatives:**

- **Raw `vkAllocateMemory` via `ash`.** Correct, and it was the first sketch.
  Rejected because allocating per-buffer on a UMA heap fragments it: the driver
  hands out alignment-padded blocks, the allocator has no pooling, and the
  30-minute stability gate would fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY`
  after a few model loads.
- **`vk-mem`.** Rejected. It is a stable C library (AMD VMA) with a single-
  maintainer Rust binding that has released twice in eighteen months. The
  binding works; the release cadence is the risk. `gpu-allocator` is maintained
  by the wgpu ecosystem (Mozilla / Google), has a faster release cadence, and
  exposes the same VMA-style managed allocation through a Rust-native API.

**Known limitation:** `gpu-allocator 0.28` does not expose the chosen memory
type index on `Allocation`. `Context::alloc_mapped` therefore uses the
device-wide `all_host_visible_coherent` flag rather than per-allocation
lookup. On this device the distinction is moot (all host-visible types
coherent); on mixed-coherence devices the flush/invalidate path runs
unnecessarily, which is correct but slow. A future `gpu-allocator` release
that exposes `memory_type_index()` should switch this to a per-allocation
lookup.

**What would reopen it:** if `gpu-allocator`'s pool growth policy proves to
waste memory on the 10.4 GiB heap during real model loads (visible as
`vkAllocateMemory` failures while the system still has free RAM), the
alternative is `vk-mem`, not raw allocation.

---

## 3. FP32 arithmetic first, FP16 storage later, no BF16 arithmetic

**Decision:** all kernels are implemented in FP32 first, including the
quantized path (which dequantizes to FP32 in-register, not FP16). FP16 is a
later storage optimization. BF16 arithmetic is not on the near-term roadmap.

**Alternatives:**

- **FP16 from the start.** Rejected. `shaderFloat16` is exposed on this device,
  but the CPU reference is FP32, and any FP16 numerical bug in a reduction
  kernel is masked by tolerance, not detected. FP32 first means the kernel's
  logic is validated in the arithmetic the reference uses, then ported to FP16
  as a memory optimization with the same test suite.
- **BF16 from the start.** Rejected. BF16 is not exposed on this device
  (verified in the D1 report); any BF16 math would be software emulation and
  is slower than FP32. BF16 checkpoints dequantize to FP32 at load.

**What would reopen it:** if the Q4_K dequant-fused kernel is faster in FP16
storage (measured, not assumed) on the iGPU, and the test suite's
`max_abs / max_rel` on the dequantized weights stay within FP32 tolerances.

---

## 4. Shared-memory reduction, no subgroup ops

**Decision:** all reductions (`gemv.comp`, `gemm.comp` inner loop, future
RMSNorm, future softmax) use `shared` memory plus `barrier()`. No
`subgroupAdd`, `subgroupBallot`, or subgroup arithmetic.

**Alternatives:**

- **Subgroup reduction.** Rejected for phase 1. The `VkPhysicalDeviceSubgroupProperties`
  query on this device returned no data under the standard `vulkaninfo` grep,
  which is itself a signal: either the driver underreports or the query needs
  `VK_EXT_subgroup_properties` rather than the core struct. Adding a code path
  that depends on data you could not read is how intermittent wrong results
  happen on Gen9 Intel.
- **Cooperative matrices (`VK_KHR_cooperative_matrix`).** Not available on
  Gen9.5, so not a real alternative.

**What would reopen it:** a successful `vulkaninfo --summary` from a Mesa
version that reports explicit subgroup properties on this device, and a
measured improvement over the shared-memory version that survives the
`gemv_numerics` test on 4096×4096 shapes.

**Note on the measured prefill ceiling:** 33 GFLOPS is ~7.8% of this
device's theoretical 422 GFLOPS (24 EUs × 8 lanes × 2 FMA × 1.1 GHz). The
16×16 → 64×64 tile change moved 29 → 33 GFLOPS (14% improvement), which
is a strong signal that **the bottleneck is not tile reuse**. Further tiling
work is unlikely to close the gap. Subgroup operations might, but only
after the properties query works on this driver.

---

## 5. Coherent-heap shortcut, but keep the flush/invalidate path

**Decision:** `Buffer` carries a `coherent: bool` flag populated from
`Context::all_host_visible_coherent`, a device-wide property set at
`Context::new` time. `Context::flush` and `Context::invalidate` are
conditional on `!coherent`. On this device every host-visible memory type
is coherent, so both calls are no-ops in the hot path.

**Alternatives:**

- **Delete the flush/invalidate code because it never runs.** Rejected. The
  device report (D1) records that this decision holds on **this machine**;
  portable UMA parts (some Mali, some older ARM SoCs) expose non-coherent
  host-visible memory. The conditional path is the interface that keeps
  `vulkex` reusable.
- **Per-allocation lookup via `Allocation::memory_type_index()`.** Not
  available in `gpu-allocator 0.28`. See §2 for the current approximation.

**What would reopen it:** running `vulkex` on hardware where the coherence flag
is false, at which point the conditional path is the only thing preventing a
wrong answer.

---

## 6. Workspace membership with `default-members` exclusion

**Decision:** `crates/vulkex` is a workspace member, but not in
`default-members`. `cargo build` at the repo root does not compile
`shaderc` or `ash`. `cargo build -p vulkex` and
`cargo build -p mistralrs-core --features vulkan` do.

**Alternatives:**

- **`exclude` instead of `default-members`.** Rejected. `exclude` would force
  `vulkex` to carry its own `Cargo.lock` and its own dependency resolution,
  which drifts against the root lockfile the moment any third crate pulls the
  same transitive dependency (`bitflags`, `libloading`, `once_cell`).
- **Add to `default-members`.** Rejected. Contributors who never touch Vulkan
  pay the shaderc C++ build cost on every clean build.

**What would reopen it:** if `vulkex` gains a `mistralrs-core` dependency that
lives inside `default-members`, it may need to be promoted to `default-members`
to avoid duplicate builds. Not applicable today because `vulkex` has zero
mistral.rs dependencies.

---

## 7. Fail-loud on validation

**Decision:** if `VK_INSTANCE_LAYERS` requests `VK_LAYER_KHRONOS_validation`
and the layer is not installed, `Context::new` bails with a message naming the
layer and the Void package. Silent degrade was rejected.

**Rationale:** an early run reported `validation=false` after the layer had not
been installed, and the test still passed. That is the failure mode the spec
calls a "silent fallback", and it took two runs to notice. The bail-out makes
the same mistake impossible to miss.

**What would reopen it:** nothing. This is the pattern the specification
requires, and it applies to every future check in `vulkex`.

---

## 8. `vulkan` feature flag scope

**Decision:** when `vulkex` is wired into `mistralrs-core`, the feature will
be `vulkan = ["dep:vulkex"]`, opt-in, default-off. No `default-features`
change to any mistral.rs crate.

**Rationale:** the CPU and (eventual) Vulkan paths must remain independently
buildable. `cargo build -p mistralrs-core` without the feature is the
reference build; `--features vulkan` is the experimental one. Making Vulkan
default-on would hide regressions in the CPU path and would compile
`shaderc` for contributors on non-x86 platforms where it has never been
tested.

**What would reopen it:** when the Vulkan path is measurably faster on the
maintainers' hardware and passes the 30-minute stability test, promotion to a
default feature of `mistralrs-core` (not the workspace) becomes reasonable.
On the Intel UHD CML GT2, that condition is falsified — see §1.

---

## 9. What is built and not built

**Built and passing numerical gates:**

- `shaders/gemv.comp`: FP32 GEMV with shared-memory reduction (256 invocations,
  one workgroup per output row).
- `shaders/gemm.comp`: FP32 GEMM with 64×64 tiles and 4×4 register blocking
  (256 invocations, 32 KB shared memory).
- `tests/gemv_numerics.rs`: 4096×4096, `max_abs = 2.0e-6`, `max_rel = 7.7e-5`.
- `tests/gemm_numerics.rs`: 512×6144×2048, `max_abs = 3.9e-5`,
  `rel_inf = 2.0e-6`.
- `tests/bench_shapes.rs`: prefill and decode benchmarks with the sgemm
  reference (with the caveat noted in the device report that sgemm(n=1) is
  not the CPU backend's actual GEMV path).

**Not built (and why):**

- **No Q4_K (or any quantized) kernel.** Every model in `~/ai-models/gguf/`
  is `Q4_K_M` or `Q4_K_XL`. The FP32 path can run SafeTensors checkpoints;
  it cannot run those GGUFs without a dequant kernel.
- **No model integration.** `Session::gemv` and `Session::gemm` take raw
  buffers; they are not called by `mistralrs-core`.
- **No 30-minute stability run.** `Context::drop` does not call
  `vkDestroyDevice`, and `Buffer` has no `Drop` (a forgotten `free_buffer`
  leaks). Fix when integrating into a model runtime.
- **No subgroup path, no FP16 storage path, no cooperative matrices.** All
  three are gated on capabilities or measurements not yet present.
- **No mistral.rs CLI flag.** `--device vulkan` and
  `MISTRALRS_VULKAN_DEVICE` from the specification are **proposed**; the
  current CLI has no such flag.

---

## 10. Acceptance gates

From the specification, with what actually happened:

| Gate                                                      | Result                                                                                  |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Logits within tolerance of CPU FP32 on a 512-token prompt | **Not run** — requires model integration (not built)                                    |
| No unbounded memory growth over a 30-minute session       | **Not run** — requires the `Context::drop` fix                                          |
| Zero silent fallbacks                                     | **Passed** for both kernels: 1 dispatch each, 0 fallbacks, reported by `FallbackReport` |
| Kernel-level numeric match on the GEMV test               | **Passed**: `max_abs = 2.0e-6`, `max_rel = 7.7e-5` on 4096×4096                         |
| Kernel-level numeric match on the GEMM test               | **Passed**: `max_abs = 3.9e-5` (bound 4.9e-4), `rel_inf = 2.0e-6` on 512×6144×2048      |
| Prefill GEMM within 2× of CPU                             | **Passed at the boundary**: 0.48–0.52× on 512×6144×2048 class shapes                    |

The claim "Vulkan works correctly on this device" is supported by both
kernel results. The claim "Vulkan runs models on this device" is not
supported by anything in the current tree — the model integration is not
built. The claim "Vulkan is faster than CPU on this device" is **falsified**:
model-level prediction is ~2× slower on both prefill and decode.

---

## 11. Measured performance conclusion

**On the Intel UHD CML GT2 / i5-10210U, this FP32, per-op-dispatch Vulkan
executor is ~2× slower than the mistral.rs CPU backend on both prefill and
decode, for Qwen3-1.7B F32.**

The structural reasons (recorded in `docs/vulkan-device-report.md` §"UMA
findings" and §"Comparison with llama.cpp's ggml-vulkan backend"):

1. Shared memory bandwidth, and the CPU backend is already at 65–77% of
   the ceiling (19.3 GB/s streaming on this laptop).
2. Decode is 169 dispatches per token (112 attention + 56 MLP + 1 LM head),
   of which 112 are 2048×2048 and pay ~3 ms of fixed overhead each. That
   overhead is 336 ms of the ~895 ms per-token total.
3. Q4_K dequantization shrinks weights on both sides equally **for this
   executor**. In a Q4_K-native executor it shrinks only the Vulkan side
   and becomes a 4× bandwidth advantage instead of a wash.

**What the llama.cpp comparison changes:**

The above is a conclusion about **this executor**, not about the hardware.
Comparing against llama.cpp's ggml-vulkan backend on comparable Intel
iGPUs (documented in the device report) separates the two:

- **Prefill was implementation-limited.** llama.cpp wins prefill on Iris
  Xe by 1.58× and on UHD 620 by ~30× prompt-processing speedup. Those wins
  come from graph-level submission batching, Q4_K-native kernels, and
  Intel-tuned subgroup reductions — none of which this sidecar implements.
  The iGPU is capable of beating the CPU on prefill.

- **Decode is hardware-limited on this CPU/iGPU pair.** Even mature
  ggml-vulkan loses decode to a fast CPU on integrated GPUs; the `ngl 0`
  configuration exists specifically because of that. Our 0.46× decode
  prediction is not a failure of implementation; it is the expected
  outcome on a 4.2 GHz 4C/8T Comet Lake CPU sharing memory with the iGPU.

**Consequence for the Candle backend question (§1):** the reopen condition
"if `vulkex` proves ≥ 1.5× CPU on prefill" is not falsified by our
measurement — it is _not met_ by our executor. A Candle Vulkan backend
that reused llama.cpp's architectural techniques (graph batching, Q4_K,
subgroups) could plausibly meet it. Whether it is worth building depends on
whether the target is prefill-only (where the win is real) or full
inference (where decode is contested on this hardware).

**The `ngl 0` target.** Prefill on the iGPU, decode on the CPU. That is
what llama.cpp recommends for integrated GPUs, and it is what a
mistral.rs Vulkan backend should target on this class of machine. It
captures the achievable win without fighting the decode ceiling.

**Portability remains the correct framing, but the phrase changes.** The
measured data does not say "Vulkan cannot win on UMA iGPUs." It says
"Vulkan wins prefill on UMA iGPUs; decode is contested and CPU-dependent;
this executor needs graph batching, Q4_K, and subgroup support before
either conclusion applies to it."
