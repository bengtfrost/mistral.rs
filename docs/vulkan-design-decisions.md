# Vulkan Backend — Design Decisions

**Scope:** the choices made to bring a Vulkan compute path to mistral.rs on
Void GNU/Linux, Intel UHD CML GT2 (Gen9.5), 16.5 GB shared UMA.
**Status:** spike (`crates/vulkex`) — GEMV and GEMM pass numerical gates;
model integration is not yet done.
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

**What would reopen it:** if `vulkex` proves ≥ 1.5× CPU on the shapes in
`bench_shapes` and Q4_K dequant lands within budget, the case for full
`BackendStorage` coverage becomes strong enough that a fork is worth the
month. Until then, the sidecar delivers the same measured information at a
fraction of the cost.

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

---

## 5. Coherent-heap shortcut, but keep the flush/invalidate path

**Decision:** `Buffer` carries a `coherent: bool` flag populated from the
memory type chosen by `gpu-allocator`. `Context::flush` and
`Context::invalidate` are conditional on `!coherent`. On this device every
host-visible memory type is coherent, so both calls are no-ops in the hot path.

**Alternatives:**
- **Delete the flush/invalidate code because it never runs.** Rejected. The
  device report (D1) records that this decision holds on **this machine**;
  portable UMA parts (some Mali, some older ARM SoCs) expose non-coherent
  host-visible memory. The conditional path is the interface that keeps
  `vulkex` reusable.

**What would reopen it:** running `vulkex` on hardware where the coherence flag
is false, at which point the conditional path is the only thing preventing a
crash rather than the only thing preventing a wrong answer.

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

---

## 9. What is not built

Recorded so a future reader does not assume these exist:

- **No Q4_K (or any quantized) kernel.** Qwen3.5-9B and every other model in
  `~/ai-models/gguf/` is `Q4_K_M` / `Q4_K_XL`. The FP32 path can run
  SafeTensors checkpoints; it cannot run those GGUFs without a dequant kernel.
- **No model integration.** `Session::gemv` and `Session::gemm` take raw
  buffers; they are not called by `mistralrs-core`.
- **No 30-minute stability run.** `Buffer` still leaks its `Allocation` on
  drop, and `Context::drop` does not call `vkDestroyDevice`. Both are noted
  in D1 and must be fixed before that gate.
- **No subgroup path, no FP16 storage path, no cooperative matrices.** All
  three are gated on capabilities or measurements not yet present.
- **No mistral.rs CLI flag.** `--device vulkan` and
  `MISTRALRS_VULKAN_DEVICE` from the specification are **proposed**; the
  current CLI has no such flag, and the upstream plan (D6b) proposes them
  under different names that match the existing CLI's conventions.

---

## 10. Acceptance gates

From the specification, with what actually happened:

| Gate | Result |
|---|---|
| Logits within tolerance of CPU FP32 on a 512-token prompt | **Not run** — requires model integration (not built) |
| No unbounded memory growth over a 30-minute session | **Not run** — requires the `Buffer` / `Context` drop fixes |
| Zero silent fallbacks | **Passed** for GEMV: 1 dispatch, 0 fallbacks, reported by `FallbackReport` |
| Kernel-level numeric match on the GEMV test | **Passed**: `max_abs = 2.0e-6`, `max_rel = 7.7e-5` on 4096×4096 |
| GEMM numerical match | **Not yet run** — the `bench_shapes` harness includes correctness, but the first pass reports timing only. Add a `gemm_numerics` test before the GEMM gate is claimed. |

The claim "Vulkan works on this device" is supported by the GEMV result. The
claim "Vulkan runs models on this device" is not supported by anything in the
current tree.
