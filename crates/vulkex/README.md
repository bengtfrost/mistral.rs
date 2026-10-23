# vulkex

Standalone Vulkan compute executor sidecar for mistral.rs. Not a Candle
`BackendStorage` implementation. Not wired into `mistralrs-core`. Not
reachable from the CLI.

A completed portability spike with honest measurements: the kernels are
numerically correct and the toolchain works, but on the reference machine
(Intel UHD CML GT2 / i5-10210U) this FP32, per-op-dispatch executor is
**~2× slower than the CPU backend** on both prefill and decode. The reasons
are architectural and documented. The crate is the measurement platform
for a future Vulkan backend, not that backend.

## Status

| Component                                        | Status        | Result                                                                        |
| ------------------------------------------------ | ------------- | ----------------------------------------------------------------------------- |
| `gemv` (FP32, shared-memory reduction)           | **Done**      | `max_abs = 2.0e-6`, `max_rel = 7.7e-5` on 4096×4096, validation active        |
| `gemm` (FP32, 64×64 tile, 4×4 register blocking) | **Done**      | `max_abs = 3.9e-5`, `rel_inf = 2.0e-6` on 512×6144×2048, validation active    |
| Prefill benchmark (Qwen3-1.7B shapes)            | **Done**      | 33 GFLOPS Vulkan vs 66 GFLOPS CPU (0.48–0.52×)                                |
| Decode benchmark (GEMV, M=1)                     | **Done**      | 3.8–13.1 GB/s Vulkan vs 19.3 GB/s CPU backend                                 |
| Model-level comparison                           | **Predicted** | ~0.46× CPU decode, ~0.5× CPU prefill (from per-token dispatch accounting)     |
| Q4_K dequant-fused kernel                        | **Not built** | Every model in `~/ai-models/gguf/` uses Q4_K; this is the kernel that matters |
| Graph-level submission batching                  | **Not built** | The largest gap to a competitive backend (per-op dispatch overhead dominates) |
| Model integration in `mistralrs-core`            | **Not built** | `vulkan` Cargo feature and CLI flag are deferred                              |
| KV cache in Vulkan buffers                       | **Not built** | Deferred until integration                                                    |
| 30-minute stability run                          | **Not built** | Blocked on `Context::drop` fix                                                |

## Measured results

Captured 2026-10-22 on the reference machine (`performance` governor, AC
power, no other applications). CPU reference is `matrixmultiply::sgemm`,
the microkernel Candle's CPU backend wraps.

### Prefill (GEMM, M = 512)

| Shape           | CPU         | Vulkan      | Ratio |
| --------------- | ----------- | ----------- | ----- |
| 512×6144×2048   | 63.7 GFLOPS | 33.1 GFLOPS | 0.52× |
| 512×2048×6144   | 65.9 GFLOPS | 33.0 GFLOPS | 0.50× |
| 512×2048×2048   | 65.6 GFLOPS | 32.8 GFLOPS | 0.50× |
| 512×151936×2048 | 68.9 GFLOPS | 33.0 GFLOPS | 0.48× |

Vulkan achieves 7.8% of the iGPU's 422 GFLOPS theoretical peak. A 4×
tile-size change (16×16 → 64×64) improved throughput by 14% — the
bottleneck is not tile reuse.

### Decode (GEMV, M = 1)

| Shape       | Vulkan GB/s |
| ----------- | ----------- |
| 2048×2048   | 3.8         |
| 2048×6144   | 11.0        |
| 6144×2048   | 8.2         |
| 151936×2048 | 13.1        |

**The CPU backend streams weights at 19.3 GB/s** (Qwen3-1.7B F32, measured
with `mistralrs bench --cpu`). Vulkan loses on every shape except the LM
head, and loses badly on the small 2048×2048 shapes that dominate per-token
cost — those are dispatch-overhead-bound, not bandwidth-bound.

### Model-level prediction

Weighted by per-token workload shape (28 layers × 4 attention projections,
MLP up/down, one LM head per token), the predicted end-to-end result is
**~1.1 T/s decode vs 2.4 T/s on CPU**. Not a prediction of failure: a
prediction that this executor, on this hardware, loses on both axes.

## Why it exists

See [`docs/vulkan-design-decisions.md`](../../docs/vulkan-design-decisions.md)
for the full rationale. The short version:

1. **Candle has no Vulkan backend.** Adding one is a `BackendStorage`
   implementation (~40 methods) plus extensions across `candle-core` and
   `candle-nn`. That is a month of work before the first numerical result.
2. **A sidecar delivers the same measurement in a week.** The kernels, the
   toolchain, the numerical gates, and the performance numbers are all
   available without committing to the framework's trait surface.
3. **The measurement decided the framework question.** The result — Vulkan
   is slower on this class of hardware, for reasons that are architectural
   rather than shader-level — is what closes the "should we build the
   Candle backend?" question. The sidecar produced that answer at 1/20th
   the cost.

## A note on platform support

Candle's supported backends are CPU, CUDA, Metal, and WASM. On most
consumer laptops — including the reference machine for this crate — the
only usable path is CPU. That is not what this crate is; it is the
measurement of what a Vulkan path would have to be.

Two observations recorded in `docs/vulkan-design-decisions.md` §1:

- **The gap is a design choice, not a technical necessity.** llama.cpp
  maintains hand-written backends for six GPU APIs; Burn compiles one
  kernel language (CubeCL) to CUDA, ROCm, Metal, Vulkan, WebGPU, and CPU.
  Both prove the platform breadth is achievable; neither was built by
  simply adding a backend to a minimalist tensor library.

- **The gap is less likely to close from upstream now.** Nvidia's
  acquisition of Hugging Face (announced 2026-08-27, expected to close
  H1 2027) does not force Candle onto CUDA, but every dollar spent on
  Vulkan or ROCm is a dollar not spent on the CUDA path. Nvidia's public
  commitments argue against overt degradation; the effect is on the
  direction of default effort, not on day-one support.

The conclusion for this crate: build what is missing, do not wait for
upstream to change.

## What it runs on

Validated on:

- Intel UHD Graphics CML GT2 (Gen9.5, PCI `0x9b41`)
- Mesa 26.2.5 / ANV
- Void GNU/Linux, kernel 6.18

CI additionally exercises the kernels on Mesa lavapipe (Ubuntu 22.04 and
24.04) under `VK_LAYER_KHRONOS_validation`. See
[`docs/vulkan-device-report.md`](../../docs/vulkan-device-report.md) for the
full device report, including the llama.cpp comparison that separates
"implementation limit" from "hardware limit."

## Build and test

```sh
cargo build -p vulkex

# Numerical gates (validation layer required)
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  cargo test -p vulkex --test gemv_numerics -- --nocapture
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  cargo test -p vulkex --test gemm_numerics -- --nocapture

# Benchmarks (release build; --ignored skips them in normal test runs)
cargo test --release -p vulkex --test bench_shapes -- --ignored --nocapture
```

`cargo build` at the repository root does **not** compile `vulkex` or its
`shaderc` dependency; `crates/vulkex` is excluded from `default-members`.
Device selection is by `VULKEX_PCI_ID` (hex, with or without `0x`) or, if
unset, the first enumerated device.

## What is not built, and why

- **No Q4_K dequant-fused kernel.** Every GGUF in `~/ai-models/gguf/` is
  Q4_K_M or Q4_K_XL. The FP32 path cannot run them without load-time
  dequantization that inflates a 5 GB GGUF to 36 GB FP32 — beyond the
  10.4 GiB iGPU budget. This is the single most important missing kernel.
- **No graph-level submission batching.** The dominant cost on small
  shapes. For a 2048×2048 GEMV, ~3 ms of ~4.46 ms is fixed submission
  overhead (command buffer allocation, descriptor update, fence wait).
  Across 112 attention dispatches per token, that is ~336 ms per token
  before any weight is read.
- **No subgroup reductions.** The `VkPhysicalDeviceSubgroupProperties`
  query returned no data on this driver, so the shared-memory path is
  used unconditionally.
- **No model integration.** `Session::gemv` and `Session::gemm` take raw
  buffers; they are not called by `mistralrs-core`.
- **No 30-minute stability run.** `Context::drop` does not call
  `vkDestroyDevice`, and `Buffer` has no `Drop` (a forgotten
  `free_buffer` still leaks).

## The target configuration on this hardware

Prefill through Vulkan, decode through CPU — the `ngl 0` split that
llama.cpp recommends for integrated GPUs. On this class of machine,
prefill is winnable with a graph-batched, Q4_K-native executor (llama.cpp
demonstrates ~1.6× on Gen12 and ~30× prompt-processing speedup on Gen9.5).
Decode is contested even with a mature backend; the CPU's 19.3 GB/s
weight streaming is not something the iGPU's shared memory controller can
beat.

The full reasoning, including the comparison with llama.cpp's
ggml-vulkan backend and the resulting reassessment of the design
decisions, is in
[`docs/vulkan-design-decisions.md`](../../docs/vulkan-design-decisions.md) §11
and
[`docs/vulkan-device-report.md`](../../docs/vulkan-device-report.md)
§"Comparison with llama.cpp's ggml-vulkan backend".

## Documentation

- [`docs/vulkan-design-decisions.md`](../../docs/vulkan-design-decisions.md) — rationale for sidecar vs Candle backend, and the platform-coverage note
- [`docs/vulkan-device-report.md`](../../docs/vulkan-device-report.md) — Phase 0 device facts, kernel results, benchmarks, llama.cpp comparison
- [`docs/vulkan-upstream-plan.md`](../../docs/vulkan-upstream-plan.md) — three-PR sequence to upstream, with measured preconditions
