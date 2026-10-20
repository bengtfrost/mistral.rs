# Vulkan Backend — Upstream Plan

Three PRs, in order. Do not open PR 2 before PR 1 merges; do not open PR 3
before PR 2 merges and Q4_K is validated.

---

## PR 1 — Add `crates/vulkex` as an opt-in workspace member

**Target:** `EricLBuehler/mistral.rs` at revision `465191d0` and current master.

**Contents:**
- `crates/vulkex/` (the spike, as it stands at D6 commit)
- `Cargo.toml`: add `"crates/vulkex"` to `members`, mirror-minus-vulkex to
  `default-members`
- `docs/vulkan-device-report.md` (D1)
- `docs/vulkan-design-decisions.md` (D6a)
- `docs/vulkan-upstream-plan.md` (D6b, this file)

**What is not in this PR:**
- No changes to `mistralrs-core`, `mistralrs-cli`, or any other crate.
- No new features on any existing crate.
- No changes to the workspace `edition`, `rust-version`, or `resolver`.

**Rationale for merging:** a buildable, tested, non-default crate that
demonstrates a working Vulkan compute path on one well-documented device is
strictly additive. It costs non-Vulkan contributors nothing (excluded from
`default-members`); it costs the maintainer one additional directory in `crates/`.
The alternative — never merging the spike — means the portability work
disappears when the branch is deleted.

**Reviewer question to pre-answer:** "Why a sidecar and not a Candle backend?"
Answer in design-decisions §1: the Candle trait surface is a month of work
before any numerical result, and the sidecar delivers the same measurement
value at a fraction of the cost. The plan is to convert to a backend **after**
the Q4_K kernel and 30-minute stability gate pass, not before.

**Failure mode to accept up front:** the maintainer may prefer that the spike
live outside the main repo (a separate `vulkex` crate on crates.io, linked from
the README). That is a fine outcome; the documentation transfers unchanged.

---

## PR 2 — Wire `vulkex` behind a `vulkan` feature in `mistralrs-core`

**Target:** `EricLBuehler/mistral.rs`, after PR 1 merges.

**Preconditions (do not open PR 2 without all four):**
1. `bench_shapes` shows Vulkan GEMM at ≥ 0.5× the `gemm`-crate CPU reference on
   Qwen3-1.7B prefill shapes (`512×6144×2048`, `512×2048×6144`,
   `512×2048×2048`).
2. `gemm_numerics` passes with `max_abs < 1e-4`, `max_rel < 1e-3` on the same
   shapes.
3. `Buffer` frees its `Allocation` on drop; `Context::drop` calls
   `vkDestroyDevice`. Without these, the 30-minute test that PR 2 must include
   will fail on memory growth.
4. A Q4_K GEMV kernel exists and matches a `gemm`-crate reference on
   dequantized weights to FP32 tolerance.

The last precondition is the one that is hardest to justify skipping. Every
model in `~/ai-models/gguf/` is Q4_K. An FP32-only Vulkan path in
`mistralrs-core` cannot run them without the load-time dequantization that
inflates a 5 GB GGUF to a 36 GB FP32 tensor. That path does not fit in the
10.4 GiB iGPU budget.

**Contents of PR 2:**
- `mistralrs-core/Cargo.toml`:
  ```toml
  [dependencies]
  vulkex = { workspace = true, optional = true }
  [features]
  vulkan = ["dep:vulkex"]
  ```
- `mistralrs-core/src/device/mod.rs` (or wherever `Device` is defined at the
  target revision): a `Device::Vulkan(VulkanDevice)` variant, populated only
  when the `vulkan` feature is on. `Device::cuda_if_available()` and
  `Device::metal_if_available()` gain siblings `Device::vulkan_if_available()`.
- CLI (`mistralrs-cli`): a `--device` flag with `cpu | cuda | metal | vulkan`
  values, default `cpu`. Env override `MISTRALRS_DEVICE=vulkan`. Both names
  are proposed; they should match whatever `--device` spelling exists at the
  target revision. If the CLI has no `--device` today, this PR adds it for
  all four backends at once, not just Vulkan.
- `crates/vulkex/tests/` gains a `model_smoke.rs` — loads a tiny transformer
  fixture (not a real model) and runs one full block. Correctness against CPU
  FP32.

**What is not in PR 2:** model weights, quantization dispatch beyond Q4_K,
KV cache in Vulkan buffers. PR 2 is "the tensor ops dispatch through Vulkan";
PR 3 is "the model fits in Vulkan buffers with KV in Vulkan buffers".

---

## PR 3 — UMA-optimized model load path

**Target:** `EricLBuehler/mistral.rs`, after PR 2 merges.

**Preconditions:** 30-minute stability gate passes on PR 2; `docs/vulkan-device-report.md`
contains a "CPU baseline" section and a "Vulkan baseline" section measured on
the same machine, same model, same prompt.

**Contents:**
- Load path that maps GGUF into `vulkex` buffers without a full CPU FP32
  intermediate. This is where the actual 10.4 GiB budget is spent.
- KV cache residency decision, measured not assumed: KV in host memory with
  per-decode copies, or KV in device memory with `vkCmdCopyBuffer` between
  layers. The device report records that all host-visible memory is coherent;
  this makes host-KV cheaper than on a discrete GPU, but does not make it free.
- Fallback report surfaced in the CLI: `--vulkan-report` prints the
  `FallbackReport` at session end.

**What is not in PR 3:** speculative decoding, vision, FlashAttention-class
kernels. Those are out of scope until a model runs end-to-end.

---

## Parallel Candle proposal (not a PR, a discussion)

The `EricLBuehler/candle` repository has an open issue (#1810, historically)
about a Vulkan backend. `vulkex` is not a Candle backend, but the design
decisions here apply directly. The proposal, if opened, would be:

> A Vulkan backend is cheaper to build as a compute executor behind a small
> `Backend` interface than as a full `BackendStorage` implementation. The
> `mistral.rs` spike (`crates/vulkex`) demonstrates a two-kernel executor
> (`gemv`, `gemm`) that covers the decode- and prefill-critical paths on an
> Intel UHD CML GT2 with 4096×4096 FP32 numerical validation. The full
> `BackendStorage` surface can follow incrementally once the executor is
> proven on multiple UMA devices.

Do not open this discussion until PR 2 has merged and the stability gate
passes. A proposal that has not survived a month of real use is noise.

---

## What to send in the PR description (template)

The maintainer has seen Vulkan PRs before. Lead with the measurement, not the
motivation:

```
## Summary

Adds `crates/vulkex`, a Vulkan compute executor sidecar, excluded from
`default-members`. `gemv` and `gemm` are implemented in FP32 with
shared-memory reduction (no subgroup ops). On Intel UHD CML GT2 / Mesa
26.2.5, GEMV on 4096×4096 passes at max_abs 2.0e-6 / max_rel 7.7e-5
against an f64 CPU reference. GEMM timing and correctness are in
`tests/bench_shapes.rs` and `tests/gemm_numerics.rs`.

## What this does not do

No model integration, no quantization kernel, no KV cache in Vulkan
buffers. This is the measurement platform for those decisions.

## Why a sidecar and not a Candle backend

See `docs/vulkan-design-decisions.md` §1.

## What non-Vulkan contributors pay

Nothing. `cargo build` at the repo root does not compile shaderc.
```

Keep it that short. Any PR description that cannot fit the measurement into
the first paragraph is not ready to be reviewed.
