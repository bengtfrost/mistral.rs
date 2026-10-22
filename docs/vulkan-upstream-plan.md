# Vulkan Backend — Upstream Plan

Three PRs, in order. **Read §0 first:** the measurements have changed
the scope of PRs 2 and 3.

---

## 0. Status after measurements

**Measured (2026-10-22):** on the Intel UHD CML GT2 / i5-10210U, this FP32
Vulkan executor is **~2× slower than the CPU backend** on both prefill
(0.48–0.52× GEMM on 512×6144×2048) and predicted decode (0.46×, from
per-token dispatch accounting). See `docs/vulkan-device-report.md` for
the full measurement.

**Consequence for the plan:**

- **PR 1** (spike crate) remains valid and worth submitting — it is
  additive, honest about its results, and useful as a reference for
  stronger UMA hardware.
- **PR 2** (mistralrs-core integration) is **re-scoped**: it should not
  be opened to make this laptop faster (it would not). It is opened to
  enable the portability target — running the same kernels on Intel Arc,
  RDNA3 APU, or a discrete GPU where compute-per-byte is higher. The
  preconditions change accordingly.
- **PR 3** (UMA-optimized load path) is deferred until a stronger UMA
  device is available for measurement.

Do not open PR 2 without a second device to benchmark against. PR 2's
value is portability; a PR that says "integration for hardware that
doesn't benefit" will not merge.

---

## PR 1 — Add `crates/vulkex` as an opt-in workspace member

**Target:** `EricLBuehler/mistral.rs` at revision `465191d0` and current master.

**Contents:**
- `crates/vulkex/` (the spike, as it stands)
- `Cargo.toml`: add `"crates/vulkex"` to `members`, mirror-minus-vulkex to
  `default-members`
- `docs/vulkan-device-report.md` (D1, with measurements)
- `docs/vulkan-design-decisions.md` (D6a, updated)
- `docs/vulkan-upstream-plan.md` (D6b, this file)

**What is not in this PR:**
- No changes to `mistralrs-core`, `mistralrs-cli`, or any other crate.
- No new features on any existing crate.
- No changes to the workspace `edition`, `rust-version`, or `resolver`.

**Rationale for merging:** a buildable, tested, non-default crate with
documented measurements is strictly additive. It costs non-Vulkan
contributors nothing (excluded from `default-members`); it costs the
maintainer one additional directory in `crates/`. The measurement it
produces is negative on this specific hardware — Vulkan is slower — and
that is a legitimate result for a spike. The alternative is losing the
portability work when the branch is deleted, and losing the negative
result for anyone considering the same path in the future.

**Reviewer question to pre-answer:** "Why a sidecar and not a Candle
backend?" Answer in `docs/vulkan-design-decisions.md` §1: the Candle
trait surface is a month of work before any numerical result, and the
sidecar delivered the same measurement in a week. The measured 0.5× on
prefill confirms the sidecar approach was correct — a full backend would
not have made this device faster either.

**Reviewer question to pre-answer:** "Why merge a slower implementation?"
Answer in this file §0: the crate documents what was measured and why the
Vulkan path does not win on this class of hardware. It is a reference for
portability to stronger UMA devices (Arc, RDNA3 APU), not a claim that
Vulkan makes this laptop faster.

**Failure mode to accept up front:** the maintainer may prefer that the
spike live outside the main repo (a separate `vulkex` crate on crates.io,
linked from the README). That is a fine outcome; the documentation
transfers unchanged.

---

## PR 2 — Wire `vulkex` behind a `vulkan` feature in `mistralrs-core`

**Target:** `EricLBuehler/mistral.rs`, after PR 1 merges **and** a stronger
UMA device is available for measurement.

**Preconditions (do not open PR 2 without all five):**
1. **A second device** (Intel Arc, AMD RDNA3 APU, or discrete GPU) on which
   `bench_shapes` shows Vulkan ≥ 1.0× CPU on prefill, or a documented
   reason to expect it (e.g. compute-to-bandwidth ratio of the target).
2. `gemm_numerics` passes with `max_abs < 5e-4`, `rel_inf < 1e-5` on the
   Qwen3-1.7B prefill shapes.
3. `Context::drop` calls `vkDestroyDevice`; `Buffer` implements `Drop` or
   is managed by a scoped guard. Without these, the 30-minute stability
   test will fail on memory growth.
4. A Q4_K GEMV kernel exists and matches a `gemm`-crate reference on
   dequantized weights to FP32 tolerance. Every model in `~/ai-models/gguf/`
   is Q4_K; an FP32-only Vulkan path in `mistralrs-core` cannot run them
   without load-time dequantization that inflates a 5 GB GGUF to a 36 GB
   FP32 tensor, which does not fit in the 10.4 GiB iGPU budget.
5. The PR description leads with the measurement on the second device —
   not with "adds a Vulkan backend", which would imply local speedup.

**Contents of PR 2:**
- `mistralrs-core/Cargo.toml`:
  ```toml
  [dependencies]
  vulkex = { workspace = true, optional = true }
  [features]
  vulkan = ["dep:vulkex"]