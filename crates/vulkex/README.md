# vulkex

Standalone Vulkan compute executor sidecar for mistral.rs. Not a Candle
`BackendStorage` implementation. Not wired into `mistralrs-core`. Not
reachable from the CLI.

## Status

- **`gemv` (FP32, shared-memory reduction)**: validated against an f64 CPU
  reference on 4096×4096 at `max_abs = 2.0e-6`, `max_rel = 7.7e-5` with
  `VK_LAYER_KHRONOS_validation` active and zero VUID errors.
- **`gemm`, quantized kernels, model integration, KV cache in Vulkan
  buffers, 30-minute stability test, `--device vulkan` CLI flag**:
  not implemented.

## Why it exists

See `docs/vulkan-design-decisions.md` for the full rationale. Summary: the
Candle `BackendStorage` trait surface is a month of work before the first
numerical result; a sidecar executor delivers the same measurement in a week
and can be converted to a backend after the Q4_K kernel and stability gates
pass.

## What it runs on

Validated on:
- Intel UHD Graphics CML GT2 (Gen9.5, 0x9b41)
- Mesa 26.2.5 / ANV
- Void GNU/Linux, kernel 6.18

See `docs/vulkan-device-report.md` for the full device report.

## Build and test

```sh
cargo build -p vulkex

VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  cargo test -p vulkex --test gemv_numerics -- --nocapture
