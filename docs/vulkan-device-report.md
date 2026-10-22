# Vulkan Device Report — Intel UHD CML GT2 (0x9b41)

- Instance API: 1.4.363
- Device API: 1.4.354
- Driver: Mesa 26.2.5 / ANV (DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
- Device: Intel(R) UHD Graphics (CML GT2), Gen9.5, integrated UMA
- Validation layer: VK_LAYER_KHRONOS_validation (Vulkan-ValidationLayers 1.4.363.0_1)

## Features (confirmed on this machine)

- storageBuffer16BitAccess = true
- shaderFloat16 = true
- Subgroup properties: `vulkaninfo | grep -A8 VkPhysicalDeviceSubgroupProperties`
  returned no output under the standard grep; not used by any current kernel.

## Limits (confirmed on this machine)

- maxStorageBufferRange = 4294967295 (~4 GiB)
- maxPerStageDescriptorStorageBuffers = 201326592 (see interpretation below)
- maxComputeSharedMemorySize = 49152
- maxComputeWorkGroupInvocations = 1024
- minStorageBufferOffsetAlignment = 4

## Memory (confirmed)

- heap 0: DEVICE_LOCAL, budget 10.40 GiB
- Every host-visible memory type reports HOST_COHERENT_BIT.
- First implementation still routes through the conditional flush/invalidate path;
  on this device the condition evaluates false and the calls are skipped.

## Descriptor limits (interpretation)

- maxPerStageDescriptorStorageBuffers reports 201326592 (192 MiB of descriptors).
  This is ANV's reporting convention, not a usable pool budget.
  Practical binding budget: <= 8 storage buffers per set; chunk across sets if needed.
  Descriptor pool failures are treated as fallback triggers, not assertion failures.

## Coherence-related code path

- flush(): if !coherent { device.flush_mapped_memory_ranges(...) }
- invalidate(): if !coherent { device.invalidate_mapped_memory_ranges(...) }
- The `coherent` flag is set from `Context::all_host_visible_coherent`, a
  device-wide property populated at `Context::new` time. `gpu-allocator 0.28`
  does not expose the chosen memory type index on `Allocation`, so
  per-allocation lookup is not possible; the device-wide flag is a
  conservative approximation. On this machine every host-visible type is
  coherent, so `coherent == true` for every buffer and both calls short-circuit.

## Kernel-level numerical results

### GEMV (FP32, 4096×4096)

- max_abs = 2.027e-6, max_rel = 7.708e-5 against an f64 CPU reference
- validation=true (VK_LAYER_KHRONOS_validation active), zero VUID errors
- submissions=1 fence_waits=1 flushes=0 invalidates=0
- gemv: 1 on Vulkan, 0 CPU fallbacks

### GEMM (FP32, 512×6144×2048)

- max_abs = 3.910e-5 (forward-error bound 4.883e-4)
- rel_inf = 1.965e-6 (||err||_∞ / ||C_cpu||_∞)
- validation=true, zero VUID errors
- submissions=1 fence_waits=1 flushes=0 invalidates=0
- gemm: 1 on Vulkan, 0 CPU fallbacks

Both kernels pass their numerical gates. The GEMM uses a 64×64 tile with
4×4 per-thread register blocking (32 KB shared memory, 256 invocations).

## Benchmark: kernel-level (Vulkan vs CPU microkernel)

Captured 2026-10-22. `cargo test --release -p vulkex --test bench_shapes -- --ignored --nocapture`.
Host: i5-10210U, performance governor, AC power, no other applications.

CPU reference: `matrixmultiply::sgemm` — the microkernel Candle's CPU backend
wraps. **This is the reference for correctness, not throughput. See the note below.**

### Prefill (GEMM, M = 512)

| Shape | CPU GFLOPS | Vulkan GFLOPS | Ratio |
|---|---|---|---|
| 512×6144×2048 | 63.7 | 33.1 | 0.52× |
| 512×2048×6144 | 65.9 | 33.0 | 0.50× |
| 512×2048×2048 | 65.6 | 32.8 | 0.50× |
| 512×151936×2048 | 68.9 | 33.0 | 0.48× |

### Decode (GEMV, M = 1) — **misleading reference, see note**

| Shape | CPU GB/s | Vulkan GB/s | Ratio |
|---|---|---|---|
| 2048×2048 | 1.6 | 3.8 | 2.36× |
| 2048×6144 | 1.6 | 11.0 | 6.94× |
| 6144×2048 | 1.6 | 8.2 | 5.17× |
| 151936×2048 | 1.6 | 13.1 | 8.22× |

**Note on the CPU GEMV reference:** `matrixmultiply::sgemm` with `n=1`
leaves most SIMD lanes idle and does not parallelize across rayon threads.
The CPU backend of mistral.rs uses a different path that streams weights at
**~19.3 GB/s** on this machine (see CPU baseline below). Against that
throughput, the Vulkan GEMV is *slower* on every shape except the LM head.
The 2.4–8.2× ratios above measure the specific sgemm(n=1) microkernel, not
the mistral.rs CPU backend.

## CPU baseline — Qwen3-1.7B F32

Captured 2026-10-22 with `RUSTFLAGS="-C target-cpu=native"` on AC power,
`performance` governor, no other applications running. Model cached in
`~/.cache/huggingface/hub`.

- Command: `target/release/mistralrs bench --cpu -m Qwen/Qwen3-1.7B --prompt-len 512 --gen-len 128`
- Model: Qwen/Qwen3-1.7B, F32 (safetensors, 7750 MiB on-disk → 7.75 GB weights)
- Prompt: 512 tokens, 128 generation tokens, 3 iterations (variance < 1%)
- **Prefill: 54.7 ± 0.4 T/s (9357 ms TTFT for 512 tokens)**
- **Decode: 2.4 ± 0.0 T/s (421 ms TPOT)**
- Peak RSS: ~8.0 GB (7.75 GB weights + KV cache at 512-token context + working set)

**Effective decode bandwidth: 7750 MiB / 421 ms = 19.3 GB/s.**

This is 65–77% of the practical DDR4 ceiling on a dual-channel Comet Lake
laptop. The CPU backend is already close to memory-bound, and this is why
the Vulkan kernel cannot exceed it on this hardware.

## Model-level comparison — measured vs predicted

### Prediction (from kernel-level benchmark, weighted per-token)

| Component | Per token |
|---|---|
| Attention (28 layers × 4 × 2048×2048, 4.46 ms each) | ~500 ms |
| MLP up (28 × 2048×6144, 4.56 ms each) | ~128 ms |
| MLP down (28 × 6144×2048, 6.16 ms each) | ~172 ms |
| LM head (1 × 151936×2048, 94.79 ms) | ~95 ms |
| **Total** | **~895 ms = 1.12 T/s** |

### Measured CPU baseline

- Prefill: 54.7 T/s
- Decode: 2.4 T/s

### Comparison

| Metric | CPU baseline | Vulkan (predicted) | Ratio |
|---|---|---|---|
| Prefill | 54.7 T/s | ~27 T/s | ~0.5× |
| Decode | 2.4 T/s | ~1.1 T/s | ~0.46× |

**Predicted end-to-end: Vulkan is ~2× slower than CPU on this machine.**

The prediction is derived from kernel timings and per-token workload shape,
not from a full model run on Vulkan (the model integration is not built —
see "Known deferred issues"). It is the most honest number available without
building the integration.

The reason is dispatch overhead on the small shapes. Each 2048×2048 GEMV
costs 4.46 ms, of which ~1.4 ms is weight streaming and ~3 ms is fixed
submission overhead (command buffer allocation, descriptor update, fence
wait). At 112 such dispatches per token (28 layers × 4 projections),
overhead is ~336 ms of the ~500 ms attention total. The CPU does not pay
this per-shape overhead — it batches all attention projections for a layer
into one matmul call.

Even perfect command-buffer batching — 28 layers' worth of dispatches
submitted together — reduces the overhead to ~10 ms per token. The total
becomes ~564 ms = 1.77 T/s. Still slower than CPU's 2.4 T/s, because the
CPU is already at the shared bandwidth ceiling and the iGPU cannot get more.

## UMA findings (synthesis of the measurements above)

Three facts, each with the design consequence:

**1. The 16.5 GB pool is not 16.5 GB of GPU memory.**
`VkPhysicalDeviceMemoryBudgetPropertiesEXT` reports 10.4 GiB `DEVICE_LOCAL`
budget on heap 0. The remaining ~6 GiB is not "free for the GPU"; it is
already counted by the OS as reserved for system use.

*Consequence:* every allocation decision must budget against 10.4 GiB, not
16.5. A Qwen3-1.7B FP32 model (7 GB weights + KV cache) fits. A
Qwen3.5-9B Q4_K (5.3 GB weights + KV cache) fits in Q4_K form but not
after FP32 dequantization (36 GB). This is the single strongest reason the
Q4_K kernel is the one that matters for this machine.

**2. The heap is host-coherent, so flush/invalidate are dead code.**
Every host-visible memory type carries `HOST_COHERENT_BIT`. `Context::flush`
and `Context::invalidate` evaluate `if !coherent` and skip on this machine.

*Consequence:* the timing results in `bench_shapes` do not include any
coherence-traffic cost. On a device where the same test runs with a
non-coherent heap, expect the numbers to move — the direction depends on
whether the driver takes the coherence penalty at allocation or per-access.
Do not extrapolate this device's timings to discrete GPUs from the same
result set.

**3. Host and device share bandwidth, and the CPU is already close to its ceiling.**
Every byte the shader reads from `Device` was written by the CPU, and every
byte it writes back competes with the OS's memory traffic. The CPU backend
of mistral.rs streams weights at 19.3 GB/s, which is 65–77% of the
dual-channel DDR4 ceiling on this laptop. Any Vulkan path competes for the
same memory controller; it cannot exceed the CPU's throughput, only
approach it.

*Consequence:* on this hardware, kernel-level wins do not translate to
model-level wins. A Vulkan GEMV that streams a 16 MB weight matrix at 3.8
GB/s is 5× slower than a CPU matmul that streams the same 16 MB at 19.3
GB/s, because the Vulkan path spends 3 ms per dispatch on overhead while
the CPU amortizes work across the whole layer. This is the structural
reason no FP32 Vulkan kernel will beat the CPU on this machine, regardless
of tile size, subgroup usage, or Q4_K dequantization.

## Conclusion: measured performance on this hardware

**On the Intel UHD CML GT2 / i5-10210U, this FP32 Vulkan executor is ~2×
slower than the mistral.rs CPU backend on both prefill and decode, for
Qwen3-1.7B F32.**

The kernels themselves are numerically correct (2e-6 GEMV, 3.9e-5 GEMM
against CPU references) and well-implemented (29→33 GFLOPS from the 16×16
to 64×64 tile change). They do not win because:

1. The iGPU shares memory bandwidth with the CPU, and the CPU backend is
   already at 65–77% of that shared ceiling.
2. Decode dominates per-token time on this class of hardware, and decode
   for Qwen3-1.7B is 112 small (2048×2048) dispatches plus 56 medium
   dispatches plus one large LM head dispatch per token. The small ones
   are dispatch-overhead-bound, not bandwidth-bound.
3. Q4_K dequantization, which is the next planned kernel, reduces weight
   traffic by ~4× on **both** CPU and Vulkan. It makes both faster in
   absolute terms but does not change the relative picture: the CPU's
   matmul still amortizes layer work across one call, while the Vulkan
   executor pays per-dispatch cost.

**Portability remains the correct framing.** On Intel Arc, AMD RDNA3 APUs,
or a discrete GPU, compute scales with EU count and bandwidth is separate,
so both prefill and decode would improve — potentially by an order of
magnitude on prefill. On this laptop specifically, the CPU path is the
faster path, and the branch is a completed portability spike rather than
a local speedup.

## Known deferred issues

- `Context::drop` does not call `device.destroy_device(None)`. Acceptable
  for short-lived processes; will leak the VkDevice across repeated Context
  creation. Fix when integrating into a model runtime.
- `Buffer` ownership is fixed (it holds its `Allocation`, and
  `Context::free_buffer` reclaims it), but there is no `Drop for Buffer`,
  so forgetting to call `free_buffer` still leaks. This is intentional for
  the spike; the model runtime integration should use a scoped guard.
- No 30-minute stability run has been performed. The Buffer/Context fixes
  above are the preconditions for it.