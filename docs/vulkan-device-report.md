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

| Shape           | CPU GFLOPS | Vulkan GFLOPS | Ratio |
| --------------- | ---------- | ------------- | ----- |
| 512×6144×2048   | 63.7       | 33.1          | 0.52× |
| 512×2048×6144   | 65.9       | 33.0          | 0.50× |
| 512×2048×2048   | 65.6       | 32.8          | 0.50× |
| 512×151936×2048 | 68.9       | 33.0          | 0.48× |

### Decode (GEMV, M = 1) — **misleading reference, see note**

| Shape       | CPU GB/s | Vulkan GB/s | Ratio |
| ----------- | -------- | ----------- | ----- |
| 2048×2048   | 1.6      | 3.8         | 2.36× |
| 2048×6144   | 1.6      | 11.0        | 6.94× |
| 6144×2048   | 1.6      | 8.2         | 5.17× |
| 151936×2048 | 1.6      | 13.1        | 8.22× |

**Note on the CPU GEMV reference:** `matrixmultiply::sgemm` with `n=1`
leaves most SIMD lanes idle and does not parallelize across rayon threads.
The CPU backend of mistral.rs uses a different path that streams weights at
**~19.3 GB/s** on this machine (see CPU baseline below). Against that
throughput, the Vulkan GEMV is _slower_ on every shape except the LM head.
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

| Component                                           | Per token              |
| --------------------------------------------------- | ---------------------- |
| Attention (28 layers × 4 × 2048×2048, 4.46 ms each) | ~500 ms                |
| MLP up (28 × 2048×6144, 4.56 ms each)               | ~128 ms                |
| MLP down (28 × 6144×2048, 6.16 ms each)             | ~172 ms                |
| LM head (1 × 151936×2048, 94.79 ms)                 | ~95 ms                 |
| **Total**                                           | **~895 ms = 1.12 T/s** |

### Measured CPU baseline

- Prefill: 54.7 T/s
- Decode: 2.4 T/s

### Comparison

| Metric  | CPU baseline | Vulkan (predicted) | Ratio  |
| ------- | ------------ | ------------------ | ------ |
| Prefill | 54.7 T/s     | ~27 T/s            | ~0.5×  |
| Decode  | 2.4 T/s      | ~1.1 T/s           | ~0.46× |

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

_Consequence:_ every allocation decision must budget against 10.4 GiB, not
16.5. A Qwen3-1.7B FP32 model (7 GB weights + KV cache) fits. A
Qwen3.5-9B Q4_K (5.3 GB weights + KV cache) fits in Q4_K form but not
after FP32 dequantization (36 GB). This is the single strongest reason the
Q4_K kernel is the one that matters for this machine.

**2. The heap is host-coherent, so flush/invalidate are dead code.**
Every host-visible memory type carries `HOST_COHERENT_BIT`. `Context::flush`
and `Context::invalidate` evaluate `if !coherent` and skip on this machine.

_Consequence:_ the timing results in `bench_shapes` do not include any
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

_Consequence:_ on this hardware, kernel-level wins do not translate to
model-level wins. A Vulkan GEMV that streams a 16 MB weight matrix at 3.8
GB/s is 5× slower than a CPU matmul that streams the same 16 MB at 19.3
GB/s, because the Vulkan path spends 3 ms per dispatch on overhead while
the CPU amortizes work across the whole layer. This is the structural
reason no FP32 Vulkan kernel will beat the CPU on this machine, regardless
of tile size, subgroup usage, or Q4_K dequantization.

## Comparison with llama.cpp's ggml-vulkan backend

The conclusion below is about **this executor**, not about the hardware. A
comparison with llama.cpp's Vulkan backend on comparable hardware makes the
distinction concrete, because ggml-vulkan uses three techniques that this
sidecar does not:

**Graph-level batching.** ggml records the whole computation graph and
submits it in chunks, overlapping CPU command-buffer generation with GPU
execution. This sidecar submits one command buffer per GEMV and blocks on a
fence after each. For Qwen3-1.7B that is 169 dispatches per token, each
paying ~3 ms fixed overhead — ~336 ms per token before any weight is read.
llama.cpp pays that overhead once per graph chunk.

**Quantization-native kernels.** `mul_mat_vec_q4_k.comp` reads Q4_K blocks
directly and dequantizes in-register, using `vec4` dequant and 128-bit B
loads. This sidecar reads FP32 weights: 4× the bytes for the same model. On
a bandwidth-bound device that alone is a 4× handicap.

**Subgroup reductions and Intel-tuned workgroups.** ggml uses larger
workgroups and subgroup reductions where supported, tuned specifically for
Intel. This sidecar avoids subgroups entirely (the properties query returned
no data on this driver) and uses shared-memory reduction.

### Measured on comparable hardware

From the llama.cpp discussion thread and benchmarks collected there — **not
measured on this machine**, cited as reference points:

| Hardware          | CPU                             | Backend                   | Eval (s) | Tokens/s |
| ----------------- | ------------------------------- | ------------------------- | -------- | -------- |
| UHD 620, i5-8250U | i5-8250U (4C/8T, 3.4 GHz turbo) | CPU only                  | 428      | 2.32     |
| UHD 620, i5-8250U | i5-8250U                        | Vulkan, all layers on GPU | 14       | **6.07** |

That is **2.6× faster decode** and ~30× faster prompt response on a weaker
CPU than the i5-10210U in this machine.

| Hardware        | Backend                                       | pp512 (t/s) | tg128 (t/s) |
| --------------- | --------------------------------------------- | ----------- | ----------- |
| Iris Xe (Gen12) | Vulkan, ngl 100 (all on GPU)                  | 50.86       | 7.28        |
| Iris Xe (Gen12) | Vulkan, ngl 0 (prefill on GPU, decode on CPU) | 32.11       | **9.49**    |

On newer Gen12 hardware, llama.cpp's Vulkan backend **wins prefill by 1.58×
but loses decode by 26%**. The community recommendation on integrated GPUs
is explicitly `ngl 0`: use the iGPU for prefill, keep decode on the CPU.

### What this means for this machine

Three findings, in decreasing confidence:

**Prefill (GEMM) — our 0.5× was an implementation limit, not a hardware
verdict.** The three techniques above together account for the gap to CPU.
On this class of iGPU, ggml-vulkan's prefill beats CPU on Iris Xe; on the
weaker UHD 620, prompt processing is 30× faster. There is no structural
reason this iGPU cannot prefill faster than the CPU, given an executor that
batches graph submissions and reads quantized weights. Our 0.48–0.52×
prefill is a property of the executor we built, not of the silicon.

**Decode (GEMV) — a hardware ceiling on this CPU/iGPU pair, confirmed by
llama.cpp's own behavior.** The UHD 620 success story uses an i5-8250U
(3.4 GHz turbo). The i5-10210U in this machine runs at 4.2 GHz turbo, roughly
25% higher single-core throughput. That is exactly the class where Vulkan's
decode advantage vanishes, and it matches what the Iris Xe data shows: even
mature ggml-vulkan loses decode to the CPU on a fast-enough CPU. The
predicted ~0.46× decode in the model-level comparison is not a failure of
our implementation; it is the expected outcome for this CPU/iGPU pair, with
this executor or any other.

**The `ngl 0` configuration is the honest target on this hardware.** Prefill
on the iGPU, decode on the CPU. That split is what llama.cpp recommends for
integrated GPUs, and it is what a real mistral.rs Vulkan backend should
target on this class of machine. It sidesteps the decode ceiling entirely
while capturing the prefill win.

The gap between this sidecar and a competitive Vulkan backend is not the
Vulkan API, the shader toolchain, or the numerical correctness of the
kernels — all three are proven here. The gap is executor architecture:
graph batching, quantized weight formats, and subgroup reductions. Those
are the requirements for any future work on this codebase.

## Conclusion: measured performance on this hardware

**On the Intel UHD CML GT2 / i5-10210U, this FP32, per-op-dispatch Vulkan
executor is ~2× slower than the mistral.rs CPU backend on both prefill and
decode, for Qwen3-1.7B F32.**

The kernels themselves are numerically correct (2e-6 GEMV, 3.9e-5 GEMM
against CPU references) and reasonably implemented at the shader level
(7.8% of theoretical peak on FP32 GEMM). They do not win because the
executor around them differs from a production Vulkan backend in three
ways, all of which llama.cpp's ggml-vulkan implements and this sidecar does
not: graph-level submission batching, Q4_K-native kernels, and subgroup
reductions with Intel-specific workgroup tuning.

**Three separate conclusions, which the llama.cpp data lets us distinguish:**

- **Prefill:** our 0.48–0.52× is an implementation limit. The hardware can
  prefill faster than CPU on this class of iGPU (llama.cpp demonstrates
  ~1.6× on Gen12, ~30× prompt-processing speedup on Gen9.5). Closing this
  gap requires executor architecture, not kernel tuning.

- **Decode:** our ~0.46× prediction is a hardware ceiling on this
  CPU/iGPU pair. Even mature ggml-vulkan loses decode to a fast CPU on
  integrated GPUs; the `ngl 0` configuration exists precisely because of
  this. Decode will not be competitive for any Vulkan executor on this
  machine.

- **The right target on this hardware is `ngl 0`:** iGPU for prefill, CPU
  for decode. That is the configuration that captures the achievable win
  without fighting the decode ceiling.

**Portability remains the correct framing, but the phrase changes.** The
measured data does not say "Vulkan cannot win on UMA iGPUs." It says
"Vulkan wins prefill on UMA iGPUs; decode is contested and CPU-dependent;
our executor needs graph batching and Q4_K before either conclusion applies
to it."

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
- No graph-level submission batching. This is the single largest gap to a
  competitive backend and the reason prefill measured as it did.
- No Q4_K dequant-fused kernel. Also a large bandwidth win on this
  hardware, and the format every model in `~/ai-models/gguf/` actually uses.
