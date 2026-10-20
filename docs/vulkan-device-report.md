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
- coherent flag is populated from mem_props.memory_types[alloc.memory_type_index()]
  .property_flags.contains(HOST_COHERENT) at allocation time.

## Spike gate result (vulkex gemv_numerics, first pass)

- Shape: 4096 x 4096 FP32 GEMV
- max_abs = 2.027e-6, max_rel = 7.708e-5 against an f64 CPU reference
- validation=true (VK_LAYER_KHRONOS_validation active), zero VUID errors
- submissions=1 fence_waits=1 flushes=0 invalidates=0
- gemv: 1 on Vulkan, 0 CPU fallbacks

## Known deferred issues

- Buffer does not call allocator.free(); acceptable for one-shot tests, must
  be fixed before the 30-minute stability run. Standard pattern: Buffer owns
  its Allocation and holds Arc<Mutex<Allocator>>; Drop calls free().
- Context::drop does not call device.destroy_device(None). Acceptable for
  short-lived processes; will leak the VkDevice across repeated Context
  creation. Fix in the same refactor as the Buffer ownership change.

## CPU baseline (target model, date fill-in)

- Command: <exact invocation>
- Model: <hf repo id>
- Device: CPU, dtype <fp32|fp16>
- Prompt: 512 tokens, greedy
- Output: 128 tokens
- Peak RSS: <from /usr/bin/time -v>
- Prefill: <tok/s>
- Decode: <tok/s>

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

**3. Host and device share bandwidth; a "GPU kernel" is not "free compute".**
Every byte the shader reads from `Device` was written by the CPU, and every
byte it writes back competes with the OS's memory traffic. On a discrete GPU
the arithmetic hides behind the memory latency; on this iGPU, the arithmetic
and the memory traffic are the same silicon budget.

*Consequence:* the target for `bench_shapes` is not "Vulkan is much faster
than CPU." It is "Vulkan is at least competitive with CPU, and the
performance scaling on larger shapes tracks the same curve as CPU rather
than falling behind." If the Vulkan GEMM starts fast on small shapes and
degrades on larger ones, the bottleneck is dispatch overhead or memory
bandwidth, not arithmetic — and the optimization path is batching dispatches,
not tuning the tile size.

**Numerical summary** (fill in from `bench_shapes` and `mistralrs bench`):

| Shape | CPU (`gemm` crate) | Vulkan | Ratio |
|---|---|---|---|
| 512×6144×2048 | `<placeholder>` | `<placeholder>` | `<placeholder>` |
| 512×2048×6144 | `<placeholder>` | `<placeholder>` | `<placeholder>` |
| 512×2048×2048 | `<placeholder>` | `<placeholder>` | `<placeholder>` |

| Session | Prefill tok/s | Decode tok/s | Peak RSS |
|---|---|---|---|
| CPU baseline (Qwen3-1.7B, 512 / 128) | `<placeholder>` | `<placeholder>` | `<placeholder>` |
| Vulkan (when integrated) | `<placeholder>` | `<placeholder>` | `<placeholder>` |
