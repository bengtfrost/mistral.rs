use anyhow::{Result, anyhow};
use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::{
    AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};

use crate::buffer::Buffer;

pub struct Context {
    pub _entry: Entry,
    pub allocator: Allocator, // must drop before device and instance
    pub device: Device,
    pub instance: Instance,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub cmd_pool: vk::CommandPool,
    pub fence: vk::Fence,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    /// True iff every host-visible memory type on this device reports
    /// HOST_COHERENT. Diagnostic only; `Buffer::coherent` is per-allocation.
    pub all_host_visible_coherent: bool,
    pub validation_enabled: bool,
}

impl Context {
    pub fn new(target_pci: Option<u32>) -> Result<Self> {
        unsafe { Self::new_inner(target_pci) }
    }

    unsafe fn new_inner(target_pci: Option<u32>) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let validation_name = c"VK_LAYER_KHRONOS_validation";
        let have_validation = available_layers.iter().any(|l| {
            let name = unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) };
            name == validation_name
        });

        // If the caller asked for validation via VK_INSTANCE_LAYERS and it is not
        // installed, fail loudly rather than silently running unvalidated.
        let env_requests_validation = std::env::var("VK_INSTANCE_LAYERS")
            .map(|v| {
                v.split(',')
                    .any(|s| s.trim() == "VK_LAYER_KHRONOS_validation")
            })
            .unwrap_or(false);
        if env_requests_validation && !have_validation {
            anyhow::bail!(
                "VK_INSTANCE_LAYERS requests VK_LAYER_KHRONOS_validation but the layer is not installed; \
                 install Vulkan-ValidationLayers or unset VK_INSTANCE_LAYERS"
            );
        }

        let validation_enabled = have_validation;
        let layer_ptrs: Vec<*const i8> = if validation_enabled {
            vec![validation_name.as_ptr()]
        } else {
            vec![]
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vulkex")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"vulkex")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let physicals = unsafe { instance.enumerate_physical_devices()? };
        if physicals.is_empty() {
            return Err(anyhow!("no Vulkan physical devices"));
        }
        let physical = physicals
            .iter()
            .copied()
            .find(|pd| {
                let props = unsafe { instance.get_physical_device_properties(*pd) };
                match target_pci {
                    Some(id) => props.device_id == id,
                    None => true,
                }
            })
            .ok_or_else(|| anyhow!("no matching Vulkan device (target_pci={target_pci:?})"))?;

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let all_host_visible_coherent = (0..mem_props.memory_type_count).all(|i| {
            let t = mem_props.memory_types[i as usize];
            let hv = t
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let coh = t
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            !hv || coh
        });

        let qfams = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = qfams
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| anyhow!("no compute queue family"))?;

        let priorities = [1.0f32];
        let q_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&q_info);
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None)? };

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Context {
            _entry: entry,
            instance,
            device,
            physical,
            queue,
            queue_family,
            allocator,
            cmd_pool,
            fence,
            mem_props,
            all_host_visible_coherent,
            validation_enabled,
        })
    }

    /// Allocate a host-visible storage buffer.
    ///
    /// `gpu-allocator` selects the memory type from `MemoryLocation::CpuToGpu`.
    /// On a UMA device this typically resolves to a `DEVICE_LOCAL | HOST_VISIBLE`
    /// type; on a discrete GPU it resolves to a staging type. The choice is the
    /// allocator's, not this function's.
    ///
    /// The returned `Buffer::coherent` flag is read from the chosen memory type's
    /// `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`. [`Context::flush`] and
    /// [`Context::invalidate`] consult that flag and short-circuit on coherent
    /// memory; on non-coherent memory they issue the corresponding
    /// `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges` calls.
    /// Callers do not need to know which case they are in — calling flush after
    /// host writes and invalidate before host reads is correct in both.
    ///
    /// # Safety
    ///
    /// - The caller must reclaim the returned buffer with [`Context::free_buffer`].
    ///   Dropping the `Buffer` without freeing leaks the underlying `Allocation`.
    /// - The buffer must not be accessed after `free_buffer` returns.
    ///
    /// [`Context::flush`]: Context::flush
    /// [`Context::invalidate`]: Context::invalidate
    /// [`Context::free_buffer`]: Context::free_buffer
    pub unsafe fn alloc_mapped(&mut self, size: u64) -> Result<Buffer> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = unsafe { self.device.create_buffer(&info, None)? };
        let reqs = unsafe { self.device.get_buffer_memory_requirements(raw) };

        let alloc = self.allocator.allocate(&AllocationCreateDesc {
            name: "vulkex.buffer",
            requirements: reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device
                .bind_buffer_memory(raw, alloc.memory(), alloc.offset())?
        };

        // Coherence is per-allocation: read the flags of the memory type that
        // gpu-allocator actually selected, not a device-wide approximation.
        // On mixed-coherence devices this lets flush/invalidate run only when
        // the specific allocation needs them.
        let mem_type_idx = alloc.memory_type_index() as usize;
        let coherent = self.mem_props.memory_types[mem_type_idx]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);

        let mapped = if alloc.mapped_ptr().is_some() {
            alloc.mapped_ptr().unwrap().as_ptr() as *mut u8
        } else {
            std::ptr::null_mut()
        };

        Ok(Buffer {
            raw,
            alloc,
            size,
            coherent,
            mapped,
        })
    }

    /// Explicit free. `Buffer` must not be used after this call.
    pub unsafe fn free_buffer(&mut self, buf: Buffer) {
        unsafe { self.device.destroy_buffer(buf.raw, None) };
        if let Err(e) = self.allocator.free(buf.alloc) {
            log::warn!("allocator.free: {e:?}");
        }
    }

    pub unsafe fn flush(&mut self, buf: &Buffer, report: &mut crate::fallback::FallbackReport) {
        if buf.coherent || !buf.is_mapped() {
            return;
        }
        let r = vk::MappedMemoryRange::default()
            .memory(buf.memory())
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { self.device.flush_mapped_memory_ranges(&[r]) }.ok();
        report.flushes += 1;
    }

    pub unsafe fn invalidate(
        &mut self,
        buf: &Buffer,
        report: &mut crate::fallback::FallbackReport,
    ) {
        if buf.coherent || !buf.is_mapped() {
            return;
        }
        let r = vk::MappedMemoryRange::default()
            .memory(buf.memory())
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { self.device.invalidate_mapped_memory_ranges(&[r]) }.ok();
        report.invalidates += 1;
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.cmd_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.cmd_pool, None);
            }
            // `vkDestroyDevice` is not called here on purpose. The allocator
            // (which drops after this method returns, before `device`) still
            // needs a live VkDevice to free its pools. ash's `Device::drop`
            // then frees function pointers but does not call vkDestroyDevice,
            // so the device handle leaks per Context creation. Fixing this
            // requires wrapping `allocator` in `ManuallyDrop` and dropping it
            // explicitly before calling destroy_device — deferred to D1
            // ("Context::drop does not call device.destroy_device(None)").
        }
    }
}
