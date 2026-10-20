use anyhow::Result;
use ash::vk;

use crate::buffer::Buffer;
use crate::context::Context;
use crate::fallback::FallbackReport;
use crate::pipeline::GemvPipeline;

pub struct GemvSession {
    pub ctx: Context,
    pub pipeline: GemvPipeline,
    pub report: FallbackReport,
    pub desc_set: vk::DescriptorSet,
}

impl GemvSession {
    pub fn new(target_pci: Option<u32>) -> Result<Self> {
        let ctx = Context::new(target_pci)?;
        let pipeline = unsafe { GemvPipeline::new(&ctx)? };
        let layouts = [pipeline.desc_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pipeline.desc_pool)
            .set_layouts(&layouts);
        let sets = unsafe { ctx.device.allocate_descriptor_sets(&alloc_info)? };
        Ok(Self {
            ctx,
            pipeline,
            report: FallbackReport::default(),
            desc_set: sets[0],
        })
    }

    /// rows x cols FP32 GEMV. All buffers must already exist and be host-visible.
    pub fn gemv(&mut self, w: &Buffer, x: &Buffer, y: &Buffer, rows: u32, cols: u32) {
        unsafe { self.gemv_inner(w, x, y, rows, cols) }
    }

    unsafe fn gemv_inner(&mut self, w: &Buffer, x: &Buffer, y: &Buffer, rows: u32, cols: u32) {
        unsafe {
            self.ctx.flush(w, &mut self.report);
            self.ctx.flush(x, &mut self.report);
        }

        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(w.raw)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(x.raw)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(y.raw)
                .offset(0)
                .range(vk::WHOLE_SIZE),
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[2])),
        ];
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };

        unsafe { self.ctx.device.reset_fences(&[self.ctx.fence]) }.ok();

        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe { self.ctx.device.allocate_command_buffers(&alloc).unwrap() };
        let cmd = cmds[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.ctx.device.begin_command_buffer(cmd, &begin) }.ok();
        unsafe {
            self.ctx.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );
            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.desc_set],
                &[],
            );
            let push = [rows, cols];
            self.ctx.device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(push.as_ptr() as *const u8, 8),
            );
            self.ctx.device.cmd_dispatch(cmd, rows, 1, 1);
            self.ctx.device.end_command_buffer(cmd).ok();
        }

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            self.ctx
                .device
                .queue_submit(self.ctx.queue, &[submit], self.ctx.fence)
        }
        .ok();
        self.report.submissions += 1;

        unsafe {
            self.ctx
                .device
                .wait_for_fences(&[self.ctx.fence], true, u64::MAX)
        }
        .ok();
        self.report.fence_waits += 1;

        unsafe {
            self.ctx
                .device
                .free_command_buffers(self.ctx.cmd_pool, &[cmd])
        };

        unsafe {
            self.ctx.invalidate(y, &mut self.report);
        }
        self.report.record_vulkan("gemv");
    }
}
