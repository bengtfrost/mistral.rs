use anyhow::Result;
use ash::vk;

use crate::buffer::Buffer;
use crate::context::Context;
use crate::fallback::FallbackReport;
use crate::pipeline::{ComputePipeline, GEMM_SPV, GEMV_SPV};

pub struct Session {
    pub ctx: Context,
    pub gemv_pipe: ComputePipeline,
    pub gemm_pipe: ComputePipeline,
    pub report: FallbackReport,
}

impl Session {
    pub fn new(target_pci: Option<u32>) -> Result<Self> {
        let ctx = Context::new(target_pci)?;
        let gemv_pipe = unsafe { ComputePipeline::new(&ctx, GEMV_SPV, 8)? }; // rows, cols
        let gemm_pipe = unsafe { ComputePipeline::new(&ctx, GEMM_SPV, 12)? }; // M, N, K
        Ok(Self {
            ctx,
            gemv_pipe,
            gemm_pipe,
            report: FallbackReport::default(),
        })
    }

    pub fn gemv(&mut self, w: &Buffer, x: &Buffer, y: &Buffer, rows: u32, cols: u32) {
        unsafe { self.dispatch_gemv(w, x, y, rows, cols) }
    }

    pub fn gemm(&mut self, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, n: u32, k: u32) {
        unsafe { self.dispatch_gemm(a, b, c, m, n, k) }
    }

    unsafe fn dispatch_gemv(&mut self, w: &Buffer, x: &Buffer, y: &Buffer, rows: u32, cols: u32) {
        unsafe {
            self.ctx.flush(w, &mut self.report);
            self.ctx.flush(x, &mut self.report);
            bind_descriptors(&self.ctx.device, &self.gemv_pipe, w, x, y);
            submit_dispatch(
                &self.ctx.device,
                self.ctx.queue,
                self.ctx.cmd_pool,
                self.ctx.fence,
                &self.gemv_pipe,
                &[rows, cols],
                (rows, 1),
                &mut self.report,
            );
            self.ctx.invalidate(y, &mut self.report);
        }
        self.report.record_vulkan("gemv");
    }

    unsafe fn dispatch_gemm(&mut self, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, n: u32, k: u32) {
        unsafe {
            self.ctx.flush(a, &mut self.report);
            self.ctx.flush(b, &mut self.report);
            bind_descriptors(&self.ctx.device, &self.gemm_pipe, a, b, c);
            submit_dispatch(
                &self.ctx.device,
                self.ctx.queue,
                self.ctx.cmd_pool,
                self.ctx.fence,
                &self.gemm_pipe,
                &[m, n, k],
                (n.div_ceil(16), m.div_ceil(64)),
                &mut self.report,
            );
            self.ctx.invalidate(c, &mut self.report);
        }
        self.report.record_vulkan("gemm");
    }
}

fn bind_descriptors(
    device: &ash::Device,
    pipe: &ComputePipeline,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
) {
    let infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(a.raw)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(b.raw)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(c.raw)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(pipe.desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(pipe.desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(pipe.desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[2])),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

unsafe fn submit_dispatch(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    fence: vk::Fence,
    pipe: &ComputePipeline,
    push: &[u32],
    workgroups: (u32, u32),
    report: &mut FallbackReport,
) {
    unsafe { device.reset_fences(&[fence]) }.ok();

    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc).unwrap()[0] };

    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin) }.ok();
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipe.layout,
            0,
            &[pipe.desc_set],
            &[],
        );
        let push_bytes = std::slice::from_raw_parts(push.as_ptr() as *const u8, push.len() * 4);
        device.cmd_push_constants(
            cmd,
            pipe.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.cmd_dispatch(cmd, workgroups.0, workgroups.1, 1);
        device.end_command_buffer(cmd).ok();
    }

    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe { device.queue_submit(queue, &[submit], fence) }.ok();
    report.submissions += 1;

    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.ok();
    report.fence_waits += 1;

    unsafe { device.free_command_buffers(cmd_pool, &[cmd]) };
}
