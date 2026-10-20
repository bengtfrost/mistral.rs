use anyhow::{Result, anyhow};
use ash::vk;

include!(concat!(env!("OUT_DIR"), "/gemv_spv.rs"));

pub struct GemvPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub shader_module: vk::ShaderModule,
}

impl GemvPipeline {
    pub unsafe fn new(ctx: &crate::Context) -> Result<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { ctx.device.create_descriptor_set_layout(&dsl_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8);
        let set_layouts = [desc_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let layout = unsafe { ctx.device.create_pipeline_layout(&pl_info, None)? };

        let sm_info = vk::ShaderModuleCreateInfo::default().code(GEMV_SPV);
        let shader_module = unsafe { ctx.device.create_shader_module(&sm_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let cp_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipelines = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[cp_info], None)
        }
        .map_err(|(_, e)| anyhow!("create_compute_pipelines: {e:?}"))?;
        let pipeline = pipelines[0];

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let desc_pool = unsafe { ctx.device.create_descriptor_pool(&pool_info, None)? };

        Ok(GemvPipeline {
            layout,
            pipeline,
            desc_set_layout,
            desc_pool,
            shader_module,
        })
    }
}
