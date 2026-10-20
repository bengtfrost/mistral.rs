use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shaders/gemv.comp");

    let src = std::fs::read_to_string("shaders/gemv.comp").expect("read shader");
    let compiler = shaderc::Compiler::new().expect("shaderc compiler");
    let mut opts = shaderc::CompileOptions::new().expect("shaderc opts");
    opts.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);

    let artifact = compiler
        .compile_into_spirv(&src, shaderc::ShaderKind::Compute, "gemv.comp", "main", Some(&opts))
        .expect("compile shader");

    let mut out = String::from("pub const GEMV_SPV: &[u32] = &[\n");
    for w in artifact.as_binary() {
        writeln!(out, "    0x{:08x},", w).unwrap();
    }
    out.push_str("];\n");

    let path = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("gemv_spv.rs");
    std::fs::write(path, out).unwrap();
}
