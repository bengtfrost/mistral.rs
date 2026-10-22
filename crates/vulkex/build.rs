use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let shaders = [("gemv", "shaders/gemv.comp"), ("gemm", "shaders/gemm.comp")];

    let compiler = shaderc::Compiler::new().expect("shaderc compiler");
    let mut opts = shaderc::CompileOptions::new().expect("shaderc opts");
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    for (name, path) in shaders {
        println!("cargo:rerun-if-changed={path}");
        let src = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("read {path}"));
        let artifact = compiler
            .compile_into_spirv(
                &src,
                shaderc::ShaderKind::Compute,
                path,
                "main",
                Some(&opts),
            )
            .unwrap_or_else(|e| panic!("compile {path}: {e}"));

        let mut out = String::new();
        writeln!(out, "pub const {}_SPV: &[u32] = &[", name.to_uppercase()).unwrap();
        for w in artifact.as_binary() {
            writeln!(out, "    0x{:08x},", w).unwrap();
        }
        out.push_str("];\n");
        std::fs::write(out_dir.join(format!("{name}_spv.rs")), out).unwrap();
    }
}
