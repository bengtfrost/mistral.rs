use matrixmultiply::sgemm;
use vulkex::Session;

fn pseudo_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push((s as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

fn target_pci() -> Option<u32> {
    std::env::var("VULKEX_PCI_ID").ok().and_then(|s| {
        let s = s.trim();
        u32::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
    })
}

#[test]
fn gemm_fp32_matches_cpu_within_tolerance() {
    env_logger::builder().is_test(true).try_init().ok();

    let m: usize = 512;
    let k: usize = 2048;
    let n: usize = 6144;

    let a = pseudo_f32(0xDEADBEEF, m * k);
    let b = pseudo_f32(0xCAFEBABE, k * n);

    let mut c_cpu = vec![0f32; m * n];
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c_cpu.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    let mut session = Session::new(target_pci()).expect("context");
    assert!(
        session.ctx.validation_enabled,
        "VK_INSTANCE_LAYERS requested VK_LAYER_KHRONOS_validation but the layer is not loaded"
    );

    let a_buf = unsafe { session.ctx.alloc_mapped((m * k * 4) as u64) }.unwrap();
    let b_buf = unsafe { session.ctx.alloc_mapped((k * n * 4) as u64) }.unwrap();
    let c_buf = unsafe { session.ctx.alloc_mapped((m * n * 4) as u64) }.unwrap();

    unsafe {
        a_buf.write_bytes(std::slice::from_raw_parts(
            a.as_ptr() as *const u8,
            a.len() * 4,
        ));
        b_buf.write_bytes(std::slice::from_raw_parts(
            b.as_ptr() as *const u8,
            b.len() * 4,
        ));
    }

    session.gemm(&a_buf, &b_buf, &c_buf, m as u32, n as u32, k as u32);

    let out_bytes = unsafe { c_buf.read_bytes(m * n * 4) };
    let gpu: Vec<f32> = (0..m * n)
        .map(|i| f32::from_le_bytes(out_bytes[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect();

    // Diagnostics: keep max_abs as a per-element metric, but do not assert on it.
    let mut max_abs = 0f32;
    for (g, r) in gpu.iter().zip(c_cpu.iter()) {
        max_abs = max_abs.max((g - r).abs());
    }

    // 1. Absolute-error bound. |A|,|B| ≤ 0.5 in the fixture, so the forward
    //    error for FP32 GEMM is bounded by K · ε_fp32 · |A|_∞ · |B|_∞.
    //    Allow 8× for accumulation-order differences vs. sgemm's microkernel.
    let abs_tol = (k as f32) * f32::EPSILON * 0.25 * 8.0;

    // 2. Whole-tensor relative error: ||err||_∞ / ||C_cpu||_∞.
    //    Unlike per-element relative error, this cannot be inflated by a
    //    single near-zero reference element (which is common: 3.14M samples
    //    from a zero-mean distribution include values around 1e-6).
    let cpu_inf = c_cpu.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let err_inf = gpu
        .iter()
        .zip(c_cpu.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0f32, f32::max);
    let rel_inf = err_inf / cpu_inf;

    println!(
        "gemm {m}x{n}x{k}: max_abs={max_abs:.3e} (bound {abs_tol:.3e}), \
         err_inf={err_inf:.3e}, cpu_inf={cpu_inf:.3e}, rel_inf={rel_inf:.3e}, \
         validation={}",
        session.ctx.validation_enabled
    );
    session.report.print();

    unsafe {
        session.ctx.free_buffer(c_buf);
        session.ctx.free_buffer(b_buf);
        session.ctx.free_buffer(a_buf);
    }

    assert!(
        max_abs < abs_tol,
        "abs diff {max_abs:.3e} exceeds forward-error bound {abs_tol:.3e}"
    );
    assert!(
        rel_inf < 1e-5,
        "tensor relative inf-norm error {rel_inf:.3e} exceeds 1e-5"
    );
    assert_eq!(
        session.report.vulkan_ops.get("gemm").copied().unwrap_or(0),
        1
    );
}
