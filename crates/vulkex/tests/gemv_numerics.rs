use vulkex::Session;

fn pseudo_random_f32(seed: u32, len: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut s = seed;
    for _ in 0..len {
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
fn gemv_fp32_matches_cpu_within_tolerance() {
    env_logger::builder().is_test(true).try_init().ok();

    let rows: u32 = 4096;
    let cols: u32 = 4096;
    let weights = pseudo_random_f32(0x9E3779B9, (rows * cols) as usize);
    let input = pseudo_random_f32(0x85EBCA6B, cols as usize);

    let mut reference = vec![0f64; rows as usize];
    for r in 0..rows as usize {
        let base = r * cols as usize;
        let mut s = 0f64;
        for c in 0..cols as usize {
            s += weights[base + c] as f64 * input[c] as f64;
        }
        reference[r] = s;
    }

    let mut session = Session::new(target_pci()).expect("context");
    assert!(
        session.ctx.validation_enabled,
        "VK_INSTANCE_LAYERS requested VK_LAYER_KHRONOS_validation but the layer is not loaded"
    );

    let w_buf = unsafe { session.ctx.alloc_mapped((rows as u64) * (cols as u64) * 4) }.unwrap();
    let x_buf = unsafe { session.ctx.alloc_mapped((cols as u64) * 4) }.unwrap();
    let y_buf = unsafe { session.ctx.alloc_mapped((rows as u64) * 4) }.unwrap();

    unsafe {
        w_buf.write_bytes(std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * 4,
        ));
        x_buf.write_bytes(std::slice::from_raw_parts(
            input.as_ptr() as *const u8,
            input.len() * 4,
        ));
    }

    session.gemv(&w_buf, &x_buf, &y_buf, rows, cols);

    let out_bytes = unsafe { y_buf.read_bytes((rows as usize) * 4) };
    let gpu: Vec<f32> = (0..rows as usize)
        .map(|i| f32::from_le_bytes(out_bytes[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (g, r) in gpu.iter().zip(reference.iter()) {
        let abs = (g - (*r as f32)).abs();
        let rel = abs / (*r as f32).abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }

    println!(
        "gemv {rows}x{cols}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, \
         validation={}, coherent_heap={}",
        session.ctx.validation_enabled, session.ctx.all_host_visible_coherent
    );
    session.report.print();

    unsafe {
        session.ctx.free_buffer(y_buf);
        session.ctx.free_buffer(x_buf);
        session.ctx.free_buffer(w_buf);
    }

    assert!(max_abs < 1e-4, "abs diff too large: {max_abs}");
    assert!(max_rel < 1e-3, "rel diff too large: {max_rel}");
    assert_eq!(
        session.report.vulkan_ops.get("gemv").copied().unwrap_or(0),
        1
    );
}
