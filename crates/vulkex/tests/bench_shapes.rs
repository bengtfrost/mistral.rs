use matrixmultiply::sgemm;
use std::time::Instant;
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

fn bench_gemm(m: usize, n: usize, k: usize, iters: usize) {
    let a = pseudo_f32(1, m * k);
    let b = pseudo_f32(2, k * n);

    let mut c_cpu = vec![0f32; m * n];
    // warm up
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
    let t_cpu = {
        let t = Instant::now();
        for _ in 0..iters {
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
        }
        t.elapsed() / iters as u32
    };

    let mut session = Session::new(target_pci()).expect("context");
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

    // warm up (first dispatch may include pipeline lazy-compile cost)
    session.gemm(&a_buf, &b_buf, &c_buf, m as u32, n as u32, k as u32);

    let t_gpu = {
        let t = Instant::now();
        for _ in 0..iters {
            session.gemm(&a_buf, &b_buf, &c_buf, m as u32, n as u32, k as u32);
        }
        t.elapsed() / iters as u32
    };

    let gflops_cpu = 2.0 * (m as f64) * (n as f64) * (k as f64) / t_cpu.as_secs_f64() / 1e9;
    let gflops_gpu = 2.0 * (m as f64) * (n as f64) * (k as f64) / t_gpu.as_secs_f64() / 1e9;
    let ratio = t_cpu.as_secs_f64() / t_gpu.as_secs_f64();

    println!(
        "GEMM {m}x{n}x{k}: cpu={:?} ({gflops_cpu:.1} GFLOPS)  vulkan={:?} ({gflops_gpu:.1} GFLOPS)  ratio={ratio:.2}x",
        t_cpu, t_gpu
    );

    unsafe {
        session.ctx.free_buffer(c_buf);
        session.ctx.free_buffer(b_buf);
        session.ctx.free_buffer(a_buf);
    }
}

fn bench_gemv(m: usize, k: usize, iters: usize) {
    let w = pseudo_f32(1, m * k);
    let x = pseudo_f32(2, k);

    let mut y_cpu = vec![0f32; m];
    // CPU reference: sgemm with N=1 is a matvec. Same microkernel as Candle's CPU path.
    unsafe {
        sgemm(
            m,
            k,
            1,
            1.0,
            w.as_ptr(),
            k as isize,
            1,
            x.as_ptr(),
            1,
            1,
            0.0,
            y_cpu.as_mut_ptr(),
            1,
            1,
        );
    }
    let t_cpu = {
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                sgemm(
                    m,
                    k,
                    1,
                    1.0,
                    w.as_ptr(),
                    k as isize,
                    1,
                    x.as_ptr(),
                    1,
                    1,
                    0.0,
                    y_cpu.as_mut_ptr(),
                    1,
                    1,
                );
            }
        }
        t.elapsed() / iters as u32
    };

    let mut session = Session::new(target_pci()).expect("context");
    let w_buf = unsafe { session.ctx.alloc_mapped((m * k * 4) as u64) }.unwrap();
    let x_buf = unsafe { session.ctx.alloc_mapped((k * 4) as u64) }.unwrap();
    let y_buf = unsafe { session.ctx.alloc_mapped((m * 4) as u64) }.unwrap();
    unsafe {
        w_buf.write_bytes(std::slice::from_raw_parts(
            w.as_ptr() as *const u8,
            w.len() * 4,
        ));
        x_buf.write_bytes(std::slice::from_raw_parts(
            x.as_ptr() as *const u8,
            x.len() * 4,
        ));
    }

    session.gemv(&w_buf, &x_buf, &y_buf, m as u32, k as u32);
    let t_gpu = {
        let t = Instant::now();
        for _ in 0..iters {
            session.gemv(&w_buf, &x_buf, &y_buf, m as u32, k as u32);
        }
        t.elapsed() / iters as u32
    };

    // Decode is bandwidth-bound: the interesting metric is GB/s of weight traffic.
    // Each call reads m*k*4 bytes of weights once.
    let bytes = (m * k * 4) as f64;
    let gbps_cpu = bytes / t_cpu.as_secs_f64() / 1e9;
    let gbps_gpu = bytes / t_gpu.as_secs_f64() / 1e9;
    let ratio = t_cpu.as_secs_f64() / t_gpu.as_secs_f64();

    println!(
        "GEMV {m}x{k}: cpu={:?} ({gbps_cpu:.1} GB/s)  vulkan={:?} ({gbps_gpu:.1} GB/s)  ratio={ratio:.2}x",
        t_cpu, t_gpu
    );

    unsafe {
        session.ctx.free_buffer(y_buf);
        session.ctx.free_buffer(x_buf);
        session.ctx.free_buffer(w_buf);
    }
}

#[test]
#[ignore]
fn bench_qwen_1_7b_shapes() {
    env_logger::builder().is_test(true).try_init().ok();
    println!("--- Qwen3-1.7B prefill shapes, M = 512 ---");
    bench_gemm(512, 6144, 2048, 3);
    bench_gemm(512, 2048, 6144, 3);
    bench_gemm(512, 2048, 2048, 3);
    bench_gemm(512, 151936, 2048, 1);

    println!("--- Qwen3-1.7B decode shapes, M = 1 (GEMV) ---");
    bench_gemv(2048, 2048, 10); // attention projections
    bench_gemv(2048, 6144, 10); // MLP up
    bench_gemv(6144, 2048, 10); // MLP down
    bench_gemv(151936, 2048, 3); // LM head (large m, 1.24 GB at FP32)
}
