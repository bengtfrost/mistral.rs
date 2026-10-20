use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct FallbackReport {
    pub vulkan_ops: BTreeMap<&'static str, u64>,
    pub cpu_fallbacks: BTreeMap<&'static str, u64>,
    pub fallback_bytes: BTreeMap<&'static str, u64>,
    pub submissions: u64,
    pub fence_waits: u64,
    pub flushes: u64,
    pub invalidates: u64,
}

impl FallbackReport {
    pub fn record_vulkan(&mut self, op: &'static str) {
        *self.vulkan_ops.entry(op).or_insert(0) += 1;
    }

    pub fn record_fallback(&mut self, op: &'static str, bytes: u64) {
        *self.cpu_fallbacks.entry(op).or_insert(0) += 1;
        *self.fallback_bytes.entry(op).or_insert(0) += bytes;
    }

    pub fn print(&self) {
        println!("=== vulkex session report ===");
        println!(
            "submissions={} fence_waits={} flushes={} invalidates={}",
            self.submissions, self.fence_waits, self.flushes, self.invalidates
        );
        for (op, n) in &self.vulkan_ops {
            println!("  {op}: {n} on Vulkan");
        }
        for (op, n) in &self.cpu_fallbacks {
            let b = self.fallback_bytes.get(op).copied().unwrap_or(0);
            println!("  {op}: {n} fell back to CPU ({b} bytes)");
        }
    }
}
