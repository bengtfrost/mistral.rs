use ash::vk;

pub struct Buffer {
    pub raw: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub coherent: bool,
    pub mapped: *mut u8,
}

impl Buffer {
    pub fn is_mapped(&self) -> bool {
        !self.mapped.is_null()
    }

    /// # Safety
    /// Caller must ensure the GPU is not currently reading or writing this buffer.
    pub unsafe fn write_bytes(&self, data: &[u8]) {
        assert!(data.len() as u64 <= self.size);
        assert!(self.is_mapped(), "buffer not host-visible");
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped, data.len()) };
    }

    /// # Safety
    /// Caller must ensure the GPU has finished writing before calling this.
    pub unsafe fn read_bytes(&self, len: usize) -> Vec<u8> {
        assert!(len as u64 <= self.size);
        assert!(self.is_mapped(), "buffer not host-visible");
        let mut out = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr(), len) };
        out
    }
}
