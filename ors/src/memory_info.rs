use crate::{api::get_api, call_ort, status::check_status};
use anyhow::Result;
use ors_sys::*;
use tracing::{debug, warn};

#[derive(Debug)]
pub(crate) struct MemoryInfo {
    pub(crate) ptr: *mut OrtMemoryInfo,
}

impl MemoryInfo {
    pub fn new(allocator: OrtAllocatorType, memory_type: OrtMemType) -> Result<Self> {
        debug!("Creating new memory info.");
        let mut memory_info_ptr: *mut OrtMemoryInfo = std::ptr::null_mut();
        let status = call_ort!(
            CreateCpuMemoryInfo,
            allocator,
            memory_type,
            &mut memory_info_ptr
        );
        check_status(status)?;
        Ok(Self {
            ptr: memory_info_ptr,
        })
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            warn!("MemoryInfo pointer is null, not dropping");
        } else {
            debug!("Dropping the memory info");
            call_ort!(ReleaseMemoryInfo, self.ptr);
        }
        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use std::ptr::null_mut;

    use tracing_test::traced_test;

    use super::*;

    #[test]
    #[traced_test]
    fn test_memory_info_constructor_destructor() {
        let memory_info =
            MemoryInfo::new(OrtAllocatorType_OrtArenaAllocator, OrtMemType_OrtMemTypeCPU).unwrap();
        std::mem::drop(memory_info);
    }

    #[test]
    #[traced_test]
    fn test_drop_empty_memory_info() {
        let memory_info = MemoryInfo { ptr: null_mut() };
        std::mem::drop(memory_info);
    }
}
