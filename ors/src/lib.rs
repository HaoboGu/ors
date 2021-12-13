#![allow(dead_code)]
#![allow(unused_variables)]
mod api;
mod env;
// mod error;
mod log;
mod session;
// mod session_io;
mod status;
// mod tensor;
mod types;
use ors_sys::*;

/// Optimization level performed by ONNX Runtime of the loaded graph
///
/// See the [official documentation](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md)
/// for more information on the different optimization levels.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum OptimizationLevel {
    /// Disable optimization
    DisableAll = GraphOptimizationLevel_ORT_DISABLE_ALL,
    /// Basic optimization
    Basic = GraphOptimizationLevel_ORT_ENABLE_BASIC,
    /// Extended optimization
    Extended = GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    /// Add optimization
    All = GraphOptimizationLevel_ORT_ENABLE_ALL,
}

impl From<OptimizationLevel> for GraphOptimizationLevel {
    fn from(val: OptimizationLevel) -> Self {
        match val {
            OptimizationLevel::DisableAll => GraphOptimizationLevel_ORT_DISABLE_ALL,
            OptimizationLevel::Basic => GraphOptimizationLevel_ORT_ENABLE_BASIC,
            OptimizationLevel::Extended => GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
            OptimizationLevel::All => GraphOptimizationLevel_ORT_ENABLE_ALL,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, ptr::null};

    use crate::api::get_api;

    #[test]
    fn it_works() {
        // Suppose that onnxruntime's dynamic library has already added in PATH
        assert_eq!(8, ors_sys::ORT_API_VERSION);
        println!("onnxruntime api verseion: {}", ors_sys::ORT_API_VERSION);
        let error_code = 1;
        let msg_ptr: *const i8 = std::ptr::null_mut();
        let create_status_fn = get_api().CreateStatus.unwrap();
        let status_ptr = unsafe { create_status_fn(error_code.try_into().unwrap(), msg_ptr) };
        assert_ne!(null(), status_ptr);
        println!("{:?}", status_ptr);
    }
}
