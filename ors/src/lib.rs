#![allow(dead_code)]
#![allow(unused_variables)]
pub mod api;
pub mod config;
pub mod env;
pub mod log;
pub mod memory_info;
pub mod session;
mod status;
pub mod tensor;
pub mod types;

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
