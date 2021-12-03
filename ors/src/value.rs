use ors_sys::*;
use std::{ffi::{CStr, c_void}, ptr::null_mut};

use crate::{status::assert_status, api::get_api};

pub(crate) fn create_tensor_with_data(mem_info: *mut OrtMemoryInfo, data_ptr: *mut c_void, data_len: usize, tensor_shape: Vec<i64>, element_cnt: usize, data_type: ONNXTensorElementDataType) -> *mut OrtValue {
    let mut ort_value_ptr: *mut OrtValue = null_mut();
    let status = unsafe {
        get_api().CreateTensorWithDataAsOrtValue.unwrap()(
            mem_info,
            data_ptr,
            data_len,
            tensor_shape.as_ptr(),
            element_cnt,
            data_type,
            &mut ort_value_ptr,
        )
    };

    assert_status(status);

    // TODO: this ort value must be freed by `OrtApi::ReleaseValue`
    return ort_value_ptr;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
    }
}
