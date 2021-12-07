use ors_sys::*;
use std::{ffi::c_void, ptr::null_mut};

use crate::{api::get_api, status::assert_status, types::TypeToTensorElementDataType};

pub(crate) fn create_tensor_with_ndarray<T>(
    mem_info: *mut OrtMemoryInfo,
    mut array: ndarray::ArrayViewMutD<T>,
) -> *mut OrtValue
where
    T: TypeToTensorElementDataType,
{
    let mut ort_value_ptr: *mut OrtValue = null_mut();
    let array_ptr = array.as_mut_ptr() as *mut c_void;
    let array_len = array.len() * std::mem::size_of::<T>();
    let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
    let shape_ptr: *const i64 = shape.as_ptr();
    let shape_len = array.shape().len();
    let onnx_data_type = T::tensor_element_data_type();
    let status = unsafe {
        get_api().CreateTensorWithDataAsOrtValue.unwrap()(
            mem_info,
            array_ptr,
            array_len,
            shape_ptr,
            shape_len,
            onnx_data_type,
            &mut ort_value_ptr,
        )
    };
    assert_status(status);

    return ort_value_ptr;
}
