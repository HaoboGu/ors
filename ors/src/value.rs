use ors_sys::*;
use std::{ffi::c_void, ptr::null_mut};

use crate::{api::get_api, status::assert_status};

pub(crate) fn create_tensor_with_data(
    mem_info: *mut OrtMemoryInfo,
    data_ptr: *mut c_void,
    data_len: usize,
    tensor_shape: Vec<i64>,
    element_cnt: usize,
    data_type: ONNXTensorElementDataType,
) -> *mut OrtValue {
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

pub(crate) fn create_tensor_with_ndarray(
    mem_info: *mut OrtMemoryInfo,
    mut array: ndarray::ArrayViewMutD<f32>,
) -> *mut OrtValue {
    let mut ort_value_ptr: *mut OrtValue = null_mut();
    let array_ptr = array.as_mut_ptr() as *mut c_void;
    let array_len = array.len() * std::mem::size_of::<f32>();
    let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
    let shape_ptr: *const i64 = shape.as_ptr();
    let shape_len = array.shape().len();
    let status = unsafe {
        get_api().CreateTensorWithDataAsOrtValue.unwrap()(
            mem_info,
            array_ptr,
            array_len,
            shape_ptr,
            shape_len,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut ort_value_ptr,
        )
    };
    assert_status(status);

    return ort_value_ptr;
}

#[cfg(test)]
mod test {
    use ndarray::{ArrayD, IxDyn};

    use crate::{
        env::create_env,
        log::LoggingLevel,
        session::{
            cast_type_info_to_tensor_info, create_session, get_allocator_mem_info,
            get_default_allocator, get_inputs_tensor_info,
        },
        tensor::{get_dimension_count, get_dimensions},
    };

    use super::*;

    #[test]
    fn test_tensor() {
        let env = create_env(LoggingLevel::Warning, "log_name".to_string());
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/cosy-local-tools/model/model.onnx";

        let session = create_session(env as *const OrtEnv, path, std::ptr::null());
        let allocator = get_default_allocator();
        let input_dims = get_inputs_tensor_info(session);
        for tensor_info in input_dims {
            let dim_cnt = get_dimension_count(tensor_info);
            let dim = get_dimensions(tensor_info, dim_cnt);
            let dim_usize = dim
                .iter()
                .map(|d| if *d <= 0 { 0 } else { *d as usize })
                .collect::<Vec<usize>>();
            println!("dim: {:?}, dim count: {}", dim_usize, dim_cnt);
            let mut array = ArrayD::from_shape_vec(IxDyn(&dim_usize), vec![0.0; 0]).unwrap();
            let ort_value = create_tensor_with_ndarray(
                get_allocator_mem_info(allocator) as *mut OrtMemoryInfo,
                array.view_mut(),
            );
            let mut type_info_ptr = null_mut();
            let status = unsafe { get_api().GetTypeInfo.unwrap()(ort_value, &mut type_info_ptr) };
            let tensor_info = cast_type_info_to_tensor_info(type_info_ptr);
            let tensor_dim_cnt = get_dimension_count(tensor_info);
            let tensor_dim = get_dimensions(tensor_info, tensor_dim_cnt);

            assert_status(status);
            println!("got ort value: {:?}", tensor_dim);
        }
    }
}
