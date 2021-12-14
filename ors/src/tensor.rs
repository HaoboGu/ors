use std::{ffi::c_void, ptr::null_mut};

use ors_sys::*;

use anyhow::Result;

use crate::{api::get_api, call_ort, status::check_status, types::TypeToTensorElementDataType};

struct Tensor {}

fn create_tensor_with_ndarray<T>(
    mem_info: *mut OrtMemoryInfo,
    mut array: ndarray::ArrayViewMutD<T>,
) -> Result<*mut OrtValue>
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
    let status = call_ort!(
        CreateTensorWithDataAsOrtValue,
        mem_info,
        array_ptr,
        array_len,
        shape_ptr,
        shape_len,
        onnx_data_type,
        &mut ort_value_ptr
    );
    check_status(status)?;

    Ok(ort_value_ptr)
}

#[cfg(test)]
mod test {
    use ndarray::{ArrayD, IxDyn};
    use tracing_test::traced_test;

    use crate::{memory_info::MemoryInfo, session::SessionBuilder};

    use super::*;

    #[test]
    #[traced_test]
    fn test_tensor_creation() {
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        let mem_info =
            MemoryInfo::new(OrtAllocatorType_OrtArenaAllocator, OrtMemType_OrtMemTypeCPU).unwrap();
        let session_builder = SessionBuilder::new().unwrap();
        let session = session_builder.build_with_model_from_file(path).unwrap();
        let mut array = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 2]), vec![0; 2]).unwrap();
        let tensor = create_tensor_with_ndarray::<i64>(mem_info.ptr, array.view_mut()).unwrap();
        assert_ne!(tensor, null_mut());
        let mut array2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[1, 2]), vec![0.; 2]).unwrap();
        let tensor2 = create_tensor_with_ndarray::<f32>(mem_info.ptr, array2.view_mut()).unwrap();
        assert_ne!(tensor2, null_mut());
        assert_ne!(tensor, tensor2);
    }
}
