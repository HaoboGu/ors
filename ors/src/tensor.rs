use std::{ffi::c_void, ptr::null_mut};

use anyhow::Result;
use enum_as_inner::EnumAsInner;
use enum_dispatch::enum_dispatch;
use ndarray::ArrayD;
use ors_sys::*;

use crate::{
    api::get_api, call_ort, memory_info::MemoryInfo, session::get_default_memory_info,
    status::check_status, types::TypeToTensorElementDataType,
};

#[derive(Debug, EnumAsInner)]
#[enum_dispatch]
pub enum TypedArray {
    F32Array(ArrayD<f32>),
    F64Array(ArrayD<f64>),
    I8Array(ArrayD<i8>),
    I16Array(ArrayD<i16>),
    I32Array(ArrayD<i32>),
    I64Array(ArrayD<i64>),
    U8Array(ArrayD<u8>),
    U16Array(ArrayD<u16>),
    U32Array(ArrayD<u32>),
    U64Array(ArrayD<u64>),
}

#[enum_dispatch(TypedArray)]
pub trait TypeToOnnxTensor {}

impl<T: TypeToTensorElementDataType> TypeToOnnxTensor for ArrayD<T> {}

// Tensor stores OrtValue ptr and owns tensor data
#[derive(Debug)]
pub struct Tensor {
    pub(crate) ptr: *mut OrtValue,
    pub data: TypedArray,
}

/// Expose owned data
#[macro_export]
macro_rules! convert_typed_array {
    ($n:ident, $t:ty, $pattern:pat => $extracted_value:expr) => {
        impl Tensor {
            /// Get owned data from tensor
            /// If the target type is not right, returns None
            fn $n(&self) -> Option<&ArrayD<$t>> {
                self.data.$n()
            }
        }
    };
}

convert_typed_array!(as_f32_array, f32, TypedArray::F32Array(d) => d);
convert_typed_array!(as_f64_array, f64, TypedArray::F64Array(d) => d);
convert_typed_array!(as_i8_array, i8, TypedArray::I8Array(d) => d);
convert_typed_array!(as_i16_array, i16, TypedArray::I16Array(d) => d);
convert_typed_array!(as_i32_array, i32, TypedArray::I32Array(d) => d);
convert_typed_array!(as_i64_array, i64, TypedArray::I64Array(d) => d);
convert_typed_array!(as_u8_array, u8, TypedArray::U8Array(d) => d);
convert_typed_array!(as_u16_array, u16, TypedArray::U16Array(d) => d);
convert_typed_array!(as_u32_array, u32, TypedArray::U32Array(d) => d);
convert_typed_array!(as_u64_array, u64, TypedArray::U64Array(d) => d);

pub fn create_tensor_with_ndarray_and_mem_info<T>(
    memory_info: &MemoryInfo,
    mut array: ArrayD<T>,
) -> Result<Tensor>
where
    T: TypeToTensorElementDataType,
    TypedArray: From<ArrayD<T>>,
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
        memory_info.ptr,
        array_ptr,
        array_len,
        shape_ptr,
        shape_len,
        onnx_data_type,
        &mut ort_value_ptr
    );
    check_status(status)?;

    Ok(Tensor {
        ptr: ort_value_ptr,
        data: TypedArray::from(array),
    })
}

// The ndarray must live longer than tensor
pub fn create_tensor_with_ndarray<T>(mut array: ndarray::ArrayD<T>) -> Result<Tensor>
where
    T: TypeToTensorElementDataType,
    TypedArray: From<ArrayD<T>>,
{
    let mut ort_value_ptr: *mut OrtValue = null_mut();
    let array_ptr = array.as_mut_ptr() as *mut c_void;
    let array_len = array.len() * std::mem::size_of::<T>();
    let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
    let shape_ptr: *const i64 = shape.as_ptr();
    let shape_len = array.shape().len();
    let onnx_data_type = T::tensor_element_data_type();
    let mem_info = get_default_memory_info()?;
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

    Ok(Tensor {
        ptr: ort_value_ptr,
        data: TypedArray::from(array),
    })
}

#[cfg(test)]
mod test {
    use std::time::SystemTime;

    use ndarray::{ArrayD, IxDyn};
    use tracing::info;
    use tracing_test::traced_test;

    use crate::session::SessionBuilder;

    use super::*;

    #[test]
    #[traced_test]
    fn test_tensor_creation() {
        let path = get_test_model_path();
        let session_builder = SessionBuilder::new().unwrap();
        let session = session_builder.build_with_model_from_file(path).unwrap();
        let array = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 2]), vec![0; 2]).unwrap();
        let start = SystemTime::now();
        let tensor = create_tensor_with_ndarray::<i64>(array).unwrap();
        info!(
            "creation of tensor costs: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        assert_ne!(tensor.ptr, null_mut());
        let array2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[1, 2]), vec![0.; 2]).unwrap();
        let start = SystemTime::now();
        let tensor2 = create_tensor_with_ndarray::<f32>(array2).unwrap();
        info!(
            "creation of tensor costs: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        assert_ne!(tensor2.ptr, null_mut());
        assert_ne!(tensor.ptr, tensor2.ptr);
    }

    #[test]
    #[traced_test]
    fn test_tensor_creation_with_memory_info() {
        let path = get_test_model_path();
        let session_builder = SessionBuilder::new().unwrap();
        let session = session_builder.build_with_model_from_file(path).unwrap();
        let array = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 2]), vec![0; 2]).unwrap();
        let memory_info = MemoryInfo::new(
            OrtAllocatorType_OrtDeviceAllocator,
            OrtMemType_OrtMemTypeCPU,
        )
        .unwrap();
        let start = SystemTime::now();
        let tensor = create_tensor_with_ndarray_and_mem_info(&memory_info, array).unwrap();
        info!(
            "creation of tensor with memory info costs: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        assert_ne!(tensor.ptr, null_mut());
        let start = SystemTime::now();
        let tensor2 = create_tensor_with_ndarray_and_mem_info::<f32>(
            &memory_info,
            ArrayD::<f32>::from_shape_vec(IxDyn(&[1, 2]), vec![0.; 2]).unwrap(),
        )
        .unwrap();
        info!(
            "creation of tensor with memory info costs: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        assert_ne!(tensor2.ptr, null_mut());
        assert_ne!(tensor.ptr, tensor2.ptr);
    }

    fn get_test_model_path() -> &'static str {
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        return path;
    }
}
