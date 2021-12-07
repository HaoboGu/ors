use std::{ffi::c_void, ptr::null_mut};

use ors_sys::*;

use crate::{api::get_api, status::assert_status, types::TypeToTensorElementDataType};

pub(crate) fn get_dimensions(
    type_info: *const OrtTensorTypeAndShapeInfo,
    dimension_cnt: usize,
) -> Vec<i64> {
    let mut dim_values: Vec<i64> = vec![0; dimension_cnt as usize];

    let dim_values_ptr = &mut dim_values;

    let status = unsafe {
        get_api().GetDimensions.unwrap()(type_info, dim_values_ptr.as_mut_ptr(), dimension_cnt)
    };

    assert_status(status);
    return dim_values;
}

pub(crate) fn get_dimension_count(type_info: *const OrtTensorTypeAndShapeInfo) -> usize {
    let mut dimension_cnt = 0;
    let status = unsafe { get_api().GetDimensionsCount.unwrap()(type_info, &mut dimension_cnt) };
    assert_status(status);

    return dimension_cnt;
}

// Get total number of elements in a tensor shape from an OrtTensorTypeAndShapeInfo.
// Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
// For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
pub(crate) fn get_tensor_shape_element_count(type_info: *const OrtTensorTypeAndShapeInfo) -> usize {
    let mut element_cnt = 0;
    let status =
        unsafe { get_api().GetTensorShapeElementCount.unwrap()(type_info, &mut element_cnt) };
    assert_status(status);
    return element_cnt;
}

pub(crate) fn get_tensor_element_type(
    type_info: *const OrtTensorTypeAndShapeInfo,
) -> ONNXTensorElementDataType {
    let mut data_type = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    let status = unsafe { get_api().GetTensorElementType.unwrap()(type_info, &mut data_type) };
    return data_type;
}

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

fn cast_type_info_to_tensor_info(type_info: *const OrtTypeInfo) -> *mut OrtTensorTypeAndShapeInfo {
    let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = null_mut();
    let status =
        unsafe { get_api().CastTypeInfoToTensorInfo.unwrap()(type_info, &mut tensor_info_ptr) };
    assert_status(status);
    return tensor_info_ptr as *mut OrtTensorTypeAndShapeInfo;
}
