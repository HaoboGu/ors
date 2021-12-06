use ors_sys::*;

use crate::{api::get_api, status::assert_status};

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
fn get_tensor_shape_element_count(type_info: *const OrtTensorTypeAndShapeInfo) -> usize {
    let mut element_cnt = 0;
    let status =
        unsafe { get_api().GetTensorShapeElementCount.unwrap()(type_info, &mut element_cnt) };

    assert_status(status);
    return element_cnt;
}