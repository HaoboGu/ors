use std::{
    ffi::CStr,
    ptr::{null, null_mut},
};

use ors_sys::*;

use crate::{
    api::get_api,
    status::check_status,
    tensor::{get_dimension_count, get_dimensions, get_tensor_element_type},
};

#[derive(Debug, Clone)]
pub struct SessionInputInfo {
    pub name: String,

    pub input_type: ONNXTensorElementDataType,

    pub input_dim: Vec<Option<i64>>,
}

#[derive(Debug, Clone)]
pub struct SessionOutputInfo {
    pub name: String,

    pub output_type: ONNXTensorElementDataType,

    pub output_dim: Vec<Option<i64>>,
}

pub(crate) fn get_session_input(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> SessionInputInfo {
    let input_name = get_input_name(session, index, allocator);
    let input_typeinfo = get_input_typeinfo(session, index);
    let dim_cnt = get_dimension_count(input_typeinfo);
    let input_dim: Vec<Option<i64>> = get_dimensions(input_typeinfo, dim_cnt)
        .into_iter()
        .map(|d| if d == -1 { None } else { Some(d) })
        .collect();
    let input_type = get_tensor_element_type(input_typeinfo);
    return SessionInputInfo {
        name: input_name,
        input_type,
        input_dim,
    };
}

pub(crate) fn get_session_output(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> SessionOutputInfo {
    let output_name = get_output_name(session, index, allocator);
    let output_typeinfo = get_output_typeinfo(session, index);
    let dim_cnt = get_dimension_count(output_typeinfo);
    let output_dim: Vec<Option<i64>> = get_dimensions(output_typeinfo, dim_cnt)
        .into_iter()
        .map(|d| if d == -1 { None } else { Some(d) })
        .collect();
    let output_type = get_tensor_element_type(output_typeinfo);
    return SessionOutputInfo {
        name: output_name,
        output_type,
        output_dim,
    };
}

fn get_input_name(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> String {
    let mut input_name_ptr = null_mut();
    let input_name_ptr_ptr = &mut input_name_ptr;
    let status = unsafe {
        get_api().SessionGetInputName.unwrap()(session, index, allocator, input_name_ptr_ptr)
    };
    check_status(status);
    unsafe {
        (*(CStr::from_ptr(input_name_ptr)))
            .to_string_lossy()
            .to_string()
    }
}

fn get_output_name(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> String {
    let mut output_name_ptr = null_mut();
    let output_name_ptr_ptr = &mut output_name_ptr;
    let status = unsafe {
        get_api().SessionGetOutputName.unwrap()(session, index, allocator, output_name_ptr_ptr)
    };
    check_status(status);
    unsafe {
        (*(CStr::from_ptr(output_name_ptr)))
            .to_string_lossy()
            .to_string()
    }
}

fn get_input_typeinfo(
    session: *const OrtSession,
    index: usize,
) -> *const OrtTensorTypeAndShapeInfo {
    let mut type_info_ptr = null_mut();
    let status =
        unsafe { get_api().SessionGetInputTypeInfo.unwrap()(session, index, &mut type_info_ptr) };
    check_status(status);

    let mut tensor_type_info_ptr = null();
    let status = unsafe {
        get_api().CastTypeInfoToTensorInfo.unwrap()(type_info_ptr, &mut tensor_type_info_ptr)
    };
    check_status(status);

    // TODO: this type info must be freed by `OrtApi::ReleaseTypeInfo`
    return tensor_type_info_ptr;
}

fn get_output_typeinfo(
    session: *const OrtSession,
    index: usize,
) -> *const OrtTensorTypeAndShapeInfo {
    let mut type_info_ptr = null_mut();
    let status =
        unsafe { get_api().SessionGetOutputTypeInfo.unwrap()(session, index, &mut type_info_ptr) };
    check_status(status);

    let mut tensor_type_info_ptr = null();
    let status = unsafe {
        get_api().CastTypeInfoToTensorInfo.unwrap()(type_info_ptr, &mut tensor_type_info_ptr)
    };
    check_status(status);

    // TODO: this type info must be freed by `OrtApi::ReleaseTypeInfo`
    return tensor_type_info_ptr;
}

pub(crate) fn get_output_count(session: *const OrtSession) -> usize {
    let mut output_count: usize = 0;
    let output_count_ptr: *mut usize = &mut output_count;
    let status = unsafe { get_api().SessionGetOutputCount.unwrap()(session, output_count_ptr) };

    check_status(status);
    output_count
}

pub(crate) fn get_input_count(session: *const OrtSession) -> usize {
    let mut input_count: usize = 0;
    let input_count_ptr: *mut usize = &mut input_count;
    let status = unsafe { get_api().SessionGetInputCount.unwrap()(session, input_count_ptr) };

    check_status(status);
    input_count
}
