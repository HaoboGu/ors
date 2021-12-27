use std::ffi::CStr;
use std::fmt::Debug;
use std::ptr::{null, null_mut};

use anyhow::Result;

use ors_sys::*;

use crate::api::get_api;
use crate::{call_ort, status::check_status};

#[derive(Clone)]
pub struct SessionInputInfo {
    pub name: String,

    pub input_type: ONNXTensorElementDataType,

    pub input_dim: Vec<Option<i64>>,
}

impl Debug for SessionInputInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input name: {}, dim: {:?}", self.name, self.input_dim)
    }
}

#[derive(Clone)]
pub struct SessionOutputInfo {
    pub name: String,

    pub output_type: ONNXTensorElementDataType,

    pub output_dim: Vec<Option<i64>>,
}

impl Debug for SessionOutputInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "output name: {}, dim: {:?}", self.name, self.output_dim)
    }
}

pub(crate) fn get_session_inputs(
    session: *const OrtSession,
    allocator: *mut OrtAllocator,
) -> Result<Vec<SessionInputInfo>> {
    let input_cnt = get_input_count(session)?;
    let mut inputs: Vec<SessionInputInfo> = vec![];
    for i in 0..input_cnt {
        inputs.push(get_session_input(session, i, allocator)?);
    }
    Ok(inputs)
}

pub(crate) fn get_session_outputs(
    session: *const OrtSession,
    allocator: *mut OrtAllocator,
) -> Result<Vec<SessionOutputInfo>> {
    let output_cnt = get_output_count(session)?;
    let mut outputs: Vec<SessionOutputInfo> = vec![];
    for i in 0..output_cnt {
        outputs.push(get_session_output(session, i, allocator)?);
    }
    Ok(outputs)
}

fn get_session_input(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<SessionInputInfo> {
    let input_name = get_input_name(session, index, allocator)?;
    let input_typeinfo = get_input_typeinfo(session, index)?;
    let dim_cnt = get_dimension_count(input_typeinfo)?;
    let input_dim: Vec<Option<i64>> = get_dimensions(input_typeinfo, dim_cnt)?
        .into_iter()
        .map(|d| if d == -1 { None } else { Some(d) })
        .collect();
    let input_type = get_tensor_element_type(input_typeinfo)?;
    release_typeinfo(input_typeinfo);
    Ok(SessionInputInfo {
        name: input_name,
        input_type,
        input_dim,
    })
}

fn get_session_output(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<SessionOutputInfo> {
    let output_name = get_output_name(session, index, allocator)?;
    let output_typeinfo = get_output_typeinfo(session, index)?;
    let dim_cnt = get_dimension_count(output_typeinfo)?;
    let output_dim: Vec<Option<i64>> = get_dimensions(output_typeinfo, dim_cnt)?
        .into_iter()
        .map(|d| if d == -1 { None } else { Some(d) })
        .collect();
    let output_type = get_tensor_element_type(output_typeinfo)?;
    release_typeinfo(output_typeinfo);
    Ok(SessionOutputInfo {
        name: output_name,
        output_type,
        output_dim,
    })
}

fn get_input_count(session: *const OrtSession) -> Result<usize> {
    let mut input_count: usize = 0;
    let input_count_ptr: *mut usize = &mut input_count;
    let status = call_ort!(SessionGetInputCount, session, input_count_ptr);
    check_status(status)?;
    Ok(input_count)
}

fn get_output_count(session: *const OrtSession) -> Result<usize> {
    let mut output_count: usize = 0;
    let output_count_ptr: *mut usize = &mut output_count;
    let status = call_ort!(SessionGetOutputCount, session, output_count_ptr);
    check_status(status)?;
    Ok(output_count)
}

fn get_input_name(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<String> {
    let mut input_name_ptr = null_mut();
    let input_name_ptr_ptr = &mut input_name_ptr;
    let status = call_ort!(
        SessionGetInputName,
        session,
        index,
        allocator,
        input_name_ptr_ptr
    );
    check_status(status)?;
    Ok(unsafe {
        (*(CStr::from_ptr(input_name_ptr)))
            .to_string_lossy()
            .to_string()
    })
}

fn get_output_name(
    session: *const OrtSession,
    index: usize,
    allocator: *mut OrtAllocator,
) -> Result<String> {
    let mut output_name_ptr = null_mut();
    let output_name_ptr_ptr = &mut output_name_ptr;
    let status = call_ort!(
        SessionGetOutputName,
        session,
        index,
        allocator,
        output_name_ptr_ptr
    );
    check_status(status)?;
    Ok(unsafe {
        (*(CStr::from_ptr(output_name_ptr)))
            .to_string_lossy()
            .to_string()
    })
}

fn get_input_typeinfo(
    session: *const OrtSession,
    index: usize,
) -> Result<*const OrtTensorTypeAndShapeInfo> {
    let mut type_info_ptr = null_mut();
    let status = call_ort!(SessionGetInputTypeInfo, session, index, &mut type_info_ptr);
    check_status(status)?;

    let mut tensor_type_info_ptr = null();
    let status = call_ort!(
        CastTypeInfoToTensorInfo,
        type_info_ptr,
        &mut tensor_type_info_ptr
    );
    check_status(status)?;

    Ok(tensor_type_info_ptr)
}

fn get_output_typeinfo(
    session: *const OrtSession,
    index: usize,
) -> Result<*const OrtTensorTypeAndShapeInfo> {
    let mut type_info_ptr = null_mut();
    let status = call_ort!(SessionGetOutputTypeInfo, session, index, &mut type_info_ptr);
    check_status(status)?;

    let mut tensor_type_info_ptr = null();
    let status = call_ort!(
        CastTypeInfoToTensorInfo,
        type_info_ptr,
        &mut tensor_type_info_ptr
    );
    check_status(status)?;

    Ok(tensor_type_info_ptr)
}

fn get_dimensions(
    type_info: *const OrtTensorTypeAndShapeInfo,
    dimension_cnt: usize,
) -> Result<Vec<i64>> {
    let mut dim_values: Vec<i64> = vec![0; dimension_cnt as usize];

    let dim_values_ptr = &mut dim_values;

    let status = call_ort!(
        GetDimensions,
        type_info,
        dim_values_ptr.as_mut_ptr(),
        dimension_cnt
    );
    check_status(status)?;
    Ok(dim_values)
}

fn get_dimension_count(type_info: *const OrtTensorTypeAndShapeInfo) -> Result<usize> {
    let mut dimension_cnt = 0;
    let status = call_ort!(GetDimensionsCount, type_info, &mut dimension_cnt);
    check_status(status)?;

    Ok(dimension_cnt)
}

fn get_tensor_element_type(
    type_info: *const OrtTensorTypeAndShapeInfo,
) -> Result<ONNXTensorElementDataType> {
    let mut data_type = ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    let status = call_ort!(GetTensorElementType, type_info, &mut data_type);
    check_status(status)?;
    Ok(data_type)
}

fn release_typeinfo(type_info: *const OrtTensorTypeAndShapeInfo) {
    call_ort!(
        ReleaseTensorTypeAndShapeInfo,
        type_info as *mut OrtTensorTypeAndShapeInfo
    );
}
