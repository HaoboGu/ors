use std::{
    ffi::CStr,
    ptr::{null, null_mut},
    vec,
};

#[cfg(target_family = "windows")]
use std::{ffi::OsStr, os::windows::prelude::OsStrExt};

#[cfg(not(target_family = "windows"))]
use std::ffi::CString;

use ndarray::{ArrayD, IxDyn};
use ors_sys::{
    OrtAllocator, OrtEnv, OrtMemoryInfo, OrtRunOptions, OrtSession, OrtSessionOptions,
    OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue,
};

use crate::{
    api::get_api,
    status::assert_status,
    tensor::{get_dimension_count, get_dimensions, get_tensor_shape_element_count},
    value::create_tensor_with_ndarray,
};

pub(crate) fn get_inputs_tensor_info(
    session: *mut OrtSession,
) -> Vec<*const OrtTensorTypeAndShapeInfo> {
    let input_cnt = get_input_count(session);
    let mut inputs_tensor_info = vec![];
    for i in 0..input_cnt {
        let type_info = get_input_typeinfo(session, i);
        inputs_tensor_info.push(type_info);
    }
    return inputs_tensor_info;
}

pub(crate) fn get_outputs_tensor_info(
    session: *mut OrtSession,
) -> Vec<*const OrtTensorTypeAndShapeInfo> {
    let output_cnt = get_output_count(session);
    let mut outputs_tensor_info = vec![];
    for i in 0..output_cnt {
        let type_info = get_output_typeinfo(session, i);
        outputs_tensor_info.push(type_info);
    }
    return outputs_tensor_info;
}

//
fn session_run(
    session: *mut OrtSession,
    run_options: *const OrtRunOptions,
    input_names: Vec<String>,
    inputs: Vec<*const OrtValue>,
    input_len: usize,
    output_names: Vec<String>,
    output_names_len: usize,
) -> Vec<*mut OrtValue> {

    let input_names_ptr: Vec<*const i8> = input_names
        .iter()
        .map(|n| CString::new(n.clone()).unwrap())
        .map(|n| n.into_raw() as *const i8)
        .collect();
    
    let output_names_cstring: Vec<CString> = output_names
    .iter()
    .map(|n| CString::new(n.clone()).unwrap()).collect();

    let output_names_ptr: Vec<*const i8> = 
        output_names_cstring
        .iter()
        .map(|n| n.as_ptr() as *const i8)
        .collect();

    let mut outputs = vec![];

    for i in 0..output_names_len {
        let output_typeinfo = get_output_typeinfo(session, i);
        let mut dim = get_dimensions(output_typeinfo, get_dimension_count(output_typeinfo));
        if i == 0 {
            println!("output dim for i: {:?}", dim);
            dim = vec![1, 2, 39949];
        } else {
            dim = vec![2, 1, 6, 2, 64];
        }
        let dim_usize = dim
            .iter()
            .map(|d| if *d <= 0 { 0 } else { *d as usize })
            .collect::<Vec<usize>>();

        // let mut element_count = get_tensor_shape_element_count(output_typeinfo);
        let mut element_count = 0;
        if i == 0 {
            element_count = 2* 39949;
        } else {
            element_count = 2 * 2 * 6 * 64;
        }
        println!("output dims: {:?}, element cnt: {}", dim_usize, element_count);
        let output_array =
            ArrayD::from_shape_vec(IxDyn(&dim_usize), vec![0.0; element_count]).unwrap();
        outputs.push(output_array);
    }

    let mut output_tensors = vec![];
    for mut array in outputs {
        println!("shape: {:?}", array.shape());
        let tensor = create_tensor_with_ndarray(
            get_allocator_mem_info(get_default_allocator() as *mut OrtAllocator)
                as *mut OrtMemoryInfo,
            array.view_mut(),
        );
        output_tensors.push(tensor);
    }

    let status = unsafe {
        get_api().Run.unwrap()(
            session,
            run_options,
            input_names_ptr.as_ptr(),
            inputs.as_ptr(),
            inputs.len(),
            output_names_ptr.as_ptr(),
            output_names_ptr.len(),
            output_tensors.as_mut_ptr(),
        )
    };
    assert_status(status);
    return output_tensors;
}

pub(crate) fn create_session(
    env: *const OrtEnv,
    model_path: &str,
    options: *const OrtSessionOptions,
) -> *mut OrtSession {
    let mut session_ptr = null_mut();

    #[cfg(target_family = "windows")]
    let c_model_path = OsStr::new(model_path)
        .encode_wide()
        .chain(Some(0)) // add NULL termination
        .collect::<Vec<_>>();
    #[cfg(not(target_family = "windows"))]
    let c_model_path = CString::new(model_path).unwrap();

    let status = unsafe {
        get_api().CreateSession.unwrap()(env, c_model_path.as_ptr(), options, &mut session_ptr)
    };
    assert_status(status);
    return session_ptr;
}

pub(crate) fn get_default_allocator() -> *mut OrtAllocator {
    let mut allocator_ptr = null_mut();
    let status = unsafe { get_api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
    assert_status(status);
    return allocator_ptr;
}

fn create_and_register_allocator(
    env: *mut OrtEnv,
    mem_info: *const OrtMemoryInfo,
) -> *mut OrtAllocator {
    let allocator_ptr = null_mut();
    let arena_cfg = null();
    let status = unsafe { get_api().CreateAndRegisterAllocator.unwrap()(env, mem_info, arena_cfg) };
    assert_status(status);
    return allocator_ptr;
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
    assert_status(status);
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
    assert_status(status);
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
    assert_status(status);

    let mut tensor_type_info_ptr = null();
    let status = unsafe {
        get_api().CastTypeInfoToTensorInfo.unwrap()(type_info_ptr, &mut tensor_type_info_ptr)
    };
    assert_status(status);

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
    assert_status(status);

    let mut tensor_type_info_ptr = null();
    let status = unsafe {
        get_api().CastTypeInfoToTensorInfo.unwrap()(type_info_ptr, &mut tensor_type_info_ptr)
    };
    assert_status(status);

    // TODO: this type info must be freed by `OrtApi::ReleaseTypeInfo`
    return tensor_type_info_ptr;
}

fn get_output_count(session: *const OrtSession) -> usize {
    let mut output_count: usize = 0;
    let output_count_ptr: *mut usize = &mut output_count;
    let status = unsafe { get_api().SessionGetOutputCount.unwrap()(session, output_count_ptr) };

    assert_status(status);
    output_count
}

fn get_input_count(session: *const OrtSession) -> usize {
    let mut input_count: usize = 0;
    let input_count_ptr: *mut usize = &mut input_count;
    let status = unsafe { get_api().SessionGetInputCount.unwrap()(session, input_count_ptr) };

    assert_status(status);
    input_count
}

pub(crate) fn get_allocator_mem_info(allocator: *const OrtAllocator) -> *const OrtMemoryInfo {
    let mut mem_info_ptr = null();
    let status = unsafe { get_api().AllocatorGetInfo.unwrap()(allocator, &mut mem_info_ptr) };
    assert_status(status);
    return mem_info_ptr;
}

pub(crate) fn cast_type_info_to_tensor_info(
    type_info: *const OrtTypeInfo,
) -> *mut OrtTensorTypeAndShapeInfo {
    let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = null_mut();
    let status =
        unsafe { get_api().CastTypeInfoToTensorInfo.unwrap()(type_info, &mut tensor_info_ptr) };
    assert_status(status);
    return tensor_info_ptr as *mut OrtTensorTypeAndShapeInfo;
}
// fn get_output_name(se)

#[cfg(test)]
mod test {
    use std::time::SystemTime;

    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::{
        env::create_env,
        log::LoggingLevel,
        tensor::{get_dimension_count, get_dimensions, get_tensor_shape_element_count, get_tensor_element_type},
        value::create_tensor_with_ndarray,
    };

    #[test]
    fn test_session() {
        let start = SystemTime::now();
        let env = create_env(LoggingLevel::Warning, "log_name".to_string());
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        // let path = "/Users/haobogu/Projects/rust/cosy-local-tools/model/model.onnx";
        let path = "/Users/haobogu/Projects/rust/ors/large_model.onnx";

        let session = create_session(env as *const OrtEnv, path, std::ptr::null());
        let allocator = get_default_allocator();
        println!("init costs: {:?}", SystemTime::now().duration_since(start));
        let input_cnt = get_input_count(session);
        let output_cnt = get_output_count(session);
        println!("input cnt: {}, output cnt: {}", input_cnt, output_cnt);
        let mut input_names: Vec<String> = vec![];
        let mut inputs: Vec<*const OrtValue> = vec![];
        for i in 0..input_cnt {
            let input_name = get_input_name(session, i, allocator);
            input_names.push(input_name.clone());
            let tensor_info = get_input_typeinfo(session, i);
            let dimension_cnt = get_dimension_count(tensor_info);
            let dimensions = get_dimensions(tensor_info, dimension_cnt);
            println!("dim: {:?} for input: {}", dimensions, input_name);
            if input_name == "input_ids" {
                let mut array = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap();
                let ort_value_ptr = create_tensor_with_ndarray(
                    get_allocator_mem_info(allocator) as *mut OrtMemoryInfo,
                    array.view_mut(),
                );
                println!("input shape: {:?}", array.shape());
                inputs.push(ort_value_ptr as *const OrtValue);
            } else {
                let mut array =
                    ArrayD::from_shape_vec(IxDyn(&[2, 1, 6, 1, 64]), vec![1.0;2*6*64]).unwrap();
                println!("input shape: {:?}", array.shape());
                let ort_value_ptr = create_tensor_with_ndarray(
                    get_allocator_mem_info(allocator) as *mut OrtMemoryInfo,
                    array.view_mut(),
                );
                inputs.push(ort_value_ptr as *const OrtValue);
            }
        }
        let mut output_names: Vec<String> = vec![];
        for i in 0..output_cnt {
            let output_name = get_output_name(session, i, allocator);
            output_names.push(output_name);
        }
        let input_len = inputs.len();
        session_run(
            session,
            null(),
            input_names,
            inputs,
            input_len,
            output_names,
            output_cnt,
        );
        println!("a costs: {:?}", SystemTime::now().duration_since(start));
    }
}
