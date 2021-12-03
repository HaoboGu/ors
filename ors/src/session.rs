use std::{
    ffi::CStr,
    ptr::{null, null_mut},
};

#[cfg(target_family = "windows")]
use std::{ffi::OsStr, os::windows::prelude::OsStrExt};

#[cfg(not(target_family = "windows"))]
use std::ffi::CString;

use ors_sys::{
    OrtAllocator, OrtEnv, OrtSession, OrtSessionOptions, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtRunOptions, OrtValue, OrtMemoryInfo,
};

use crate::{api::get_api, status::{assert_status, self}};

// TODO: get all parameters of a single session run
// 
fn session_run(session: *mut OrtSession, run_options: *const OrtRunOptions, input_names: Vec<String>, inputs: Vec<OrtValue>, input_len: usize, output_names: Vec<String>, output_names_len: usize) -> Vec<OrtValue> {


    vec![]
}

fn create_session(
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

fn get_default_allocator() -> *mut OrtAllocator {
    let mut allocator_ptr = null_mut();
    let status = unsafe { get_api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
    assert_status(status);
    return allocator_ptr;
}

fn create_and_register_allocator(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo) -> *mut OrtAllocator {
    let mut allocator_ptr = null_mut();
    let arena_cfg = null();
    let status = unsafe {
        get_api().CreateAndRegisterAllocator.unwrap()(
            env,
            mem_info,
            arena_cfg,
        )
    };
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

fn get_input_typeinfo(session: *const OrtSession, index: usize) -> *mut OrtTypeInfo {
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
    return type_info_ptr;
}

fn get_input_count(session: *const OrtSession) -> usize {
    let mut input_count: usize = 0;
    let input_count_ptr: *mut usize = &mut input_count;
    let status = unsafe { get_api().SessionGetInputCount.unwrap()(session, input_count_ptr) };

    assert_status(status);
    unsafe { *input_count_ptr }
}

fn cast_type_info_to_tensor_info(type_info: *const OrtTypeInfo) -> *mut OrtTensorTypeAndShapeInfo {
    let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = null_mut();
    let status =
        unsafe { get_api().CastTypeInfoToTensorInfo.unwrap()(type_info, &mut tensor_info_ptr) };
    assert_status(status);
    return tensor_info_ptr as *mut OrtTensorTypeAndShapeInfo;
}
// fn get_output_name(se)

mod test {
    use std::time::SystemTime;

    use super::*;
    use crate::{
        env::create_env,
        log::LoggingLevel,
        tensor::{get_dimension_count, get_dimensions}, value::create_tensor_with_data,
    };

    #[test]
    fn test_session() {
        let start = SystemTime::now();
        let env = create_env(LoggingLevel::Warning, "log_name".to_string());
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/cosy-local-tools/model/model.onnx";

        let session = create_session(env as *const OrtEnv, path, std::ptr::null());
        let allocator = get_default_allocator();
        println!("init costs: {:?}", SystemTime::now().duration_since(start));
        let input_cnt = get_input_count(session);
        for i in 0..input_cnt {
            let input_name = get_input_name(session, i, allocator);
            println!("{}", input_name);
            let type_info = get_input_typeinfo(session, i);
            let tensor_info = cast_type_info_to_tensor_info(type_info);
            let dimension_cnt = get_dimension_count(tensor_info);
            println!("dimension cnt: {}", dimension_cnt);
            let dimensions = get_dimensions(tensor_info, dimension_cnt);
            println!("dimensions: {:?}", dimensions);
        }
        println!("a costs: {:?}", SystemTime::now().duration_since(start));
    }
}
