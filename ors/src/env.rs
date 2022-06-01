use std::{
    ffi::CString,
    ops::DerefMut,
    ptr::null_mut,
    sync::{atomic::AtomicPtr, Arc, Mutex},
};

use crate::{api::get_api, call_ort, log::custom_logger, status::check_status};
use lazy_static::lazy_static;
use ors_sys::*;
use tracing::debug;

lazy_static! {
    static ref ENV: Arc<Mutex<AtomicPtr<OrtEnv>>> =
        Arc::new(Mutex::new(AtomicPtr::new(create_env())));
}

// This function can be only called once
fn create_env() -> *mut OrtEnv {
    debug!("Creating onnxruntime environment");
    let mut env_ptr: *mut OrtEnv = std::ptr::null_mut();
    let logging_function: OrtLoggingFunction = Some(custom_logger);
    let logger_param: *mut std::ffi::c_void = null_mut();
    let name = CString::new("onnxruntime").unwrap();

    let status = call_ort!(
        CreateEnvWithCustomLogger,
        logging_function,
        logger_param,
        OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
        name.as_ptr(),
        &mut env_ptr
    );

    // panic when failed to create env
    check_status(status).expect("Failed to create inference environment");

    env_ptr
}

pub(crate) fn get_env_ptr() -> *mut OrtEnv {
    let mut env_guard = ENV.try_lock().unwrap();
    *env_guard.deref_mut().get_mut()
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use tracing_test::traced_test;

    use crate::api::initialize_runtime;

    use super::*;

    #[test]
    #[traced_test]
    fn test_env() {
        setup_runtime();
        let p = get_env_ptr();
        assert_ne!(p, null_mut());
    }

    fn setup_runtime() {
        #[cfg(target_os = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\onnxruntime.dll";
        #[cfg(target_os = "macos")]
        let path = "/usr/local/lib/libonnxruntime.1.11.1.dylib";
        #[cfg(target_os = "linux")]
        let path = "/usr/local/lib/libonnxruntime.so";
        initialize_runtime(Path::new(path)).unwrap();
    }
}
