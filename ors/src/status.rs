use crate::{api::get_api, call_ort};
use anyhow::{anyhow, Result};
use ors_sys::*;
use std::ffi::{CStr, CString};

/// Create an OrtStatus from a null terminated string
fn create_status(error_code: OrtErrorCode, msg: String) -> *const OrtStatus {
    let msg = CString::new(msg).unwrap();
    call_ort!(CreateStatus, error_code, msg.as_ptr())
}

/// Get OrtErrorCode from OrtStatus
fn get_error_code(status: *const OrtStatus) -> OrtErrorCode {
    call_ort!(GetErrorCode, status)
}

/// Release an OrtStatus
fn release_status(status: *mut OrtStatus) {
    call_ort!(ReleaseStatus, status)
}

/// Get error string from OrtStatus
fn get_error_msg(status: *const OrtStatus) -> String {
    let msg = unsafe { CStr::from_ptr(get_api().GetErrorMessage.unwrap()(status)) };
    (*msg.to_string_lossy()).to_string()
}

/// Check an OrtStatus, returns Ok(()) if the api runs good
pub(crate) fn check_status(status: *mut OrtStatus) -> Result<()> {
    if status.is_null() {
        Ok(())
    } else if OrtErrorCode_ORT_OK == get_error_code(status) {
        release_status(status);
        Ok(())
    } else {
        // Extract onnxruntime error and then release the status
        let err = anyhow!(
            "onnxruntime error: {}:{}",
            get_error_code(status),
            get_error_msg(status)
        );
        release_status(status);
        Err(err)
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use crate::api::initialize_runtime;

    use super::*;

    #[test]
    fn test_ort_status() {
        setup_runtime();
        let status = create_status(OrtErrorCode_ORT_MODEL_LOADED, "OKOKOKO".to_string());
        assert_eq!(8, get_error_code(status));
        release_status(status as *mut OrtStatus);
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
