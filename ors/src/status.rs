use crate::{api::get_api, call_ort};
use ors_sys::*;
use std::ffi::{CStr, CString};
use anyhow::{anyhow, Result};

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

// Check an OrtStatus, returns Ok(()) if the api runs good
pub fn check_status(status: *mut OrtStatus) -> Result<()> {
    if status.is_null() || OrtErrorCode_ORT_OK == get_error_code(status) {
        Ok(())
    } else {
        // Extract onnxruntime error and then release the status
        let err = anyhow!("onnxruntime error: {}:{}", get_error_code(status), get_error_msg(status));
        release_status(status);
        Err(err)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ort_status() {
        let status = create_status(OrtErrorCode_ORT_MODEL_LOADED, "OKOKOKO".to_string());
        assert_eq!(9, get_error_code(status));
        release_status(status as *mut OrtStatus);
    }
}
