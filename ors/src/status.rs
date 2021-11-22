use crate::api::get_api;
use ors_sys::*;
use std::ffi::CString;

fn create_status(error_code: OrtErrorCode, msg: String) -> *const OrtStatus {
    let msg = CString::new(msg).unwrap();
    return unsafe { get_api().CreateStatus.unwrap()(error_code, msg.as_ptr()) };
}

fn get_error_code(status: *const OrtStatus) -> OrtErrorCode {
    return unsafe { get_api().GetErrorCode.unwrap()(status) };
}

fn release_status(status: *mut OrtStatus) {
    unsafe { get_api().ReleaseStatus.unwrap()(status) }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ort_status() {
        let status = create_status(OrtErrorCode_ORT_MODEL_LOADED, "OKOKOKO".to_string());
        println!("error code: {}", get_error_code(status));
        release_status(status as *mut OrtStatus);
    }
}
