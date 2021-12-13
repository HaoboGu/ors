use ors_sys::*;
use std::{ffi::CString, ptr::null_mut};

use crate::api::get_api;
use crate::call_ort;
use crate::status::check_status;

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

    let status = call_ort!(
        CreateSession,
        env,
        c_model_path.as_ptr(),
        options,
        &mut session_ptr
    );
    check_status(status).unwrap();
    return session_ptr;
}

#[cfg(test)]
mod test {
    use tracing_test::traced_test;

    use super::*;
    use crate::env::get_env_ptr;

    #[test]
    #[traced_test]
    fn test_create_session() {
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        // let path = "/Users/haobogu/Projects/rust/cosy-local-tools/model/model.onnx";
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        let session = create_session(get_env_ptr() as *const OrtEnv, path, null_mut());
        assert_ne!(session, null_mut());
    }
}
