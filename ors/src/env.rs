use crate::{api::get_api, log::LoggingLevel, status::assert_status};
use ors_sys::{OrtEnv, OrtLoggingLevel};
use std::ffi::CString;

fn create_env(logging_level: LoggingLevel, log_id: String) -> *mut OrtEnv {
    let log_id = CString::new(log_id).unwrap();
    let mut env_ptr = std::ptr::null_mut();
    let status = unsafe {
        get_api().CreateEnv.unwrap()(
            OrtLoggingLevel::from(logging_level),
            log_id.as_ptr(),
            &mut env_ptr,
        )
    };
    assert_status(status);
    return env_ptr;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_env() {
        create_env(LoggingLevel::Verbose, "log_name".to_string());
    }
}
