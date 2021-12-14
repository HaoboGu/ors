use anyhow::Result;
use ors_sys::*;
#[cfg(not(target_family = "windows"))]
use std::ffi::CString;
#[cfg(target_family = "windows")]
use std::{ffi::OsStr, os::windows::prelude::OsStrExt};
use std::{path::Path, ptr::null_mut};

use crate::api::get_api;
use crate::status::check_status;
use crate::{call_ort, OptimizationLevel};

pub struct Session {
    session_ptr: *mut OrtSession,
}

pub struct SessionBuilder {
    session_options_ptr: *mut OrtSessionOptions,
}

impl SessionBuilder {
    pub fn new() -> Result<Self> {
        let allocator_type = OrtAllocatorType_OrtArenaAllocator;
        let mem_type = OrtMemType_OrtMemTypeDefault;
        let mut session_options_ptr: *mut OrtSessionOptions = null_mut();
        let status = call_ort!(CreateSessionOptions, &mut session_options_ptr);
        check_status(status)?;

        Ok(SessionBuilder {
            session_options_ptr,
        })
    }

    pub fn build_with_model_from_file<P>(self, model_filepath: P) -> Result<Session>
    where
        P: AsRef<Path>,
    {
        let filepath = model_filepath.as_ref();
        todo!();
    }

    pub fn build_with_model_in_momory(self, model_bytes: &[u8]) -> Result<Session> {
        todo!();
    }

    /// Configure the session to use a number of threads
    pub fn intra_number_threads(self, num_threads: i32) -> Result<SessionBuilder> {
        let status = call_ort!(SetIntraOpNumThreads, self.session_options_ptr, num_threads);
        check_status(status)?;
        Ok(self)
    }

    pub fn inter_number_threads(self, num_threads: i32) -> Result<SessionBuilder> {
        todo!();
    }

    /// Set the session's optimization level
    pub fn graph_optimization_level(self, opt_level: OptimizationLevel) -> Result<SessionBuilder> {
        // Sets graph optimization level
        let status = call_ort!(
            SetSessionGraphOptimizationLevel,
            self.session_options_ptr,
            opt_level.into()
        );
        check_status(status)?;
        Ok(self)
    }

    /// Set execution mode
    /// TODO: Wrap ExecutionMode
    pub fn execution_mode(self, execution_mode: i32) -> Result<SessionBuilder> {
        todo!();
    }

    pub fn cpu_mem_arena_enabled(self, cpu_mem_arena_enabled: bool) -> Result<SessionBuilder> {
        todo!();
    }

    pub fn mem_pattern_enabled(self, mem_pattern_enabled: bool) -> Result<SessionBuilder> {
        todo!();
    }
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
