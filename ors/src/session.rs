use self::io::{get_session_outputs, SessionInputInfo, SessionOutputInfo};
use crate::api::get_api;
use crate::call_ort;
use crate::config::{SessionExecutionMode, SessionGraphOptimizationLevel};
use crate::env::get_env_ptr;
use crate::session::io::get_session_inputs;
use crate::status::check_status;
use anyhow::{anyhow, Result};
use ors_sys::*;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::{null, null_mut};
#[cfg(not(target_family = "windows"))]
use std::{ffi::OsString, os::unix::prelude::OsStrExt};
#[cfg(target_family = "windows")]
use std::{ffi::OsString, os::windows::prelude::OsStrExt};

pub(crate) mod io;

#[derive(Debug)]
pub struct Session {
    session_ptr: *mut OrtSession,
    allocator: *mut OrtAllocator,
    mem_info: *mut OrtMemoryInfo,
    input_info: Vec<SessionInputInfo>,
    output_info: Vec<SessionOutputInfo>,
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
        let mut session_ptr: *mut OrtSession = null_mut();
        if !filepath.exists() {
            return Err(anyhow!(
                "Model doesn't exist at {}",
                filepath.to_string_lossy()
            ));
        }

        // Build an OsString than a vector of bytes to pass to C
        let model_path = OsString::from(filepath);
        #[cfg(target_family = "windows")]
        let model_path: Vec<u16> = model_path
            .encode_wide()
            .chain(std::iter::once(0)) // Make sure we have a null terminated string
            .collect();
        #[cfg(not(target_family = "windows"))]
        let model_path: Vec<std::os::raw::c_char> = model_path
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
            .map(|b| *b as std::os::raw::c_char)
            .collect();

        let status = call_ort!(
            CreateSession,
            get_env_ptr(),
            model_path.as_ptr(),
            self.session_options_ptr,
            &mut session_ptr
        );
        check_status(status)?;

        let allocator = get_default_allocator()?;
        let mem_info = get_allocator_mem_info(allocator)?;

        let input_info = get_session_inputs(session_ptr, allocator)?;
        let output_info = get_session_outputs(session_ptr, allocator)?;
        Ok(Session {
            session_ptr,
            allocator,
            mem_info,
            input_info,
            output_info,
        })
    }

    pub fn build_with_model_in_momory(self, model_bytes: &[u8]) -> Result<Session> {
        let mut session_ptr: *mut OrtSession = null_mut();

        let model = model_bytes.as_ptr() as *const c_void;
        let model_length = model_bytes.len();
        let status = call_ort!(
            CreateSessionFromArray,
            get_env_ptr(),
            model,
            model_length,
            self.session_options_ptr,
            &mut session_ptr
        );
        check_status(status)?;

        let allocator = get_default_allocator()?;
        let mem_info = get_allocator_mem_info(allocator)?;

        let input_info = get_session_inputs(session_ptr, allocator)?;
        let output_info = get_session_outputs(session_ptr, allocator)?;
        Ok(Session {
            session_ptr,
            allocator,
            mem_info,
            input_info,
            output_info,
        })
    }

    /// Configure the session to use a number of threads
    pub fn intra_number_threads(self, num_threads: i32) -> Result<SessionBuilder> {
        let status = call_ort!(SetIntraOpNumThreads, self.session_options_ptr, num_threads);
        check_status(status)?;
        Ok(self)
    }

    pub fn inter_number_threads(self, num_threads: i32) -> Result<SessionBuilder> {
        let status = call_ort!(SetInterOpNumThreads, self.session_options_ptr, num_threads);
        check_status(status)?;
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn graph_optimization_level(
        self,
        opt_level: SessionGraphOptimizationLevel,
    ) -> Result<SessionBuilder> {
        // Sets graph optimization level
        let status = call_ort!(
            SetSessionGraphOptimizationLevel,
            self.session_options_ptr,
            opt_level.into()
        );
        check_status(status)?;
        Ok(self)
    }

    /// Set execution mode.
    ///
    /// Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance. See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
    pub fn execution_mode(self, execution_mode: SessionExecutionMode) -> Result<SessionBuilder> {
        let status = call_ort!(
            SetSessionExecutionMode,
            self.session_options_ptr,
            execution_mode.into()
        );
        check_status(status)?;
        Ok(self)
    }

    /// Enable the memory arena on CPU
    ///
    /// Arena may pre-allocate memory for future usage
    pub fn cpu_mem_arena_enabled(self, cpu_mem_arena_enabled: bool) -> Result<SessionBuilder> {
        if cpu_mem_arena_enabled {
            let status = call_ort!(EnableCpuMemArena, self.session_options_ptr);
            check_status(status)?;
        }
        Ok(self)
    }

    /// Enable the memory pattern optimization
    ///
    /// The idea is if the input shapes are the same, we could trace the internal memory allocation and generate a memory pattern for future request. So next time we could just do one allocation with a big chunk for all the internal memory allocation
    ///
    /// Note: Memory pattern optimization is only available when Sequential Execution mode is enabled
    pub fn mem_pattern_enabled(self, mem_pattern_enabled: bool) -> Result<SessionBuilder> {
        if mem_pattern_enabled {
            let status = call_ort!(EnableMemPattern, self.session_options_ptr);
            check_status(status)?;
        }
        Ok(self)
    }
}

pub(crate) fn get_default_allocator() -> Result<*mut OrtAllocator> {
    let mut allocator_ptr = null_mut();
    let status = call_ort!(GetAllocatorWithDefaultOptions, &mut allocator_ptr);
    check_status(status)?;
    Ok(allocator_ptr)
}

pub(crate) fn get_default_memory_info() -> Result<*mut OrtMemoryInfo> {
    let allocator = get_default_allocator()?;
    return get_allocator_mem_info(allocator);
}

pub(crate) fn get_allocator_mem_info(allocator: *const OrtAllocator) -> Result<*mut OrtMemoryInfo> {
    let mut mem_info_ptr = null();
    let status = unsafe { get_api().AllocatorGetInfo.unwrap()(allocator, &mut mem_info_ptr) };
    check_status(status)?;
    Ok(mem_info_ptr as *mut OrtMemoryInfo)
}

#[cfg(test)]
mod test {
    use super::*;
    use tracing_test::traced_test;

    fn get_path() -> &'static str {
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        return path;
    }

    #[test]
    #[traced_test]
    fn test_create_session() {
        let session_builder = SessionBuilder::new().unwrap();
        let session = session_builder
            .intra_number_threads(4)
            .unwrap()
            .graph_optimization_level(SessionGraphOptimizationLevel::All)
            .unwrap()
            .execution_mode(SessionExecutionMode::Parallel)
            .unwrap()
            .cpu_mem_arena_enabled(true)
            .unwrap()
            .mem_pattern_enabled(true)
            .unwrap()
            .build_with_model_from_file(get_path())
            .unwrap();

        println!("{:#?}", session);
        assert_ne!(session.session_ptr, null_mut());
    }

    #[test]
    #[traced_test]
    fn test_session_drop() {
        {
            let session_builder = SessionBuilder::new().unwrap();
            let session = session_builder
                .build_with_model_from_file(get_path())
                .unwrap();
            assert_ne!(session.session_ptr, null_mut());
        }
        let session_builder2 = SessionBuilder::new().unwrap();
        let session2 = session_builder2
            .build_with_model_from_file(get_path())
            .unwrap();
        assert_ne!(session2.session_ptr, null_mut());
    }
}
