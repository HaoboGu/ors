use self::io::{get_session_outputs, SessionInputInfo, SessionOutputInfo};
use crate::api::get_api;
use crate::call_ort;
use crate::config::{SessionExecutionMode, SessionGraphOptimizationLevel};
use crate::env::get_env_ptr;
use crate::session::io::get_session_inputs;
use crate::status::check_status;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use ors_sys::*;
use std::ffi::{c_void, CString};
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

pub fn run(session: &mut Session, inputs: &Vec<Tensor>, outputs: &mut Vec<Tensor>) -> Result<()> {
    let input_names_ptr: Vec<*const i8> = session
        .input_info
        .iter()
        .map(|n| CString::new(n.name.clone()).unwrap())
        .map(|n| n.into_raw() as *const i8)
        .collect();

    let output_names_cstring: Vec<CString> = session
        .output_info
        .iter()
        .map(|n| CString::new(n.name.clone()).unwrap())
        .collect();

    let output_names_ptr: Vec<*const i8> = output_names_cstring
        .iter()
        .map(|n| n.as_ptr() as *const i8)
        .collect();

    let inputs_ptr: Vec<*const OrtValue> =
        inputs.iter().map(|i| (i.ptr) as *const OrtValue).collect();
    let mut outputs_ptr: Vec<*mut OrtValue> = outputs.iter().map(|o| o.ptr).collect();

    let status = unsafe {
        get_api().Run.unwrap()(
            session.session_ptr,
            null(),
            input_names_ptr.as_ptr(),
            inputs_ptr.as_ptr(),
            inputs.len(),
            output_names_ptr.as_ptr(),
            output_names_ptr.len(),
            outputs_ptr.as_mut_ptr(),
        )
    };
    check_status(status)?;
    Ok(())
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
    use std::time::SystemTime;

    use super::*;
    use crate::tensor::create_tensor_with_ndarray;
    use ndarray::{ArrayD, IxDyn};
    use tracing_test::traced_test;

    #[test]
    #[traced_test]
    fn test_session_run() {
        let session_builder = SessionBuilder::new().unwrap();
        let mut session = session_builder
            .graph_optimization_level(SessionGraphOptimizationLevel::All)
            .unwrap()
            .build_with_model_from_file(get_path())
            .unwrap();

        // Model input
        let mut inputs: Vec<Tensor> = vec![];
        // Input data
        let input_ids_data: Vec<i64> = vec![
            50256, 50256, 50256, 50256, 13466, 7541, 287, 15489, 1989, 1456, 318, 281, 1672, 286,
            308, 457, 17, 2746,
        ];
        let mut input_ids = ArrayD::<i64>::from_shape_vec(IxDyn(&[2, 9]), input_ids_data).unwrap();
        let mut positions_ids = ArrayD::<i64>::from_shape_vec(
            IxDyn(&[2, 9]),
            vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        )
        .unwrap();
        let mut attension_mask = ArrayD::<f32>::from_shape_vec(
            IxDyn(&[2, 9]),
            vec![
                0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        )
        .unwrap();
        // Create tensors
        let input_ids_tensor = create_tensor_with_ndarray::<i64>(input_ids.view_mut()).unwrap();
        let position_ids_tensor =
            create_tensor_with_ndarray::<i64>(positions_ids.view_mut()).unwrap();
        let attention_mask_tensor =
            create_tensor_with_ndarray::<f32>(attension_mask.view_mut()).unwrap();
        inputs.push(input_ids_tensor);
        inputs.push(position_ids_tensor);
        inputs.push(attention_mask_tensor);
        // Initialize pasts
        let past_num = 12;
        for i in 0..past_num {
            let mut past =
                ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 2, 12, 0, 64]), vec![]).unwrap();
            inputs.push(create_tensor_with_ndarray::<f32>(past.view_mut()).unwrap());
        }

        let mut outputs: Vec<Tensor> = vec![];
        let mut presents = vec![];
        let mut logits =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 9, 50257]), vec![0.0; 2 * 9 * 50257]).unwrap();
        let logits_tensor = create_tensor_with_ndarray::<f32>(logits.view_mut()).unwrap();
        outputs.push(logits_tensor);
        let past_num = 12;
        for i in 0..past_num {
            let key = format!("present_{}", i);
            let mut present = ArrayD::<f32>::from_shape_vec(
                IxDyn(&[2, 2, 12, 9, 64]),
                vec![0.0; 2 * 2 * 12 * 9 * 64],
            )
            .unwrap();
            let present_tensor = create_tensor_with_ndarray::<f32>(present.view_mut()).unwrap();
            outputs.push(present_tensor);
            presents.push(present);
        }
        let inference_1_start = SystemTime::now();
        run(&mut session, &inputs, &mut outputs).unwrap();

        println!("inference result: logits: {:?}", logits);
        println!(
            "inference costs: {:?}",
            SystemTime::now().duration_since(inference_1_start)
        );
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

    fn get_path() -> &'static str {
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        return path;
    }
}
