use std::ptr::{null, null_mut};

#[cfg(target_family = "windows")]
use std::{
    ffi::{CString, OsStr},
    os::windows::prelude::OsStrExt,
};

#[cfg(not(target_family = "windows"))]
use std::ffi::CString;

use ors_sys::*;

use crate::{
    api::get_api,
    session_io::{SessionInputInfo, SessionOutputInfo},
    status::assert_status,
};

// TODO: run with io binding
// Run session inference
fn session_run(
    session: *mut OrtSession,
    run_options: *const OrtRunOptions,
    inputs: Vec<*mut OrtValue>,
    mut outputs: Vec<*mut OrtValue>,
    input_info: Vec<SessionInputInfo>,
    output_info: Vec<SessionOutputInfo>,
) {
    let input_names_ptr: Vec<*const i8> = input_info
        .iter()
        .map(|n| CString::new(n.name.clone()).unwrap())
        .map(|n| n.into_raw() as *const i8)
        .collect();

    let output_names_cstring: Vec<CString> = output_info
        .iter()
        .map(|n| CString::new(n.name.clone()).unwrap())
        .collect();

    let output_names_ptr: Vec<*const i8> = output_names_cstring
        .iter()
        .map(|n| n.as_ptr() as *const i8)
        .collect();

    let inputs_ptr: Vec<*const OrtValue> = inputs.iter().map(|i| (*i) as *const OrtValue).collect();

    let status = unsafe {
        get_api().Run.unwrap()(
            session,
            null(),
            input_names_ptr.as_ptr(),
            inputs_ptr.as_ptr(),
            inputs.len(),
            output_names_ptr.as_ptr(),
            output_names_ptr.len(),
            outputs.as_mut_ptr(),
        )
    };
    assert_status(status);
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

fn create_and_register_allocator(
    env: *mut OrtEnv,
    mem_info: *const OrtMemoryInfo,
) -> *mut OrtAllocator {
    let allocator_ptr = null_mut();
    let arena_cfg = null();
    let status = unsafe { get_api().CreateAndRegisterAllocator.unwrap()(env, mem_info, arena_cfg) };
    assert_status(status);
    return allocator_ptr;
}

fn get_allocator_mem_info(allocator: *const OrtAllocator) -> *const OrtMemoryInfo {
    let mut mem_info_ptr = null();
    let status = unsafe { get_api().AllocatorGetInfo.unwrap()(allocator, &mut mem_info_ptr) };
    assert_status(status);
    return mem_info_ptr;
}

// fn get_output_name(se)

#[cfg(test)]
mod test {
    use std::time::SystemTime;

    use ndarray::{s, ArrayD, IxDyn};

    use super::*;
    use crate::{
        env::create_env,
        log::LoggingLevel,
        session_io::{get_input_count, get_output_count, get_session_input, get_session_output},
        tensor::create_tensor_with_ndarray,
    };

    #[test]
    fn test_session() {
        let start = SystemTime::now();
        let env = create_env(LoggingLevel::Warning, "log_name".to_string());
        #[cfg(target_family = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\gpt2.onnx";
        #[cfg(not(target_family = "windows"))]
        // let path = "/Users/haobogu/Projects/rust/cosy-local-tools/model/model.onnx";
        let path = "/Users/haobogu/Projects/rust/ors/ors/sample/gpt2.onnx";
        let session = create_session(env as *const OrtEnv, path, std::ptr::null());
        let allocator = get_default_allocator();
        println!("init costs: {:?}", SystemTime::now().duration_since(start));
        let input_cnt = get_input_count(session);
        let output_cnt = get_output_count(session);
        println!("input cnt: {}, output cnt: {}", input_cnt, output_cnt);

        // Create gpt2 inputs:
        // gpt2 model sample inputs:
        // input_ids: int64, [[50256, 50256, 50256, 50256, 13466,  7541,   287, 15489,  1989], [ 1456,   318,   281,  1672,   286,   308,   457,    17,  2746]]
        // shape: [2, 9]
        // attention_mask: float32, [[0., 0., 0., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        // position_ids: int64, [[0, 0, 0, 0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        // 'past_0': array([], shape=(2, 2, 12, 0, 64), dtype=float32),
        // 'past_1': array([], shape=(2, 2, 12, 0, 64), dtype=float32), ...
        let mem_info = get_allocator_mem_info(get_default_allocator()) as *mut OrtMemoryInfo;
        let inference_1_start = SystemTime::now();
        let input_ids_data: Vec<i64> = vec![
            50256, 50256, 50256, 50256, 13466, 7541, 287, 15489, 1989, 1456, 318, 281, 1672, 286,
            308, 457, 17, 2746,
        ];
        let mut input: Vec<*mut OrtValue> = vec![];
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
        let input_ids_tensor =
            create_tensor_with_ndarray::<i64>(mem_info as *mut OrtMemoryInfo, input_ids.view_mut());
        let position_ids_tensor = create_tensor_with_ndarray::<i64>(
            mem_info as *mut OrtMemoryInfo,
            positions_ids.view_mut(),
        );
        let attension_mask_tensor = create_tensor_with_ndarray::<f32>(
            mem_info as *mut OrtMemoryInfo,
            attension_mask.view_mut(),
        );
        input.push(input_ids_tensor);
        input.push(position_ids_tensor);
        input.push(attension_mask_tensor);
        let past_num = 12;
        for i in 0..past_num {
            let mut past =
                ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 2, 12, 0, 64]), vec![]).unwrap();
            let key = format!("past_{}", i);
            input.push(create_tensor_with_ndarray::<f32>(
                mem_info as *mut OrtMemoryInfo,
                past.view_mut(),
            ));
        }
        // created input is a map

        // Create gpt2 outputs
        // gpt2 model output dims:
        // {'logits': [2, 9, 50257],
        // 'present_0': [2, 2, 12, 9, 64],
        // 'present_1': [2, 2, 12, 9, 64], ...
        let mut presents = vec![];
        let mut output_tensors = vec![];
        let mut logits =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 9, 50257]), vec![0.0; 2 * 9 * 50257]).unwrap();
        let logits_tensor = create_tensor_with_ndarray::<f32>(
            mem_info,
            logits.slice_mut(s![.., 0..9, ..]).into_dyn(),
        );
        output_tensors.push(logits_tensor);
        for i in 0..past_num {
            let key = format!("present_{}", i);
            let mut present = ArrayD::<f32>::from_shape_vec(
                IxDyn(&[2, 2, 12, 9, 64]),
                vec![0.0; 2 * 2 * 12 * 9 * 64],
            )
            .unwrap();
            let present_tensor = create_tensor_with_ndarray::<f32>(mem_info, present.view_mut());
            output_tensors.push(present_tensor);
            presents.push(present);
        }

        let mut inputs_info = vec![];
        for i in 0..input_cnt {
            let input_info = get_session_input(session, i, allocator);
            println!("input info: {:?}", input_info);
            inputs_info.push(input_info);
        }
        let mut outputs_info = vec![];
        for i in 0..output_cnt {
            let output_info = get_session_output(session, i, allocator);
            println!("output info: {:?}", output_info);
            outputs_info.push(output_info);
        }

        session_run(
            session,
            null(),
            input,
            output_tensors,
            inputs_info.clone(),
            outputs_info.clone(),
        );

        // println!("the first inference result: logits: {:?}", logits);
        // println!("firset presents: {:?}", presents.get(0).unwrap());
        println!(
            "the first inference costs: {:?}",
            SystemTime::now().duration_since(inference_1_start)
        );

        // The second inference
        let inference_2_start = SystemTime::now();
        let mut new_input: Vec<*mut OrtValue> = vec![];
        let mut input_ids = ArrayD::<i64>::from_shape_vec(IxDyn(&[2, 1]), vec![1234, 568]).unwrap();
        let mut positions_ids = ArrayD::<i64>::from_shape_vec(IxDyn(&[2, 1]), vec![5, 9]).unwrap();
        let mut attension_mask =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 1]), vec![1., 1.]).unwrap();
        let input_ids_tensor =
            create_tensor_with_ndarray::<i64>(mem_info as *mut OrtMemoryInfo, input_ids.view_mut());
        let position_ids_tensor = create_tensor_with_ndarray::<i64>(
            mem_info as *mut OrtMemoryInfo,
            positions_ids.view_mut(),
        );
        let attension_mask_tensor = create_tensor_with_ndarray::<f32>(
            mem_info as *mut OrtMemoryInfo,
            attension_mask.view_mut(),
        );
        new_input.push(input_ids_tensor);
        new_input.push(position_ids_tensor);
        new_input.push(attension_mask_tensor);
        for mut past in presents {
            new_input.push(create_tensor_with_ndarray(mem_info, past.view_mut()));
        }

        let mut new_output_tensors = vec![];
        let mut new_logits =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 1, 50257]), vec![0.0; 2 * 1 * 50257]).unwrap();
        let logits_tensor = create_tensor_with_ndarray::<f32>(mem_info, new_logits.view_mut());
        new_output_tensors.push(logits_tensor);
        let mut new_presents = vec![];
        for i in 0..past_num {
            let key = format!("present_{}", i);
            let mut present = ArrayD::<f32>::from_shape_vec(
                IxDyn(&[2, 2, 12, 10, 64]),
                vec![0.0; 2 * 2 * 12 * 10 * 64],
            )
            .unwrap();
            let present_tensor = create_tensor_with_ndarray::<f32>(mem_info, present.view_mut());
            new_output_tensors.push(present_tensor);
            new_presents.push(present);
        }

        session_run(
            session,
            null(),
            new_input,
            new_output_tensors,
            inputs_info.clone(),
            outputs_info.clone(),
        );

        // println!("the second result logits: {:?}", new_logits);
        // println!("second presents: {:?}", new_presents.get(0).unwrap());
        println!(
            "second inference costs: {:?}",
            SystemTime::now().duration_since(inference_2_start)
        );
        println!(
            "total inference costs: {:?}",
            SystemTime::now().duration_since(inference_1_start)
        );
    }
}
