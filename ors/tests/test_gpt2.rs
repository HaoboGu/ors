use std::path::Path;

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ors::{
    api::initialize_runtime,
    config::SessionGraphOptimizationLevel,
    session::{run, SessionBuilder},
    tensor::{create_tensor_with_ndarray, Tensor},
};

// The model file `gpt2.onnx` can be generated using this script: 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb 
#[test]
fn test_gpt2_generation() -> Result<()> {
    initialize_runtime(Path::new("path/to/your/onnxruntime.dll"))?;
    let mut session = SessionBuilder::new()?
        .graph_optimization_level(SessionGraphOptimizationLevel::All)?
        .build_with_model_from_file("path/to/your/opt2.onnx")?;

    // Input data
    let input_ids = ArrayD::<i64>::from_shape_vec(
        IxDyn(&[2, 9]),
        vec![
            50256, 50256, 50256, 50256, 13466, 7541, 287, 15489, 1989, 1456, 318, 281, 1672, 286,
            308, 457, 17, 2746,
        ],
    )?;
    let positions_ids = ArrayD::<i64>::from_shape_vec(
        IxDyn(&[2, 9]),
        vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    )?;
    let attension_mask = ArrayD::<f32>::from_shape_vec(
        IxDyn(&[2, 9]),
        vec![
            0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ],
    )?;

    // Suppose that input_ids, position_ids and attention_mask are all ndarray::ArrayD
    // Create inputs
    let mut inputs: Vec<Tensor> = vec![];
    let input_ids_tensor = create_tensor_with_ndarray::<i64>(input_ids)?;
    let position_ids_tensor = create_tensor_with_ndarray::<i64>(positions_ids)?;
    let attention_mask_tensor = create_tensor_with_ndarray::<f32>(attension_mask)?;
    inputs.push(input_ids_tensor);
    inputs.push(position_ids_tensor);
    inputs.push(attention_mask_tensor);

    // Create outputs: logits & pasts
    let mut outputs: Vec<Tensor> = vec![];
    // Create logits
    let logits = ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 9, 50257]), vec![0.0; 2 * 9 * 50257])?;
    // Create tensor from logits ndarray and add it to output, the inference result will be stored in logits ndarray
    let logits_tensor = create_tensor_with_ndarray::<f32>(logits)?;
    outputs.push(logits_tensor);
    // Create presents
    let present_num = 12;
    for _i in 0..present_num {
        let past = ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 2, 12, 0, 64]), vec![])?;
        inputs.push(create_tensor_with_ndarray::<f32>(past)?);
    }
    let past_num = 12;
    for _i in 0..past_num {
        let present = ArrayD::<f32>::from_shape_vec(
            IxDyn(&[2, 2, 12, 9, 64]),
            vec![0.0; 2 * 2 * 12 * 9 * 64],
        )?;
        let present_tensor = create_tensor_with_ndarray::<f32>(present)?;
        outputs.push(present_tensor);
    }

    // Run inference
    run(&mut session, &inputs, &mut outputs)?;

    // Check the result
    println!("inference result: logits: {:?}", outputs[0]);

    Ok(())
}
