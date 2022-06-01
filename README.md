# ors - onnxruntime bindings for rust
This project provides Rust bindings of Microsoft's [onnxruntime](https://github.com/microsoft/onnxruntime), which is a machine learning inference and training framework.

Warning: This project is in very early stage and not finished yet. There are still many bugs as far as I know. Don't use it in production.

## Prerequisites
This crate requires you have onnxruntime's C library version v1.11.1 in your system. You can use `initialize_runtime()` to read the C library:

```rust
use ors::api::initialize_runtime;
use std::path::Path;

fn setup_runtime() {
    #[cfg(target_os = "windows")]
    let path = "/path/to/onnxruntime.dll";
    #[cfg(target_os = "macos")]
    let path = "/path/to/libonnxruntime.1.11.1.dylib";
    #[cfg(target_os = "linux")]
    let path = "/path/to/libonnxruntime.so";
    initialize_runtime(Path::new(path)).unwrap();
}
```

## Example

First, add this crate to your `cargo.toml`

```toml
ors = "0.0.9"
```

This crate provides `SessionBuilder` which helps you create your inference session. Your don't need to create onnxruntime inference environment, which is handled by this crate:
```rust
use ors::{
  config::SessionGraphOptimizationLevel,
  session::{SessionBuilder, run},
  tensor::{create_tensor_with_ndarray, Tensor},
}

setup_runtime();
let session_builder = SessionBuilder::new().unwrap();

// Create an inference session from a model 
let mut session = session_builder
    .graph_optimization_level(SessionGraphOptimizationLevel::All)
    .unwrap()
    // Model conversion script can be found here: https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb 
    .build_with_model_from_file("./gpt2.onnx")
    .unwrap();
```

Create tensor from `ndarray::ArrayD` and add created tensors to model input:
```rust
// Suppose that input_ids, position_ids and attention_mask are all ndarray::ArrayD
let mut inputs: Vec<Tensor> = vec![];
let input_ids_tensor = create_tensor_with_ndarray::<i64>(input_ids).unwrap();
let position_ids_tensor = create_tensor_with_ndarray::<i64>(positions_ids).unwrap();
let attention_mask_tensor = create_tensor_with_ndarray::<f32>(attension_mask).unwrap();
inputs.push(input_ids_tensor);
inputs.push(position_ids_tensor);
inputs.push(attention_mask_tensor);

// Add other inputs
// ...
```

Do same things for model outputs:
```rust
let mut outputs: Vec<Tensor> = vec![];
// You should specify the output shape when creating the ndarray
let mut logits = ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 9, 50257]), vec![0.0; 2 * 9 * 50257]).unwrap();

// Create tensor from logits and add it to output
let logits_tensor = create_tensor_with_ndarray::<f32>(logits).unwrap();
outputs.push(logits_tensor);

// Add other outputs
// ...
```

Run inference session, the model's output will be wrote to `ndarray::ArrayD` which are used to create output tensors.
```rust
run(&mut session, &inputs, &mut outputs);

// Check the result
println!("inference result: logits: {:?}", outputs[0]);
```

output:
```
inference result: logits: [[[-15.88228, -15.500423, -17.979624, -18.302347, -17.527521, ..., -23.000717, -23.806093, -22.637945, -22.227428, -15.411578],
  ...
  [-89.78022, -89.84351, -94.203995, -95.20875, -96.05158, ..., -101.95325, -103.50048, -101.1202, -98.740845, -90.956375]],

 [[-33.367958, -32.94488, -36.2036, -36.568382, -35.434883, ..., -41.491924, -42.189476, -42.094162, -40.86978, -33.79733],
  ...
  [-101.5143, -101.56593, -103.117065, -105.66759, -104.360954, ..., -104.53616, -107.3546, -109.82067, -110.87442, -101.61766]]], shape=[2, 9, 50257], strides=[452313, 50257, 1], layout=Cc (0x5), dynamic ndim=3
```

## Credits
This project is initially a fork of [onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs). Lots of code is copied from onnxruntime-rs. Thanks nbigaouette for the great work.

## License
This project is licensed under either of

Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
at your option.
