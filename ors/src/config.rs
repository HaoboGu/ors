use ors_sys::*;

/// Optimization level performed by ONNX Runtime of the loaded graph
///
/// See the [official documentation](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md)
/// for more information on the different optimization levels.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum SessionGraphOptimizationLevel {
    /// Disable optimization
    DisableAll = GraphOptimizationLevel_ORT_DISABLE_ALL,
    /// Basic optimization
    Basic = GraphOptimizationLevel_ORT_ENABLE_BASIC,
    /// Extended optimization
    Extended = GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    /// Add optimization
    All = GraphOptimizationLevel_ORT_ENABLE_ALL,
}

impl From<SessionGraphOptimizationLevel> for GraphOptimizationLevel {
    fn from(val: SessionGraphOptimizationLevel) -> Self {
        match val {
            SessionGraphOptimizationLevel::DisableAll => GraphOptimizationLevel_ORT_DISABLE_ALL,
            SessionGraphOptimizationLevel::Basic => GraphOptimizationLevel_ORT_ENABLE_BASIC,
            SessionGraphOptimizationLevel::Extended => GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
            SessionGraphOptimizationLevel::All => GraphOptimizationLevel_ORT_ENABLE_ALL,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum SessionExecutionMode {
    Sequential = ExecutionMode_ORT_SEQUENTIAL,
    Parallel = ExecutionMode_ORT_PARALLEL,
}

impl From<SessionExecutionMode> for ExecutionMode {
    fn from(mode: SessionExecutionMode) -> Self {
        match mode {
            SessionExecutionMode::Parallel => ExecutionMode_ORT_PARALLEL,
            SessionExecutionMode::Sequential => ExecutionMode_ORT_SEQUENTIAL,
        }
    }
}
