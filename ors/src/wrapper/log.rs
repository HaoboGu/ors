use ors_sys::*;

// Logging level of the ONNX Runtime C API
// Borrowed from: https://github.com/nbigaouette/onnxruntime-rs/blob/master/onnxruntime/src/lib.rs
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum LoggingLevel {
    /// Verbose log level
    Verbose = OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE as OnnxEnumInt,
    /// Info log level
    Info = OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO as OnnxEnumInt,
    /// Warning log level
    Warning = OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING as OnnxEnumInt,
    /// Error log level
    Error = OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR as OnnxEnumInt,
    /// Fatal log level
    Fatal = OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL as OnnxEnumInt,
}

impl From<LoggingLevel> for OrtLoggingLevel {
    fn from(val: LoggingLevel) -> Self {
        match val {
            LoggingLevel::Verbose => OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
            LoggingLevel::Info => OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
            LoggingLevel::Warning => OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
            LoggingLevel::Error => OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
            LoggingLevel::Fatal => OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
        }
    }
}
