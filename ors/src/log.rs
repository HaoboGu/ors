use std::ffi::CStr;

use ors_sys::*;
use tracing::{debug, error, info, span, trace, warn, Level};

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

/// Runtime's logging sends the code location where the log happened, will be parsed to this struct.
#[derive(Debug)]
struct CodeLocation<'a> {
    file: &'a str,
    line_number: &'a str,
    function: &'a str,
}

impl<'a> From<&'a str> for CodeLocation<'a> {
    fn from(code_location: &'a str) -> Self {
        let mut splitter = code_location.split(' ');
        let file_and_line_number = splitter.next().unwrap_or("<unknown file:line>");
        let function = splitter.next().unwrap_or("<unknown module>");
        let mut file_and_line_number_splitter = file_and_line_number.split(':');
        let file = file_and_line_number_splitter
            .next()
            .unwrap_or("<unknown file>");
        let line_number = file_and_line_number_splitter
            .next()
            .unwrap_or("<unknown line number>");

        CodeLocation {
            file,
            line_number,
            function,
        }
    }
}

/// Callback from C that will handle the logging, forwarding the runtime's logs to the tracing crate.
pub(crate) extern "C" fn custom_logger(
    _params: *mut std::ffi::c_void,
    severity: OrtLoggingLevel,
    category: *const i8,
    logid: *const i8,
    code_location: *const i8,
    message: *const i8,
) {
    let log_level = match severity {
        ors_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
        ors_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
        ors_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING => Level::INFO,
        ors_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR => Level::WARN,
        ors_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL => Level::ERROR,
        _ => Level::INFO,
    };

    assert_ne!(category, std::ptr::null());
    let category = unsafe { CStr::from_ptr(category) };
    assert_ne!(code_location, std::ptr::null());
    let code_location = unsafe { CStr::from_ptr(code_location) }
        .to_str()
        .unwrap_or("unknown");
    assert_ne!(message, std::ptr::null());
    let message = unsafe { CStr::from_ptr(message) };

    assert_ne!(logid, std::ptr::null());
    let logid = unsafe { CStr::from_ptr(logid) };

    // Parse the code location
    let code_location: CodeLocation = code_location.into();

    let span = span!(
        Level::TRACE,
        "onnxruntime",
        category = category.to_str().unwrap_or("<unknown>"),
        file = code_location.file,
        line_number = code_location.line_number,
        function = code_location.function,
        logid = logid.to_str().unwrap_or("<unknown>"),
    );
    let _enter = span.enter();

    match log_level {
        Level::TRACE => trace!("{:?}", message),
        Level::DEBUG => debug!("{:?}", message),
        Level::INFO => info!("{:?}", message),
        Level::WARN => warn!("{:?}", message),
        Level::ERROR => error!("{:?}", message),
    }
}
