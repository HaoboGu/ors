use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use ors_sys::*;
use std::path::Path;
use std::ptr::null_mut;
use std::sync::{atomic::AtomicPtr, Arc, Mutex};

// The instance of onnxruntime api
#[cfg(not(feature = "runtime-linking"))]
lazy_static! {
    static ref API: Arc<Mutex<AtomicPtr<OrtApi>>> = {
        let api_base = unsafe { OrtGetApiBase() };

        assert_ne!(api_base, std::ptr::null());

        let api = unsafe { (&(*api_base).GetApi.unwrap())(ORT_API_VERSION) };

        Arc::new(Mutex::new(AtomicPtr::new(api as *mut OrtApi)))
    };
}

#[cfg(feature = "runtime-linking")]
lazy_static! {
    static ref API: Arc<Mutex<AtomicPtr<OrtApi>>> =
        Arc::new(Mutex::new(AtomicPtr::new(null_mut())));
    static ref LIB: Arc<Mutex<AtomicPtr<onnxruntime>>> =
        Arc::new(Mutex::new(AtomicPtr::new(null_mut())));
}
#[cfg(feature = "runtime-linking")]
pub fn load_runtime(path: &Path) -> Result<onnxruntime> {
    // TODO: don't drop loaded lib
    let ort = match unsafe { onnxruntime::new(path) } {
        Ok(ort) => ort,
        Err(err) => return Err(anyhow!("Failed to load onnxruntime shared library")),
    };
    let base: *const OrtApiBase = unsafe { ort.OrtGetApiBase() };
    assert_ne!(base, std::ptr::null());
    let api: *const OrtApi = unsafe { (&(*base).GetApi.unwrap())(ORT_API_VERSION) };
    let mut g_api = API.lock().expect("Failed to get api");
    *g_api = AtomicPtr::new(api as *mut OrtApi);
    Ok(ort)
}

/// Macro for calling unsafe methods using get_api()
/// You can also use this macro to call onnxruntime api
/// ## Arguments
///
/// The first argument is the API name, and the others are parameters
///
/// ## Example
/// Call OrtApi::CreateStatus
/// ```
/// let msg = CString::new("error msg").unwrap();
/// let status = call_ort!(CreateStatus, OrtErrorCode_ORT_ENGINE_ERROR, msg.as_ptr());
/// ```
#[macro_export]
macro_rules! call_ort {
    ($api_name:ident, $($parameter:expr),*) => {
        unsafe {
            get_api().$api_name.unwrap()($($parameter),*)
        }
    };
}

/// get_api exposes an OrtApi instance in crate
pub(crate) fn get_api() -> OrtApi {
    let p = *(API.try_lock().unwrap()).get_mut();
    unsafe { *p }
}

#[cfg(test)]
mod test {
    use std::ffi::CString;

    use tracing_test::traced_test;

    use super::*;

    #[test]
    #[traced_test]
    fn test_get_dynamic_load() {
        let ort = load_runtime(Path::new("D:\\Projects\\Rust\\ors\\onnxruntime.dll"))
            .expect("Failed to load onnxruntime");
        test_macro();
    }

    #[test]
    fn test_macro() {
        let msg = CString::new("error msg").unwrap();
        let status = call_ort!(CreateStatus, OrtErrorCode_ORT_ENGINE_ERROR, msg.as_ptr());
        let error_code = call_ort!(GetErrorCode, status);
        assert_eq!(error_code, 5);
    }
}
