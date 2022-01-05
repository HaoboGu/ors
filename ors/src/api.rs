use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use ors_sys::*;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::null_mut;
use std::sync::{
    atomic::{AtomicBool, AtomicPtr, Ordering},
    Arc, Mutex,
};
use tracing::debug;

lazy_static! {
    static ref API: Arc<Mutex<AtomicPtr<OrtApi>>> =
        Arc::new(Mutex::new(AtomicPtr::new(null_mut())));
    static ref LIB: Arc<Mutex<AtomicPtr<ManuallyDrop<onnxruntime>>>> =
        Arc::new(Mutex::new(AtomicPtr::new(null_mut())));
    static ref INITIALIZED: AtomicBool = AtomicBool::new(false);
}

/// Initialize onnxruntime from a shared lib
/// This function MUST be called before accessing any APIs of onnxruntime
/// Note: Only the fist call of `initialize_runtime` makes sense
/// ## Example
/// ```rust
/// initialize_runtime(Path::new("/path/to/onnxruntime")).expect("Failed to load onnxruntime");
/// test_macro();
/// ```
pub fn initialize_runtime(path: &Path) -> Result<()> {
    // If the runtime has been initialized, just return
    let initialized = INITIALIZED.load(Ordering::SeqCst);
    if initialized {
        debug!("onnxruntime has been initialized");
        return Ok(());
    }

    // Otherwise, load onnxruntime shared library
    let mut ort = match unsafe { onnxruntime::new(path) } {
        Ok(ort) => ManuallyDrop::new(ort),
        Err(err) => return Err(anyhow!("Failed to load onnxruntime shared library")),
    };

    // Wrap the lib using ManuallyDrop
    // The loaded lib is dropped only when we call drop manually
    let mut lib_ptr = LIB.lock().expect("Failed to get lib");
    *lib_ptr = AtomicPtr::new(&mut ort);

    // Set the api entry then
    let api_base = unsafe { ort.OrtGetApiBase() };
    let api: *const OrtApi = unsafe { (&(*api_base).GetApi.unwrap())(ORT_API_VERSION) };
    let mut g_api = API.lock().expect("Failed to get api");
    *g_api = AtomicPtr::new(api as *mut OrtApi);
    let _b = INITIALIZED
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(true))
        .unwrap();
    Ok(())
}

pub fn drop_lib() -> Result<()> {
    // If the runtime has not been initialized, just return
    let initialized = INITIALIZED.load(Ordering::SeqCst);
    if !initialized {
        return Ok(());
    }

    // Set LIB to null ptr and drop the loaded lib
    let lib_atomic_ptr = LIB.lock().expect("Failed to get lib");
    let lib_raw_ptr = lib_atomic_ptr
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |m| Some(null_mut()))
        .map_err(|e| anyhow!("Failed to get library ptr"))?;
    unsafe { lib_raw_ptr.drop_in_place() };

    // Set initialized to false
    let _b = INITIALIZED
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(false))
        .unwrap();
    Ok(())
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
    let initialized = INITIALIZED.load(Ordering::SeqCst);
    // If the library is not initialize, just panic
    if !initialized {
        panic!("The library has not been initialized, you should initialize it first using initialize_runtime()");
    }
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
        setup_runtime();
        test_macro();
    }

    #[test]
    fn test_initialize_twice() {
        setup_runtime();
        setup_runtime();
        test_macro();
    }

    #[test]
    #[traced_test]
    #[should_panic(
        expected = "The library has not been initialized, you should initialize it first using initialize_runtime()"
    )]
    fn test_drop_loaded_lib() {
        setup_runtime();
        test_macro();
        drop_lib().unwrap();
        test_macro();
    }

    #[test]
    #[should_panic]
    fn test_macro() {
        let msg = CString::new("error msg").unwrap();
        let status = call_ort!(CreateStatus, OrtErrorCode_ORT_ENGINE_ERROR, msg.as_ptr());
        let error_code = call_ort!(GetErrorCode, status);
        assert_eq!(error_code, 5);
    }

    fn setup_runtime() {
        #[cfg(target_os = "windows")]
        let path = "D:\\Projects\\Rust\\ors\\onnxruntime.dll";
        #[cfg(target_os = "macos")]
        let path = "/usr/local/lib/libonnxruntime.1.8.1.dylib";
        #[cfg(target_os = "linux")]
        let path = "/usr/local/lib/libonnxruntime.so";
        initialize_runtime(Path::new(path)).expect("Failed to initialize runtime");
    }
}
