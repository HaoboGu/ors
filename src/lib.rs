mod env;
mod session;
mod tensor;
mod status;
mod error;
use std::sync::{Arc, Mutex, atomic::AtomicPtr};

use lazy_static::*;
use ors_sys as sys;

// Make functions `extern "C"` for normal targets.
// This behaviors like `extern "system"`.
#[cfg(not(all(target_os = "windows", target_arch = "x86")))]
macro_rules! extern_system_fn {
    ($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
    ($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}


#[cfg(test)]
mod tests {


    use super::*;

    fn g_ort() -> ors_sys::OrtApi {
        let G_ORT_API: Arc<Mutex<AtomicPtr<sys::OrtApi>>> = {
            let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
            assert_ne!(base, std::ptr::null());
            let get_api: extern_system_fn!{ unsafe fn(u32) -> *const sys::OrtApi } =
                unsafe { (*base).GetApi.unwrap() };
            let api: *const sys::OrtApi = unsafe { get_api(sys::ORT_API_VERSION) };
            Arc::new(Mutex::new(AtomicPtr::new(api as *mut sys::OrtApi)))
        };
        let mut api_ref = G_ORT_API
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");
        let api_ref_mut: &mut *mut sys::OrtApi = api_ref.get_mut();
        let api_ptr_mut: *mut sys::OrtApi = *api_ref_mut;
    
        assert_ne!(api_ptr_mut, std::ptr::null_mut());
    
        unsafe { *api_ptr_mut }
    }


    extern_system_fn! {
        /// Callback from C that will handle the logging, forwarding the runtime's logs to the tracing crate.
        pub(crate) fn custom_logger(
            _params: *mut std::ffi::c_void,
            severity: sys::OrtLoggingLevel,
            category: *const i8,
            logid: *const i8,
            code_location: *const i8,
            message: *const i8,
        ) {}

    #[test]
    fn it_works() {
        println!("onnxruntime api verseion: {}", ors_sys::ORT_API_VERSION);
        sys::load().expect("failed to load");
        assert_eq!(8, ors_sys::ORT_API_VERSION);
        let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();
        let mut env_ptr1: sys::OrtLoggingFunction = Some(custom_logger);
        let mut env_ptr2 = std::ptr::null_mut();
        let mut env_ptr4 = std::ptr::null_mut();
        let create_env = g_ort().CreateEnvWithCustomLogger.unwrap();
        let status = {
            unsafe {
                create_env(
                    env_ptr1,
                    env_ptr2
                    ,sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE.into(), env_ptr4
                    ,&mut env_ptr
                )
            }
        };
    }
    }
}