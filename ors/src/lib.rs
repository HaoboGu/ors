mod env;
mod session;
mod tensor;
mod status;
mod error;
mod value;

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, sync::{Arc, Mutex, atomic::AtomicPtr}};
    use ors_sys as sys;
    use std::ptr::null;

    // Make functions `extern "C"` for normal targets.
    // This behaviors like `extern "system"`.
    #[cfg(not(all(target_os = "windows", target_arch = "x86")))]
    macro_rules! extern_system_fn {
        ($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
        ($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
        ($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
        ($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
    }

    fn g_ort() -> ors_sys::OrtApi {
        let g_ort_api: Arc<Mutex<AtomicPtr<sys::OrtApi>>> = {
            let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
            assert_ne!(base, std::ptr::null());
            let get_api: extern_system_fn!{ unsafe fn(u32) -> *const sys::OrtApi } =
                unsafe { (*base).GetApi.unwrap() };
            let api: *const sys::OrtApi = unsafe { get_api(sys::ORT_API_VERSION) };
            Arc::new(Mutex::new(AtomicPtr::new(api as *mut sys::OrtApi)))
        };
        let mut api_ref = g_ort_api
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");
        let api_ref_mut: &mut *mut sys::OrtApi = api_ref.get_mut();
        let api_ptr_mut: *mut sys::OrtApi = *api_ref_mut;
        assert_ne!(api_ptr_mut, std::ptr::null_mut());
        unsafe { *api_ptr_mut }
    }

    #[test]
    fn it_works() {
        // Suppose that onnxruntime's dynamic library has already added in PATH
        assert_eq!(8, ors_sys::ORT_API_VERSION);
        println!("onnxruntime api verseion: {}", ors_sys::ORT_API_VERSION);
        let error_code = 1;
        let msg_ptr: *const i8 = std::ptr::null_mut();
        let create_status_fn = g_ort().CreateStatus.unwrap();
        let status_ptr = unsafe { 
            create_status_fn(error_code.try_into().unwrap(), msg_ptr)
        };
        assert_ne!(null(), status_ptr);
        println!("{:?}", status_ptr);
    }
}