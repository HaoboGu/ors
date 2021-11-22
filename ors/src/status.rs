use std::{
    ffi::{CStr, CString},
    sync::{atomic::AtomicPtr, Arc, Mutex},
};

use lazy_static::{lazy_static};
use ors_sys::*;
lazy_static! {
    static ref API: Arc<Mutex<AtomicPtr<OrtApi>>> = {
        let api_base = unsafe { OrtGetApiBase() };

        assert_ne!(api_base, std::ptr::null());

        let api = unsafe { (&(*api_base).GetApi.unwrap())(ORT_API_VERSION) };

        Arc::new(Mutex::new(AtomicPtr::new(api as *mut OrtApi)))
    };
}

fn get_api() -> OrtApi {
    let p = *(API.try_lock().unwrap()).get_mut();
    unsafe { *p }
}

fn create_status() {
    let msg = CString::new("OKOKOKO").unwrap();
    let status: *const OrtStatus =
        unsafe { get_api().CreateStatus.unwrap()(OrtErrorCode_ORT_MODEL_LOADED, msg.as_ptr()) };
    let error_code = unsafe { get_api().GetErrorCode.unwrap()(status) };
    println!("error code: {}", error_code)
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        create_status()
    }
}
