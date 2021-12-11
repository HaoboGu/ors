use lazy_static::lazy_static;
use ors_sys::*;
use std::sync::{atomic::AtomicPtr, Arc, Mutex};

// The entry of onnxruntime api
lazy_static! {
    static ref API: Arc<Mutex<AtomicPtr<OrtApi>>> = {
        let api_base = unsafe { OrtGetApiBase() };

        assert_ne!(api_base, std::ptr::null());

        let api = unsafe { (&(*api_base).GetApi.unwrap())(ORT_API_VERSION) };

        Arc::new(Mutex::new(AtomicPtr::new(api as *mut OrtApi)))
    };
}

/// Macro for calling unsafe methods using get_api()
#[macro_export]
macro_rules! unsafe_api_call {
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

    use super::*;
    
    #[test]
    fn test_macro() {
        let msg = CString::new("error msg").unwrap();
        let status = unsafe_api_call!(CreateStatus, OrtErrorCode_ORT_ENGINE_ERROR, msg.as_ptr());
        let error_code = unsafe_api_call!(GetErrorCode, status);
        assert_eq!(error_code, 5);
    }
}