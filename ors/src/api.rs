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

/// get_api exposes an OrtApi instance in crate
pub(crate) fn get_api() -> OrtApi {
    let p = *(API.try_lock().unwrap()).get_mut();
    unsafe { *p }
}
