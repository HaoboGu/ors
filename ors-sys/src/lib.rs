#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// Disable clippy and `u128` not being FFI-safe (see #1)
#![allow(clippy::all)]
#![allow(improper_ctypes)]

pub mod linking;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/bindings.rs"
));


#[cfg(target_os = "windows")]
pub type OnnxEnumInt = i32;
#[cfg(not(target_os = "windows"))]
pub type OnnxEnumInt = u32;

pub mod Library {
    use std::path::PathBuf;

    pub fn find() -> Option<PathBuf> {
        return Some(PathBuf::from("/Users/haobogu/Library/Caches/.cosy/env/libonnxruntime.1.8.1.dylib"));
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(8, ORT_API_VERSION);
    }
}
