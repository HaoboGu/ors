use ors_sys::*;
use std::ffi::CStr;
pub struct OrtValue {}

impl OrtValue {
    fn new() {
        let api: &OrtApiBase = unsafe { &(*OrtGetApiBase()) };
        let version = unsafe {
            CStr::from_ptr(api.GetVersionString.unwrap()())
                .to_string_lossy()
                .into_owned()
        };
        println!("version: {}", version);
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let o = OrtValue::new();
    }
}
