#[cfg(feature = "dynamic-loading")]
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/windows/x86_64/bindings_dynamic.rs"
));

#[cfg(feature = "dynamic-loading")]
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/macos/x86_64/bindings_dynamic.rs"
));

#[cfg(feature = "dynamic-loading")]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/macos/aarch64/bindings_dynamic.rs"
));

#[cfg(feature = "dynamic-loading")]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/linux/x86_64/bindings_dynamic.rs"
));