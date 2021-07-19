#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/bindings/windows/x86_64/bindings.rs"
));

// TODO: add more bindings
// #[cfg(all(target_os = "darwin", target_arch = "x86_64"))]
// include!(concat!(
//     env!("CARGO_MANIFEST_DIR"),
//     "/src/bindings/darwin/x86_64/bindings.rs"
// ));

// #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
// include!(concat!(
//     env!("CARGO_MANIFEST_DIR"),
//     "/src/bindings/linux/x86_64/bindings.rs"
// ));
