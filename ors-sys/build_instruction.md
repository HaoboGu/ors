# Build Instruction
## Generate Bindings on m1 Mac 
First, go to `ors-sys` folder:
```
cd ors-sys
```

### Build for x86_64-apple-darwin
```shell
# Add rust target
rustup target install x86_64-apple-darwin
# Build
cargo build --release --target=x86_64-apple-darwin --features "generete-bindings"
```

### Build for x86_64-unknown-linux-gnu
```shell
# Add rust target
rustup target add x86_64-unknown-linux-gnu
# Add cross compiling toolchain for rust and C
brew tap messense/macos-cross-toolchains
# install x86_64-unknown-linux-gnu toolchain
brew install x86_64-unknown-linux-gnu

# Build
export CC_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-gcc
export CXX_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-g++
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc
cargo build --release --target x86_64-unknown-linux-gnu --features "generete-bindings"
``` 

### Build for x86_64-pc-windows-gnu
```
brew install mingw-w64
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu --features "generete-bindings"
```
