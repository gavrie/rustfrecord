fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            // println!("cargo:rustc-link-arg=-Wl,--no-as-needed"); // not for macos
            // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-lib=torch");
            println!("cargo:rustc-link-lib=python3.9");
            // /usr/lib/aarch64-linux-gnu/
            // = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo 
            // (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)

        }
        _ => {}
    }
}
