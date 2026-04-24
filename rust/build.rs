use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate must live under repo_root/rust")
        .to_path_buf()
}

fn maybe_libtorch_env_paths() -> Option<(Vec<String>, Vec<String>)> {
    let libtorch = env::var("LIBTORCH").ok();
    let include = env::var("LIBTORCH_INCLUDE").ok();
    let lib = env::var("LIBTORCH_LIB").ok();

    match (libtorch, include, lib) {
        (_, Some(include), Some(lib)) => Some((vec![include], vec![lib])),
        (Some(root), None, None) => Some((
            vec![
                format!("{root}/include"),
                format!("{root}/include/torch/csrc/api/include"),
            ],
            vec![format!("{root}/lib")],
        )),
        (Some(root), Some(include), None) => Some((vec![include], vec![format!("{root}/lib")])),
        (Some(root), None, Some(lib)) => Some((
            vec![
                format!("{root}/include"),
                format!("{root}/include/torch/csrc/api/include"),
            ],
            vec![lib],
        )),
        _ => None,
    }
}

fn maybe_python_torch_paths() -> Option<(Vec<String>, Vec<String>)> {
    if env::var("HCP_ENABLE_TORCH").ok().as_deref() != Some("1") {
        return None;
    }

    let script = r#"
from torch.utils.cpp_extension import include_paths, library_paths
print("INCLUDES=" + "|".join(include_paths()))
print("LIBS=" + "|".join(library_paths()))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut includes = Vec::new();
    let mut libs = Vec::new();
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("INCLUDES=") {
            includes = value.split('|').map(ToOwned::to_owned).collect();
        } else if let Some(value) = line.strip_prefix("LIBS=") {
            libs = value.split('|').map(ToOwned::to_owned).collect();
        }
    }
    if includes.is_empty() || libs.is_empty() {
        None
    } else {
        Some((includes, libs))
    }
}

fn has_library(lib_dirs: &[String], name: &str) -> bool {
    let candidates = [
        format!("lib{name}.so"),
        format!("lib{name}.dylib"),
        format!("{name}.lib"),
    ];
    lib_dirs.iter().any(|dir| {
        candidates
            .iter()
            .any(|file| Path::new(dir).join(file).exists())
    })
}

fn link_libtorch_cuda(cuda_libs_present: bool) {
    if !cuda_libs_present {
        return;
    }

    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux") {
        // CUDA kernels are registered by static initializers in libtorch_cuda.
        // Keep the shared libraries even when the linker sees no direct symbols;
        // pass the libraries in the same linker-arg group so rustc cannot reorder them.
        println!(
            "cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state"
        );
    } else {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }
}

fn main() {
    let root = repo_root();
    println!("cargo:rustc-check-cfg=cfg(hcp_torch_enabled)");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("src/ringattn_runtime.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("src/rust_bridge.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/hcp_ringattn/core/ringattn_runtime.h")
            .display()
    );
    println!("cargo:rerun-if-env-changed=HCP_ENABLE_TORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_INCLUDE");
    println!("cargo:rerun-if-env-changed=LIBTORCH_LIB");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(root.join("include"))
        .file(root.join("src/ringattn_runtime.cc"))
        .file(root.join("src/rust_bridge.cc"));

    let torch_paths = if env::var("HCP_ENABLE_TORCH").ok().as_deref() == Some("1") {
        maybe_libtorch_env_paths().or_else(maybe_python_torch_paths)
    } else {
        None
    };

    if let Some((includes, libs)) = torch_paths {
        println!("cargo:rustc-cfg=hcp_torch_enabled");
        build.define("HCP_ENABLE_TORCH", None);
        build.flag_if_supported("-Wno-unused-parameter");
        for include in includes {
            build.include(include);
        }
        for lib in &libs {
            println!("cargo:rustc-link-search=native={lib}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        }
        println!("cargo:rustc-link-lib=dylib=torch");
        println!("cargo:rustc-link-lib=dylib=torch_cpu");
        println!("cargo:rustc-link-lib=dylib=c10");
        link_libtorch_cuda(has_library(&libs, "torch_cuda") && has_library(&libs, "c10_cuda"));
    }

    build.compile("hcp_ringattn_cxx_bridge");
}
