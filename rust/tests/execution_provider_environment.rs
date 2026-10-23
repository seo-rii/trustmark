// Copyright 2026 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

// A separate integration-test binary keeps this global ORT environment out of the unit tests.
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use trustmark::{ExecutionProvider, ProviderFallback, SessionOptions, Trustmark, Variant, Version};

#[test]
fn cpu_selection_and_fallback_ignore_environment_providers() {
    // An impossible device ordinal also fails on hosts with a working CUDA installation.
    assert!(ort::init()
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(i32::MAX)
            .build()
            .error_on_failure()])
        .commit()
        .unwrap());
    assert!(Session::builder()
        .unwrap()
        .commit_from_file("./models/encoder_Q.onnx")
        .is_err());

    for requested in [
        ExecutionProvider::Cpu,
        ExecutionProvider::Cuda {
            device_id: i32::MAX,
        },
    ] {
        let options = SessionOptions {
            execution_provider: requested,
            fallback: ProviderFallback::Cpu,
            ..SessionOptions::default()
        };
        let tm = Trustmark::with_options("./models", Variant::Q, Version::Bch5, options).unwrap();
        for info in [tm.encoder_provider(), tm.decoder_provider()].map(Option::unwrap) {
            assert_eq!(info.requested, requested);
            assert_eq!(info.registered, ExecutionProvider::Cpu);
            assert_eq!(
                info.fallback_reason.is_some(),
                requested != ExecutionProvider::Cpu
            );
        }
    }
}
