// Copyright 2025 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

//! Building the ONNX Runtime sessions, including the choice of execution provider.

use std::{fmt::Display, path::Path};

use ort::session::{
    builder::{GraphOptimizationLevel, SessionBuilder},
    Session,
};

use crate::{Error, SessionOptions};

/// The hardware backend ONNX Runtime runs the encoder and decoder on.
///
/// Providers other than [`ExecutionProvider::Cpu`] need the cargo feature of the same name, which
/// makes `ort` download an ONNX Runtime build with that provider, and the provider's own runtime
/// libraries on the machine that runs the model. Without the feature, requesting the provider
/// fails (or falls back to the CPU, see [`ProviderFallback`]) when the model is loaded.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionProvider {
    /// ONNX Runtime's built-in CPU provider. Always available.
    #[default]
    Cpu,
    /// NVIDIA GPUs through the CUDA provider.
    ///
    /// Needs the `cuda` feature. The ONNX Runtime build that `ort` downloads for it (1.22) loads
    /// CUDA 12 and cuDNN 9 when the provider is registered (on Linux `libcudart.so.12`,
    /// `libcublas.so.12`, `libcublasLt.so.12`, `libcufft.so.11`, `libcudnn.so.9` and the cuDNN
    /// libraries it opens), so they must be on the library search path. See the crate README.
    Cuda {
        /// The CUDA device ordinal, `0` for the first GPU.
        device_id: i32,
    },
}

impl ExecutionProvider {
    /// The `ort` provider to register, or the reason it cannot be registered. `None` for the CPU,
    /// which ONNX Runtime always uses and needs no registration.
    fn dispatch(
        self,
    ) -> Option<Result<ort::execution_providers::ExecutionProviderDispatch, String>> {
        match self {
            ExecutionProvider::Cpu => None,
            ExecutionProvider::Cuda { device_id } => Some(if cfg!(feature = "cuda") {
                Ok(ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build())
            } else {
                Err("trustmark was built without the `cuda` feature".to_owned())
            }),
        }
    }
}

impl Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::Cpu => f.write_str("CPU"),
            ExecutionProvider::Cuda { device_id } => write!(f, "CUDA (device {device_id})"),
        }
    }
}

/// What happens when the requested [`ExecutionProvider`] cannot be used.
///
/// A provider cannot be used when this crate was built without its feature, when ONNX Runtime
/// fails to register it (for example because a runtime library such as cuDNN is missing), or when
/// ONNX Runtime fails to create the session with it (for example because there is no such device).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ProviderFallback {
    /// Fail to load the model with [`Error::ExecutionProvider`].
    #[default]
    Error,
    /// Load the model on the CPU instead, and record why in [`ProviderInfo::fallback_reason`].
    Cpu,
}

/// The execution provider a session was requested with and the one it was created with.
///
/// Returned by [`Trustmark::encoder_provider`](crate::Trustmark::encoder_provider) and
/// [`Trustmark::decoder_provider`](crate::Trustmark::decoder_provider).
///
/// `registered` is the provider ONNX Runtime accepted for the session: registering it succeeded and
/// the session was created with it. It does not mean that every node of the model runs on that
/// provider. ONNX Runtime places nodes the provider does not support on the CPU, and does not
/// report that placement through its API; it only logs it. To see it, run with the environment
/// variable `ORT_LOG=verbose` and a [`tracing`](https://docs.rs/tracing) subscriber, which receives
/// ONNX Runtime's log messages through `ort`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderInfo {
    /// The provider requested in [`SessionOptions::execution_provider`].
    pub requested: ExecutionProvider,
    /// The provider the session was created with: `requested`, or [`ExecutionProvider::Cpu`]
    /// after a fallback.
    pub registered: ExecutionProvider,
    /// Why `requested` could not be used, if the session fell back to the CPU.
    pub fallback_reason: Option<String>,
}

impl ProviderInfo {
    fn new(requested: ExecutionProvider) -> Self {
        Self {
            requested,
            registered: requested,
            fallback_reason: None,
        }
    }
}

/// Create a session for the model at `path` as `options` asks, falling back to the CPU if the
/// options allow it.
pub(crate) fn load(
    path: &Path,
    options: &SessionOptions,
) -> Result<(Session, ProviderInfo), Error> {
    let requested = options.execution_provider;
    let Some(provider) = requested.dispatch() else {
        let session = builder(options)?.commit_from_file(path)?;
        return Ok((session, ProviderInfo::new(requested)));
    };
    let (source, session_failed) = match provider {
        Err(reason) => (ort::Error::new(reason), false),
        Ok(provider) => {
            match builder(options)?.with_execution_providers([provider.error_on_failure()]) {
                Err(source) => (source, false),
                Ok(builder) => match builder.commit_from_file(path) {
                    Ok(session) => return Ok((session, ProviderInfo::new(requested))),
                    Err(source) => (source, true),
                },
            }
        }
    };
    if options.fallback == ProviderFallback::Error && !session_failed {
        return Err(Error::ExecutionProvider {
            provider: requested,
            source,
        });
    }
    // Creating the session can fail because of the provider (out of device memory, say) or
    // because of the model file. Only blame the provider if the model loads on the CPU.
    let session = builder(options)?.commit_from_file(path)?;
    match options.fallback {
        ProviderFallback::Error => Err(Error::ExecutionProvider {
            provider: requested,
            source,
        }),
        ProviderFallback::Cpu => Ok((
            session,
            ProviderInfo {
                requested,
                registered: ExecutionProvider::Cpu,
                fallback_reason: Some(source.to_string()),
            },
        )),
    }
}

fn builder(options: &SessionOptions) -> Result<SessionBuilder, Error> {
    // The selected provider and CPU fallback must not inherit the host's default providers.
    let mut builder = Session::builder()?
        .with_no_environment_execution_providers()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(options.intra_threads)?;
    if let Some(spinning) = options.intra_op_spinning {
        builder = builder.with_intra_op_spinning(spinning)?;
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;

    use super::*;
    use crate::{Trustmark, Variant, Version};

    const WATERMARK: &str = "1011011110011000111111000000011111011111011100000110110110111";

    fn roundtrip(tm: &Trustmark) {
        let input = image::open("../images/ghost.png").unwrap();
        let encoded = tm.encode(WATERMARK.to_owned(), input, 0.95).unwrap();
        let decoded = tm
            .decode(DynamicImage::ImageRgba8(encoded.to_rgba8()))
            .unwrap();
        assert_eq!(decoded, WATERMARK);
    }

    fn cuda(fallback: ProviderFallback) -> SessionOptions {
        SessionOptions {
            execution_provider: ExecutionProvider::Cuda { device_id: 0 },
            fallback,
            ..SessionOptions::default()
        }
    }

    #[test]
    fn default_options_use_the_cpu() {
        let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let cpu = ProviderInfo::new(ExecutionProvider::Cpu);
        assert_eq!(tm.encoder_provider(), &cpu);
        assert_eq!(tm.decoder_provider(), &cpu);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_feature_is_an_error() {
        let err = Trustmark::with_options(
            "./models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Error),
        )
        .err()
        .unwrap();
        assert!(
            matches!(
                err,
                Error::ExecutionProvider {
                    provider: ExecutionProvider::Cuda { device_id: 0 },
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("`cuda` feature"), "{err}");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_feature_falls_back_to_the_cpu() {
        let tm = Trustmark::with_options(
            "./models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Cpu),
        )
        .unwrap();
        for info in [tm.encoder_provider(), tm.decoder_provider()] {
            assert_eq!(info.requested, ExecutionProvider::Cuda { device_id: 0 });
            assert_eq!(info.registered, ExecutionProvider::Cpu);
            assert!(info
                .fallback_reason
                .as_deref()
                .unwrap()
                .contains("`cuda` feature"));
        }
        roundtrip(&tm);
    }

    /// Holds on any machine: either CUDA is usable, or the session falls back and says why.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_with_cpu_fallback_reports_the_provider_in_use() {
        let tm = Trustmark::with_options(
            "./models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Cpu),
        )
        .unwrap();
        for info in [tm.encoder_provider(), tm.decoder_provider()] {
            assert_eq!(info.requested, ExecutionProvider::Cuda { device_id: 0 });
            match info.registered {
                ExecutionProvider::Cuda { device_id: 0 } => assert_eq!(info.fallback_reason, None),
                ExecutionProvider::Cpu => assert!(info.fallback_reason.is_some()),
                other => panic!("unexpected provider {other}"),
            }
        }
        roundtrip(&tm);
    }

    /// Needs an NVIDIA GPU with CUDA 12 and cuDNN 9: `cargo test --features cuda -- --ignored`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs an NVIDIA GPU with CUDA 12 and cuDNN 9"]
    fn cuda_roundtrip_on_gpu() {
        let tm = Trustmark::with_options(
            "./models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Error),
        )
        .unwrap();
        let cuda = ProviderInfo::new(ExecutionProvider::Cuda { device_id: 0 });
        assert_eq!(tm.encoder_provider(), &cuda);
        assert_eq!(tm.decoder_provider(), &cuda);
        roundtrip(&tm);
    }

    /// ONNX Runtime rejects a device ordinal that does not exist when the provider is registered.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs an NVIDIA GPU with CUDA 12 and cuDNN 9"]
    fn cuda_missing_device_is_a_provider_error() {
        let options = SessionOptions {
            execution_provider: ExecutionProvider::Cuda { device_id: 999 },
            ..SessionOptions::default()
        };
        let err = Trustmark::with_options("./models", Variant::Q, Version::Bch5, options)
            .err()
            .unwrap();
        assert!(matches!(err, Error::ExecutionProvider { .. }), "{err:?}");
    }

    /// A model that cannot be loaded at all is reported as such, not as a provider problem.
    #[test]
    fn missing_model_with_cpu_fallback_is_still_an_error() {
        let err = Trustmark::with_options(
            "./no-such-models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Cpu),
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::Ort(_)), "{err:?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs an NVIDIA GPU with CUDA 12 and cuDNN 9"]
    fn missing_model_on_gpu_is_not_a_provider_error() {
        let err = Trustmark::with_options(
            "./no-such-models",
            Variant::Q,
            Version::Bch5,
            cuda(ProviderFallback::Error),
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::Ort(_)), "{err:?}");
    }
}
