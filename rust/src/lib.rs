// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

//! # Trustmark
//!
//! An implementation of TrustMark watermarking for the Content Authenticity Initiative (CAI) in
//! Rust, as described in:
//!
//! ---
//!
//! **TrustMark - Universal Watermarking for Arbitrary Resolution Images**
//!
//! <https://arxiv.org/abs/2311.18297>
//!
//! [Tu Bui]<sup>1</sup>, [Shruti Agarwal]<sup>2</sup>, [John Collomosse]<sup>1,2</sup>
//!
//! <sup>1</sup>DECaDE Centre for the Decentralized Digital Economy, University of Surrey, UK.\
//! <sup>2</sup>Adobe Research, San Jose CA.
//!
//! ---
//!
//! This is a re-implementation of the [trustmark] Python library.
//!
//! [Tu Bui]: https://www.surrey.ac.uk/people/tu-bui
//! [Shruti Agarwal]: https://research.adobe.com/person/shruti-agarwal/
//! [John Collomosse]: https://www.collomosse.com/
//! [trustmark]: https://pypi.org/project/trustmark/
//!
//! ## Example
//!
//! ```rust
//! use trustmark::{Trustmark, Version, Variant};
//!
//! # fn main() {
//! let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
//! let input = image::open("../images/ghost.png").unwrap();
//! let output = tm.encode("0010101".to_owned(), input, 0.95);
//! # }
//! ```
use std::{path::Path, sync::Mutex};

use image::{DynamicImage, GenericImageView as _};
use ort::{
    session::Session,
    value::{TensorValueType, Value},
};

use self::{bits::Bits, image_processing::ModelImage};

mod bits;
mod image_processing;
mod model;
mod session;

/// A loaded Trustmark model.
///
/// [`Trustmark::new`] and [`Trustmark::with_options`] load both the encoder and the decoder.
/// [`Trustmark::encoder_only`] and [`Trustmark::decoder_only`] load one of them, for a process
/// that only embeds or only reads watermarks.
pub struct Trustmark {
    encoder: Option<Mutex<Session>>,
    decoder: Option<Mutex<Session>>,
    encoder_provider: Option<ProviderInfo>,
    decoder_provider: Option<ProviderInfo>,
    version: Version,
    variant: Variant,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("watermark is corrupt or missing")]
    CorruptWatermark,
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),
    #[error("image processing error: {0}")]
    ImageProcessing(#[from] image_processing::Error),
    #[error("bits processing error: {0}")]
    Bits(bits::Error),
    #[error("invalid model variant")]
    InvalidModelVariant,
    #[error("ONNX session lock is poisoned")]
    SessionLockPoisoned,
    /// The requested execution provider could not be used, and [`SessionOptions::fallback`] is
    /// [`ProviderFallback::Error`].
    #[error("could not use the {provider} execution provider: {source}")]
    ExecutionProvider {
        provider: ExecutionProvider,
        source: ort::Error,
    },
    /// [`Trustmark::encode`] was called on a model loaded with [`Trustmark::decoder_only`].
    #[error("the encoder was not loaded")]
    EncoderNotLoaded,
    /// [`Trustmark::decode`] was called on a model loaded with [`Trustmark::encoder_only`].
    #[error("the decoder was not loaded")]
    DecoderNotLoaded,
}

impl From<bits::Error> for Error {
    fn from(value: bits::Error) -> Self {
        match value {
            bits::Error::CorruptWatermark => Error::CorruptWatermark,
            err => Error::Bits(err),
        }
    }
}

pub use bits::Version;
pub use model::Variant;
pub use session::{ExecutionProvider, ProviderFallback, ProviderInfo};

/// ONNX Runtime threading and execution provider for the encoder and decoder sessions.
///
/// The default keeps the behaviour of [`Trustmark::new`]: 8 intra-op threads per session, the
/// runtime's default spinning, and the CPU execution provider. A host that runs several models
/// side by side (one per worker thread, for example) sets `intra_threads` to its own budget and
/// usually turns spinning off, so idle sessions do not burn CPU.
///
/// ```no_run
/// use trustmark::{ExecutionProvider, ProviderFallback, SessionOptions, Trustmark, Variant, Version};
///
/// // Needs the `cuda` feature; uses the CPU if the GPU cannot be used.
/// let options = SessionOptions {
///     execution_provider: ExecutionProvider::Cuda { device_id: 0 },
///     fallback: ProviderFallback::Cpu,
///     ..SessionOptions::default()
/// };
/// let tm = Trustmark::with_options("./models", Variant::Q, Version::Bch5, options).unwrap();
/// if let Some(reason) = &tm.decoder_provider().unwrap().fallback_reason {
///     eprintln!("decoding on the CPU: {reason}");
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Intra-op threads of each session (the encoder and the decoder each get this many).
    pub intra_threads: usize,
    /// Whether intra-op threads spin while waiting for work; `None` keeps the runtime default.
    pub intra_op_spinning: Option<bool>,
    /// The hardware backend of both sessions. Defaults to the CPU.
    pub execution_provider: ExecutionProvider,
    /// What to do if `execution_provider` cannot be used. Defaults to returning an error.
    pub fallback: ProviderFallback,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            intra_threads: 8,
            intra_op_spinning: None,
            execution_provider: ExecutionProvider::Cpu,
            fallback: ProviderFallback::Error,
        }
    }
}

impl Trustmark {
    /// Load a Trustmark model.
    pub fn new<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        version: Version,
    ) -> Result<Self, Error> {
        Self::with_options(models, variant, version, SessionOptions::default())
    }

    /// Load a Trustmark model with explicit ONNX Runtime threading and execution provider.
    ///
    /// Fails with [`Error::ExecutionProvider`] if `options.execution_provider` cannot be used and
    /// `options.fallback` is [`ProviderFallback::Error`]. With [`ProviderFallback::Cpu`], check
    /// [`Trustmark::encoder_provider`] and [`Trustmark::decoder_provider`] to see which provider
    /// each session ended up with.
    pub fn with_options<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        version: Version,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        Self::load(models.as_ref(), variant, version, options, true, true)
    }

    /// Load only the encoder of a Trustmark model.
    ///
    /// Only the encoder file is read, and only its session is created. [`Trustmark::decode`]
    /// returns [`Error::DecoderNotLoaded`].
    pub fn encoder_only<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        version: Version,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        Self::load(models.as_ref(), variant, version, options, true, false)
    }

    /// Load only the decoder of a Trustmark model.
    ///
    /// Only the decoder file is read, and only its session is created. [`Trustmark::encode`]
    /// returns [`Error::EncoderNotLoaded`]. Decoding does not depend on a [`Version`]: the decoder
    /// reads it from the watermark.
    pub fn decoder_only<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        Self::load(
            models.as_ref(),
            variant,
            Version::default(),
            options,
            false,
            true,
        )
    }

    fn load(
        models: &Path,
        variant: Variant,
        version: Version,
        options: SessionOptions,
        encoder: bool,
        decoder: bool,
    ) -> Result<Self, Error> {
        let load = |file: String| session::load(&models.join(file), &options);
        let (encoder, encoder_provider) = encoder
            .then(|| load(variant.encoder_filename()))
            .transpose()?
            .unzip();
        let (decoder, decoder_provider) = decoder
            .then(|| load(variant.decoder_filename()))
            .transpose()?
            .unzip();
        Ok(Self {
            encoder: encoder.map(Mutex::new),
            decoder: decoder.map(Mutex::new),
            encoder_provider,
            decoder_provider,
            version,
            variant,
        })
    }

    /// The execution provider requested for the encoder session and the one it was created with,
    /// or `None` if the encoder was not loaded.
    pub fn encoder_provider(&self) -> Option<&ProviderInfo> {
        self.encoder_provider.as_ref()
    }

    /// The execution provider requested for the decoder session and the one it was created with,
    /// or `None` if the decoder was not loaded.
    pub fn decoder_provider(&self) -> Option<&ProviderInfo> {
        self.decoder_provider.as_ref()
    }

    /// Encode a watermark into an image.
    ///
    /// `watermark` is a bitstring encoding the watermark identifier to encode. `img` is the image
    /// which will be watermarked. `strength` is a number between 0 and 1 indicating how strong the
    /// resulting watermark should be. 0.95 is a normal strength.
    pub fn encode(
        &self,
        watermark: String,
        img: DynamicImage,
        strength: f32,
    ) -> Result<DynamicImage, Error> {
        let (original_width, original_height) = img.dimensions();
        let aspect_ratio = original_width as f32 / original_height as f32;

        // the image is always encoded with size 256x256
        let encode_size = 256;

        let input_img: Value<TensorValueType<f32>> =
            ModelImage(encode_size, self.variant, img.clone()).try_into()?;
        // before the payload is checked against `self.version`, which decoder-only models lack
        let encoder = self.encoder.as_ref().ok_or(Error::EncoderNotLoaded)?;
        let bits: Value<TensorValueType<f32>> =
            Bits::apply_error_correction_and_schema(watermark, self.version)?.into();
        let output_img = {
            let mut encoder = encoder.lock().map_err(|_| Error::SessionLockPoisoned)?;
            let outputs = encoder.run(ort::inputs![
                "onnx::Concat_0" => input_img,
                "onnx::Gemm_1" => bits,
            ])?;
            outputs["image"].try_extract_array::<f32>()?.to_owned()
        };

        // Need to calculate and apply the residual.
        let input_img: Value<TensorValueType<f32>> =
            ModelImage(encode_size, self.variant, img.clone()).try_into()?;
        let residual = (self.variant.strength_multiplier() * strength)
            * (output_img - input_img.try_extract_array::<f32>()?);

        // Residual should be small perturbations.
        let mut residual = residual.clamp(-0.2, 0.2);
        if (self.variant == Variant::Q && !(0.5..=2.0).contains(&aspect_ratio))
            || self.variant == Variant::P
        {
            residual = image_processing::remove_boundary_artifact(
                residual,
                (original_width as usize, original_height as usize),
                self.variant,
            );
        }

        let ModelImage(_, _, residual) = (encode_size, self.variant, residual).try_into()?;

        Ok(image_processing::apply_residual(img, residual))
    }

    /// Decode a watermark from an image.
    pub fn decode(&self, img: DynamicImage) -> Result<String, Error> {
        // P variant has a smaller decode size
        let decode_size = if self.variant == Variant::P { 224 } else { 256 };

        let img: Value<TensorValueType<f32>> =
            ModelImage(decode_size, self.variant, img).try_into()?;
        let watermark = {
            let mut decoder = self
                .decoder
                .as_ref()
                .ok_or(Error::DecoderNotLoaded)?
                .lock()
                .map_err(|_| Error::SessionLockPoisoned)?;
            let outputs = decoder.run(ort::inputs![
                "image" => img,
            ])?;
            outputs["output"].try_extract_array::<f32>()?.to_owned()
        };
        let watermark: Bits = watermark.try_into()?;
        Ok(watermark.get_data())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_models() {
        Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
    }

    /// A models directory that holds only `file` from `./models`.
    fn models_with_only(file: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("trustmark-{}-{file}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = std::fs::canonicalize(Path::new("./models").join(file)).unwrap();
        if std::fs::hard_link(&model, dir.join(file)).is_err() {
            std::fs::copy(&model, dir.join(file)).unwrap();
        }
        dir
    }

    #[test]
    fn encoder_only_loads_just_the_encoder() {
        let models = models_with_only(&Variant::Q.encoder_filename());
        let tm = Trustmark::encoder_only(
            &models,
            Variant::Q,
            Version::Bch5,
            SessionOptions::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(models).unwrap();
        assert!(tm.encoder_provider().is_some());
        assert_eq!(tm.decoder_provider(), None);

        let input = image::open("../images/ghost.png").unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = DynamicImage::ImageRgba8(
            tm.encode(watermark.clone(), input, 0.95)
                .unwrap()
                .to_rgba8(),
        );
        let err = tm.decode(encoded.clone()).unwrap_err();
        assert!(matches!(err, Error::DecoderNotLoaded), "{err:?}");

        let full = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        assert_eq!(full.decode(encoded).unwrap(), watermark);
    }

    #[test]
    fn decoder_only_loads_just_the_decoder() {
        let full = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let input = image::open("../images/ghost.png").unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = full.encode(watermark.clone(), input.clone(), 0.95).unwrap();

        let models = models_with_only(&Variant::Q.decoder_filename());
        let tm = Trustmark::decoder_only(&models, Variant::Q, SessionOptions::default()).unwrap();
        std::fs::remove_dir_all(models).unwrap();
        assert_eq!(tm.encoder_provider(), None);
        assert!(tm.decoder_provider().is_some());

        let decoded = tm
            .decode(DynamicImage::ImageRgba8(encoded.to_rgba8()))
            .unwrap();
        assert_eq!(decoded, watermark);
        let err = tm.encode(watermark, input, 0.95).unwrap_err();
        assert!(matches!(err, Error::EncoderNotLoaded), "{err:?}");
    }

    #[test]
    fn roundtrip_with_one_thread_and_no_spinning() {
        let options = SessionOptions {
            intra_threads: 1,
            intra_op_spinning: Some(false),
            ..SessionOptions::default()
        };
        let tm = Trustmark::with_options("./models", Variant::Q, Version::Bch5, options).unwrap();
        let input = image::open("../images/ghost.png").unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = tm.encode(watermark.clone(), input, 0.95).unwrap();
        let decoded = tm
            .decode(DynamicImage::ImageRgba8(encoded.to_rgba8()))
            .unwrap();
        assert_eq!(watermark, decoded);
    }

    fn roundtrip(path: impl AsRef<Path>) {
        let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let input = image::open(path.as_ref()).unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = tm.encode(watermark.clone(), input, 0.95).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(encoded.to_rgba8())
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let input = image::load_from_memory(png.get_ref()).unwrap();
        let decoded = tm.decode(input).unwrap();
        assert_eq!(watermark, decoded);
    }

    #[test]
    fn roundtrip_ghost() {
        roundtrip("../images/ghost.png");
    }

    #[test]
    fn roundtrip_ufo() {
        roundtrip("../images/ufo_240.jpg");
    }

    #[test]
    fn shared_model_roundtrips_on_multiple_threads() {
        let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let input = image::open("../images/ghost.png").unwrap();
        let start = std::sync::Barrier::new(2);
        let watermarks = [
            "1011011110011000111111000000011111011111011100000110110110111",
            "0011011110011000111111000000011111011111011100000110110110111",
        ];

        std::thread::scope(|scope| {
            let handles = watermarks.map(|watermark| {
                let input = input.clone();
                let tm = &tm;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    let encoded = tm.encode(watermark.to_owned(), input, 0.95).unwrap();
                    let decoded = tm
                        .decode(DynamicImage::ImageRgba8(encoded.to_rgba8()))
                        .unwrap();
                    assert_eq!(decoded, watermark);
                })
            });

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
}
