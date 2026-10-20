// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::cmp;

use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use image::{
    imageops::FilterType, DynamicImage, GenericImageView as _, GrayAlphaImage, GrayImage,
    ImageBuffer, Rgb32FImage, RgbImage, Rgba32FImage, RgbaImage,
};
use ndarray::{s, Array, ArrayD, Axis, ShapeError};
use ort::value::{TensorValueType, Value};

use crate::Variant;

/// Re-normalize a floating point value (either scalar or array) from the range [0,1] to the range
/// [-1, 1].
macro_rules! convert_from_0_1_to_neg1_1 {
    ($f:expr) => {
        $f * 2. - 1.
    };
}

/// Re-normalize a floating point value (either scalar or array) from the range [-1, 1] to the
/// range [0, 1].
macro_rules! convert_from_neg1_1_to_0_1 {
    ($f:expr) => {
        ($f + 1.) / 2.
    };
}

pub(super) struct ModelImage(pub(super) u32, pub(super) Variant, pub(super) DynamicImage);

/// The error type for the `image_processing` module.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Something went wrong during inference.
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),

    /// We were unable to make an `ndarray::Array` of the requested shape.
    #[error("shape error: {0}")]
    Shape(#[from] ShapeError),

    /// The input array has an unexpected shape.
    #[error("invalid shape")]
    InvalidShape,

    // We could not create an `ImageBuffer` with the requested array.
    #[error("invalid image")]
    Image,

    /// We were unable to resize the input image.
    #[error("resize error: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),
}

/// Converting a borrowed `ModelImage` leaves the image to the caller, which can keep using it
/// without having cloned it.
impl TryFrom<&ModelImage> for Value<TensorValueType<f32>> {
    type Error = Error;

    fn try_from(&ModelImage(size, variant, ref img): &ModelImage) -> Result<Self, Self::Error> {
        let (w, h, xpos, ypos) = center_crop_size_and_offset(variant, img);

        let options = ResizeOptions::new()
            .crop(xpos as f64, ypos as f64, w as f64, h as f64)
            .resize_alg(ResizeAlg::Interpolation(
                fast_image_resize::FilterType::Bilinear,
            ));
        let modified_img = resize_img(img, size, size, options)?;

        let img = modified_img.into_rgb32f().into_vec();
        let array = Array::from(img);

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_0_1_to_neg1_1!(array);

        let mut array = array
            .to_shape([size as usize, size as usize, 3])?
            .insert_axis(Axis(3))
            .reversed_axes();
        array.swap_axes(2, 3);
        assert_eq!(array.shape(), &[1, 3, size as usize, size as usize]);
        Ok(Value::from_array(array.to_owned())?)
    }
}

impl TryFrom<ModelImage> for Value<TensorValueType<f32>> {
    type Error = Error;

    fn try_from(img: ModelImage) -> Result<Self, Self::Error> {
        Self::try_from(&img)
    }
}

impl TryFrom<(u32, Variant, ArrayD<f32>)> for ModelImage {
    type Error = Error;

    fn try_from(
        (size, variant, mut array): (u32, Variant, ArrayD<f32>),
    ) -> Result<Self, Self::Error> {
        let &[1, 3, height, width] = &array.shape().to_owned()[..] else {
            return Err(Error::InvalidShape);
        };
        array.swap_axes(2, 3);
        let array = array.reversed_axes().remove_axis(Axis(3));
        let array = array.to_shape([width * height * 3])?;

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_neg1_1_to_0_1!(array);

        let image = Rgb32FImage::from_vec(width as u32, height as u32, array.to_vec())
            .ok_or(Error::Image)?;

        Ok(Self(size, variant, image.into()))
    }
}

/// Apply `residual` to the `input`.
///
/// This function upscales `residual` to be the size of of `input`, then adds `residual` to the
/// color channels of `input`. The result is an `Rgba32F` image if `input` has an alpha channel
/// (which is kept unchanged), and an `Rgb32F` image otherwise.
pub(super) fn apply_residual(input: DynamicImage, residual: DynamicImage) -> DynamicImage {
    let (w, h) = input.dimensions();

    // Adds the upscaled residual in place to the color channels of `samples`, which holds
    // `channels` floats per pixel. Upscaling only after the input has been converted (and
    // dropped) keeps at most two full-size images alive.
    let add_residual = |samples: &mut [f32], channels: usize| {
        let residual = residual
            .resize_exact(w, h, FilterType::Triangle)
            .into_rgb32f();

        for (pixel, residual) in samples
            .chunks_exact_mut(channels)
            .zip(residual.chunks_exact(3))
        {
            for (x, &y) in pixel.iter_mut().zip(residual) {
                let sum = convert_from_0_1_to_neg1_1!(*x) + convert_from_0_1_to_neg1_1!(y);
                *x = convert_from_neg1_1_to_0_1!(f32::min(sum, 1.0));
            }
        }
    };

    if input.color().has_alpha() {
        let mut output = input.into_rgba32f();
        add_residual(&mut output, 4);
        output.into()
    } else {
        let mut output = input.into_rgb32f();
        add_residual(&mut output, 3);
        output.into()
    }
}

/// Return whether the model sees a center-cropped square instead of the whole image.
///
/// The P variant is always center-cropped. Other variants are center-cropped when one side of the
/// image is more than twice as long as the other.
pub(super) fn is_center_cropped(variant: Variant, (width, height): (u32, u32)) -> bool {
    height > width * 2 || width > height * 2 || variant == Variant::P
}

/// Return the size and offset of the "center-cropped" image.
///
/// Returns `(width, height, xpos, ypos)` for the square to crop.
///
/// For long-skinny images or short-wide images, we want to crop a square image with side length of
/// the shorter side out of the center of the image for the model.
fn center_crop_size_and_offset(variant: Variant, img: &DynamicImage) -> (u32, u32, u32, u32) {
    let (width, height) = img.dimensions();

    if is_center_cropped(variant, (width, height)) {
        let m = cmp::min(height, width);
        let offset = (cmp::max(height, width) - m) / 2;

        let xpos;
        let ypos;
        if height > width {
            xpos = 0;
            ypos = offset;
        } else {
            ypos = 0;
            xpos = offset;
        }

        (m, m, xpos, ypos)
    } else {
        (width, height, 0, 0)
    }
}

/// Returns a new `DynamicImage`, resized to `width` by `height` with the specified `ResizeOptions`.
fn resize_img(
    img: &DynamicImage,
    width: u32,
    height: u32,
    options: ResizeOptions,
) -> Result<DynamicImage, Error> {
    let mut modified_img = match img {
        DynamicImage::ImageLuma8(_) => DynamicImage::ImageLuma8(GrayImage::new(width, height)),
        DynamicImage::ImageLumaA8(_) => {
            DynamicImage::ImageLumaA8(GrayAlphaImage::new(width, height))
        }
        DynamicImage::ImageRgb8(_) => DynamicImage::ImageRgb8(RgbImage::new(width, height)),
        DynamicImage::ImageRgba8(_) => DynamicImage::ImageRgba8(RgbaImage::new(width, height)),
        DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma16(ImageBuffer::new(width, height)),
        DynamicImage::ImageLumaA16(_) => {
            DynamicImage::ImageLumaA16(ImageBuffer::new(width, height))
        }
        DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgb16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgba16(_) => DynamicImage::ImageRgba16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgb32F(_) => DynamicImage::ImageRgb32F(Rgb32FImage::new(width, height)),
        DynamicImage::ImageRgba32F(_) => {
            DynamicImage::ImageRgba32F(Rgba32FImage::new(width, height))
        }
        // Technically unreachable, but we error for safety.
        _ => return Err(Error::Image),
    };
    Resizer::new().resize(img, &mut modified_img, &options)?;

    Ok(modified_img)
}

/// Applies the mean padding boundary artifact mitigation.
///
/// Center cropped images have a vertical line problem along the boundary of the residual. This
/// transformation makes this boundary less visible.
pub(super) fn remove_boundary_artifact(
    mut residual: ArrayD<f32>,
    (width, height): (usize, usize),
    _variant: Variant,
) -> ArrayD<f32> {
    // We're going to replace the border of the residual with the mean and also pad the non-center
    // areas with the mean value.
    let channel_means: Vec<f32> = (0_usize..3)
        .map(|i| residual.slice(s![.., i, .., ..]).mean().unwrap())
        .collect();

    // We want one dimension of the output to be 256 and we we want the aspect ratio of the output
    // to match the input image.
    let other_dim;
    let mut mean_padded: ndarray::Array4<f32> = if width > height {
        other_dim = ((width as f32 / height as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, 256_usize, other_dim])
    } else {
        other_dim = ((height as f32 / width as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, other_dim, 256])
    };

    // This softens the transition between the residual area and the rest of the image.
    let border = 2;
    for (i, mean) in channel_means.iter().enumerate() {
        residual.slice_mut(s![0, i, ..border, ..]).fill(*mean);
        residual.slice_mut(s![0, i, -border.., ..]).fill(*mean);
        residual.slice_mut(s![0, i, .., -border..]).fill(*mean);
        residual.slice_mut(s![0, i, .., ..border]).fill(*mean);
        mean_padded.slice_mut(s![0, i, .., ..]).fill(*mean);
    }

    if width > height {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., .., leftover..(leftover + 256)])
            .assign(&residual);
    } else {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., leftover..(leftover + 256), ..])
            .assign(&residual);
    }

    mean_padded.into_dyn()
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;

    #[test]
    fn renormalize_from_0_1() {
        assert_eq!(convert_from_0_1_to_neg1_1!(0.), -1.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.5), 0.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.99), 0.98);
    }

    #[test]
    fn renormalize_from_neg1_1() {
        assert_eq!(convert_from_neg1_1_to_0_1!(-1.), 0.);
        assert_eq!(convert_from_neg1_1_to_0_1!(0.5), 0.75);
        assert_eq!(convert_from_neg1_1_to_0_1!(-0.1), 0.45);
    }

    #[test]
    fn apply_residual_to_color_channels() {
        let residual: DynamicImage = Rgb32FImage::from_fn(4, 4, |x, y| {
            image::Rgb([0.4 + 0.05 * x as f32, 0.6 - 0.05 * y as f32, 0.6])
        })
        .into();
        let input = RgbaImage::from_fn(6, 5, |x, y| {
            image::Rgba([(x * 50) as u8, (y * 60) as u8, 255, (x * y * 12) as u8])
        });

        // Reference: add the upscaled residual in [-1, 1], saturating at 1, and keep the alpha.
        let upscaled = residual
            .resize_exact(6, 5, FilterType::Triangle)
            .into_rgb32f();
        let original = DynamicImage::ImageRgba8(input.clone()).into_rgba32f();
        let DynamicImage::ImageRgba32F(output) =
            apply_residual(input.clone().into(), residual.clone())
        else {
            panic!("RGBA input must produce an Rgba32F image");
        };
        for ((out, original), residual) in output
            .pixels()
            .zip(original.pixels())
            .zip(upscaled.pixels())
        {
            for c in 0..3 {
                let sum = (original[c] * 2. - 1.) + (residual[c] * 2. - 1.);
                assert_eq!(out[c], (sum.min(1.) + 1.) / 2.);
            }
            assert_eq!(out[3], original[3]);
        }

        // The color channels of an RGB input get the same values.
        let rgb = DynamicImage::ImageRgba8(input).to_rgb8();
        let DynamicImage::ImageRgb32F(output_rgb) = apply_residual(rgb.into(), residual) else {
            panic!("RGB input must produce an Rgb32F image");
        };
        for (rgb, rgba) in output_rgb.pixels().zip(output.pixels()) {
            assert_eq!(rgb.0, rgba.0[..3]);
        }
    }

    #[test]
    fn normal_image() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (100, 110, 0, 0)
        );
    }

    #[test]
    fn skinny_image() {
        let image = DynamicImage::new(10, 100, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 0, 45)
        );
    }

    #[test]
    fn wide_image() {
        let image = DynamicImage::new(101, 10, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 45, 0)
        );
    }

    #[test]
    fn always_crop_p() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::P, &image),
            (100, 100, 0, 5)
        );
    }

    #[test]
    fn crop_boundary() {
        for variant in [Variant::B, Variant::C, Variant::Q] {
            // Exactly 2:1 is not center-cropped...
            for (width, height) in [(1024, 512), (512, 1024)] {
                let image = DynamicImage::new(width, height, image::ColorType::L8);
                assert!(!is_center_cropped(variant, (width, height)));
                assert_eq!(
                    center_crop_size_and_offset(variant, &image),
                    (width, height, 0, 0)
                );
            }

            // ...but one pixel past 2:1 is.
            let image = DynamicImage::new(1025, 512, image::ColorType::L8);
            assert!(is_center_cropped(variant, (1025, 512)));
            assert_eq!(
                center_crop_size_and_offset(variant, &image),
                (512, 512, 256, 0)
            );
            let image = DynamicImage::new(512, 1025, image::ColorType::L8);
            assert!(is_center_cropped(variant, (512, 1025)));
            assert_eq!(
                center_crop_size_and_offset(variant, &image),
                (512, 512, 0, 256)
            );
        }

        assert!(is_center_cropped(Variant::P, (512, 512)));
        assert!(is_center_cropped(Variant::P, (1024, 512)));
    }

    #[test]
    fn remove_boundary_artifact_tall() {
        let residual: Array4<f32> = Array4::ones([1, 3, 256, 256]);
        let width = 256;
        let height = 298;

        let output = remove_boundary_artifact(residual.into_dyn(), (width, height), Variant::P);

        assert_eq!(output.shape(), &[1, 3, 298, 256]);
    }
}
