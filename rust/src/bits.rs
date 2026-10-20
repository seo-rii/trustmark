// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

use ndarray::{Array1, ArrayD, Axis};
use ort::value::{TensorValueType, Value};

const VERSION_BITS: u16 = 4;

mod bch;

#[derive(Debug)]
pub(super) struct Bits(String);

/// Error type for the `bits` module.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Something went wrong while doing inference.
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),

    /// A character was encounted that was not a '0' or a '1'. Strings that specify bitstrings
    /// should only use these two characters.
    #[error("allowed chars are '0' and '1'")]
    InvalidChar,

    /// A bitstring has too many bits to fit in the requested version's data payload.
    #[error(
        "input bitstring ({bits} bits) has more bits than version allows ({version_allows} bits)"
    )]
    InvalidDataLength { version_allows: usize, bits: usize },

    /// A bitstring of a certain length is required, and a different length was encountered.
    #[error("must be of length 100")]
    InvalidLength,

    /// Model input/output did not have the expected dimensions.
    #[error("invalid dimensions")]
    InvalidDim,

    /// String does not represent a known version.
    #[error("invalid version")]
    InvalidVersion,

    /// Watermark is missing or corrupted.
    ///
    /// Either the image did not have a valid watermark, or too many transmission errors/image
    /// artifacts occurred such that the watermark is no longer recoverable.
    #[error("corrupt watermark")]
    CorruptWatermark,
}

impl Bits {
    /// Constructs a `Bits`, adding in the additional error correction and schema bits.
    pub(super) fn apply_error_correction_and_schema(
        mut input: String,
        version: Version,
    ) -> Result<Self, Error> {
        let data_bits: usize = version.data_bits().into();

        if input.chars().any(|c| c != '0' && c != '1') {
            return Err(Error::InvalidChar);
        }

        if input.len() > data_bits {
            return Err(Error::InvalidDataLength {
                bits: input.len(),
                version_allows: data_bits,
            });
        }

        // pad the input to a whole number of bytes
        input.push_str(&"0".repeat(data_bits.next_multiple_of(8) - input.len()));

        // pack the input into bytes
        let data: Vec<u8> = input
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();

        // calculate the error correction bits
        let mut ecc_state = bch::bch_init(version.allowed_bit_flips() as u32, bch::POLYNOMIAL);
        let ecc = bch::bch_encode(&mut ecc_state, &data);

        // form a bitstring from the error correction bits
        let mut error_correction: String = ecc
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");

        // split off unneeded padding
        input.truncate(data_bits);
        error_correction.truncate(version.ecc_bits().into());

        // form the encoded string
        Ok(Self(format!(
            "{input}{error_correction}{}",
            version.bitstring()
        )))
    }

    /// Get the data out of a `Bits` by removing the error correction bits.
    pub(super) fn get_data(self) -> String {
        let version = self.get_version();
        let Self(mut s) = self;
        s.truncate(version.data_bits().into());
        s
    }

    /// Get the version from the bits.
    pub(super) fn get_version(&self) -> Version {
        match &self.0[98..100] {
            "00" => Version::BchSuper,
            "01" => Version::Bch5,
            "10" => Version::Bch4,
            "11" => Version::Bch3,
            _ => unreachable!(),
        }
    }

    /// Construct a `Bits` from a bitstring.
    ///
    /// This function checks for bitflips in the bitstring using the error-correcting bits, and
    /// corrects them if there are fewer bitflips than are supported by the version. As a last
    /// resort, this function checks for bitflips in the version identifier by trying all possible
    /// versions.
    fn new(s: String) -> Result<Self, Error> {
        let schema_version = Bits::schema_version(&s)?;
        evaluation_order(schema_version)
            .find_map(|version| Bits::new_with_version(&s, version, false).ok())
            .map(|(bits, _)| bits)
            .ok_or(Error::CorruptWatermark)
    }

    /// Decode a bitstring under `options`, collecting every version that passes the BCH check.
    pub(super) fn decode_with(s: &str, options: &DecodeOptions) -> Result<Decoded, Error> {
        let schema_version = Bits::schema_version(s)?;
        let candidates: Vec<Candidate> = evaluation_order(schema_version)
            .filter_map(|version| {
                let (bits, corrected_bits) =
                    Bits::new_with_version(s, version, options.padding_check).ok()?;
                Some(Candidate {
                    version,
                    payload: bits.get_data(),
                    corrected_bits: corrected_bits.into(),
                })
            })
            .collect();
        let accepted = candidates
            .iter()
            .find(|candidate| options.accepts(candidate.version, schema_version))
            .ok_or(Error::CorruptWatermark)?;

        Ok(Decoded {
            payload: accepted.payload.clone(),
            version: accepted.version,
            corrected_bits: accepted.corrected_bits,
            schema_version,
            candidates,
        })
    }

    /// Check that `s` is a 100 bit bitstring and return the version its schema bits name, if any.
    fn schema_version(s: &str) -> Result<Option<Version>, Error> {
        if s.chars().any(|c| c != '0' && c != '1') {
            return Err(Error::InvalidChar);
        }

        if s.len() != 100 {
            return Err(Error::InvalidLength);
        }

        Ok(Version::from_bitstring(&s[96..]).ok())
    }

    /// Check and correct `s` as `version`, returning the corrected bits and the number of
    /// corrected bit flips.
    ///
    /// With `padding_check`, a correction in the zero bits that pad the payload to a whole byte is
    /// an error.
    fn new_with_version(
        s: &str,
        version: Version,
        padding_check: bool,
    ) -> Result<(Self, u8), Error> {
        let data_bits: usize = version.data_bits().into();
        let ecc_bits: usize = version.ecc_bits().into();

        let mut data = s[..data_bits].to_string();
        let mut ecc = s[data_bits..data_bits + ecc_bits].to_string();

        // pad to a whole number of bytes
        data.push_str(&"0".repeat(data_bits.next_multiple_of(8) - data.len()));
        ecc.push_str(&"0".repeat(ecc_bits.next_multiple_of(8) - ecc.len()));

        // pack into bytes
        let mut data: Vec<u8> = data
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();
        let ecc: Vec<u8> = ecc
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();

        // validate and correct
        let mut ecc_state = bch::bch_init(version.allowed_bit_flips() as u32, bch::POLYNOMIAL);
        let bitflips = bch::bch_decode(&mut ecc_state, &mut data, &ecc);

        if bitflips > version.allowed_bit_flips() {
            return Err(Error::CorruptWatermark);
        }

        // No encoder sets the padding bits, so a correction there means the bits are not within
        // the correction capacity of any packet an encoder can produce.
        let padding_mask = (1u8 << (data_bits.next_multiple_of(8) - data_bits)) - 1;
        if padding_check && data.last().is_some_and(|byte| byte & padding_mask != 0) {
            return Err(Error::CorruptWatermark);
        }

        // unpack data and ecc
        let mut data: String = data
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");
        let mut ecc: String = ecc
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");
        data.truncate(data_bits);
        ecc.truncate(ecc_bits);

        Ok((
            Bits(format!("{data}{ecc}{}", version.bitstring())),
            bitflips,
        ))
    }
}

/// The order in which versions are tried: the one named by the schema bits (or the default version
/// when they name none), then the others.
fn evaluation_order(schema_version: Option<Version>) -> impl Iterator<Item = Version> {
    let first = schema_version.unwrap_or_default();
    std::iter::once(first).chain(
        [
            Version::Bch3,
            Version::Bch4,
            Version::Bch5,
            Version::BchSuper,
        ]
        .into_iter()
        .filter(move |version| *version != first),
    )
}

/// Threshold the decoder's output into a bitstring.
pub(super) fn bitstring(array: ArrayD<f32>) -> Result<String, Error> {
    if array.shape() != [1, 100] {
        return Err(Error::InvalidDim);
    }
    Ok(array
        .iter()
        .map(|bit| if *bit < 0. { '0' } else { '1' })
        .collect())
}

/// Which watermarks [`Trustmark::decode_with`](crate::Trustmark::decode_with) accepts.
///
/// Every watermark ends in four schema bits that name its [`Version`]. Those bits are not protected
/// by the error correction, so by default (the behaviour of
/// [`Trustmark::decode`](crate::Trustmark::decode)) a packet that fails the BCH check under the
/// version its schema bits name is tried under every other version, and the first one that passes
/// wins. That recovers watermarks whose schema bits were damaged, but it also lets a damaged or
/// unwatermarked image pass as a weaker version with a payload that was never embedded.
///
/// Passing the BCH check only means that the bits are within the version's correction capacity of
/// some valid codeword. It is not authentication and not proof of provenance. Of 100,000 uniformly
/// random 100-bit strings, 10.1% pass the default options (8.5% as [`Version::Bch3`]), 0.57% pass
/// [`strict`](Self::strict) options, and 0.21% pass when only [`Version::Bch5`] is allowed. About
/// 9% of genuine [`Version::BchSuper`], [`Version::Bch5`] and [`Version::Bch4`] watermarks also
/// pass the BCH check as [`Version::Bch3`], so one whose schema bits were damaged can decode to a
/// payload that was never embedded; see [`Decoded::is_ambiguous`].
///
/// When the embedding version is known, allowing only it with
/// [`with_allowed_versions`](Self::with_allowed_versions) both ignores damaged schema bits and
/// accepts the least noise.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Bit set of allowed versions, indexed by `Version::schema_id`.
    allowed: u8,
    fallback: bool,
    padding_check: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            allowed: 0b1111,
            fallback: true,
            padding_check: false,
        }
    }
}

impl DecodeOptions {
    /// Allow every version, fall back from the schema bits and accept corrections in the padding,
    /// like [`Trustmark::decode`](crate::Trustmark::decode).
    pub fn new() -> Self {
        Self::default()
    }

    /// Accept only the version named by the schema bits, and reject corrections in the padding.
    ///
    /// This is `DecodeOptions::new().with_fallback(false).with_padding_check(true)`. It rejects
    /// watermarks whose schema bits were damaged instead of guessing their version.
    pub fn strict() -> Self {
        Self::new().with_fallback(false).with_padding_check(true)
    }

    /// Accept only the given versions.
    ///
    /// Pass a single version when the watermark is known to have been embedded with it. Versions
    /// that are not allowed are still checked and reported in [`Decoded::candidates`], but never
    /// accepted. An empty set rejects every watermark.
    #[must_use]
    pub fn with_allowed_versions(mut self, versions: impl IntoIterator<Item = Version>) -> Self {
        self.allowed = versions
            .into_iter()
            .fold(0, |allowed, version| allowed | 1 << version.schema_id());
        self
    }

    /// Whether versions other than the one named by the schema bits may be accepted (default
    /// `true`).
    ///
    /// With `false`, only the version named by the four schema bits is accepted, and a watermark
    /// whose schema bits are not one of the four defined patterns (`0000` to `0011`) is rejected.
    /// The Python reference decoder reads only the last two schema bits and never falls back.
    #[must_use]
    pub fn with_fallback(mut self, fallback: bool) -> Self {
        self.fallback = fallback;
        self
    }

    /// Whether to reject bits that only pass the BCH check by correcting one of the zero bits
    /// that pad the payload to a whole byte (default `false`).
    ///
    /// No encoder sets those bits, so bits that need such a correction are further than the
    /// version's correction capacity from every packet an encoder can produce, and a watermark with
    /// no more bit flips than the capacity never needs one. The check lowers the share of random
    /// 100-bit strings that pass the default options from 10.1% to 8.7%.
    /// [`Trustmark::decode`](crate::Trustmark::decode) and the Python and JavaScript decoders
    /// accept these corrections; the C++ decoder rejects them. A version that fails the check is
    /// not listed in [`Decoded::candidates`].
    #[must_use]
    pub fn with_padding_check(mut self, padding_check: bool) -> Self {
        self.padding_check = padding_check;
        self
    }

    fn accepts(&self, version: Version, schema_version: Option<Version>) -> bool {
        self.allowed & 1 << version.schema_id() != 0
            && (self.fallback || schema_version == Some(version))
    }
}

impl Debug for DecodeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let allowed: Vec<Version> = [
            Version::BchSuper,
            Version::Bch5,
            Version::Bch4,
            Version::Bch3,
        ]
        .into_iter()
        .filter(|version| self.allowed & 1 << version.schema_id() != 0)
        .collect();
        f.debug_struct("DecodeOptions")
            .field("allowed_versions", &allowed)
            .field("fallback", &self.fallback)
            .field("padding_check", &self.padding_check)
            .finish()
    }
}

/// A watermark decoded by [`Trustmark::decode_with`](crate::Trustmark::decode_with).
///
/// Passing the BCH check is not proof that the image carries this watermark; see
/// [`DecodeOptions`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Decoded {
    /// The decoded payload bits, as returned by [`Trustmark::decode`](crate::Trustmark::decode).
    pub payload: String,
    /// The version the payload was decoded with.
    pub version: Version,
    /// How many bit flips the BCH decoder corrected in the payload and error correction bits.
    /// Flips in the four schema bits are not counted.
    pub corrected_bits: u32,
    /// The version named by the four schema bits, or `None` if they are not one of the defined
    /// patterns.
    pub schema_version: Option<Version>,
    /// Every version under which the bits pass the BCH check (and the padding check, if enabled),
    /// whether or not the options allow it, in the order they are tried: the version named by the
    /// schema bits (or [`Version::BchSuper`] when they name none) first, then [`Version::Bch3`],
    /// [`Version::Bch4`], [`Version::Bch5`] and [`Version::BchSuper`]. The accepted version is the
    /// first allowed one.
    pub candidates: Vec<Candidate>,
}

impl Decoded {
    /// Whether the accepted version differs from the one named by the schema bits, so it was found
    /// by trying other versions.
    pub fn used_fallback(&self) -> bool {
        self.schema_version != Some(self.version)
    }

    /// Whether the bits pass the BCH check under more than one version.
    ///
    /// Versions have different payload lengths, so each candidate has a different payload. Only
    /// the schema bits (or the caller's knowledge of the embedding version) tell them apart.
    pub fn is_ambiguous(&self) -> bool {
        self.candidates.len() > 1
    }
}

/// A version under which decoded bits pass the BCH check; see [`Decoded::candidates`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Candidate {
    /// The version.
    pub version: Version,
    /// The payload bits under this version.
    pub payload: String,
    /// How many bit flips the BCH decoder corrected under this version.
    pub corrected_bits: u32,
}

impl From<Bits> for Value<TensorValueType<f32>> {
    fn from(Bits(s): Bits) -> Self {
        let floats: Vec<f32> = s
            .chars()
            .map(|c| match c {
                '0' => 0.0,
                '1' => 1.0,
                _ => unreachable!(),
            })
            .collect();

        let array = Array1::from(floats);
        Value::from_array(array.insert_axis(Axis(0))).unwrap()
    }
}

impl TryFrom<ArrayD<f32>> for Bits {
    type Error = Error;

    fn try_from(array: ArrayD<f32>) -> Result<Self, Self::Error> {
        Bits::new(bitstring(array)?)
    }
}

/// The error correction schema
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Version {
    /// Tolerates 8 bit flips
    #[default]
    BchSuper,
    /// Tolerates 5 bit flips
    Bch5,
    /// Tolerates 4 bit flips
    Bch4,
    /// Tolerates 3 bit flips
    Bch3,
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Version::BchSuper => "BCH_SUPER",
            Version::Bch5 => "BCH_5",
            Version::Bch4 => "BCH_4",
            Version::Bch3 => "BCH_3",
        };
        write!(f, "{s}")
    }
}

impl FromStr for Version {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let version = match s {
            "BCH_SUPER" => Version::BchSuper,
            "BCH_5" => Version::Bch5,
            "BCH_4" => Version::Bch4,
            "BCH_3" => Version::Bch3,
            _ => return Err(Error::InvalidVersion),
        };

        Ok(version)
    }
}

impl Version {
    /// Get the number of allowed bit flips for this version.
    fn allowed_bit_flips(&self) -> u8 {
        match self {
            Version::BchSuper => 8,
            Version::Bch5 => 5,
            Version::Bch4 => 4,
            Version::Bch3 => 3,
        }
    }

    /// Get the number of data bits for this version.
    pub fn data_bits(&self) -> u16 {
        match self {
            Version::BchSuper => 40,
            Version::Bch5 => 61,
            Version::Bch4 => 68,
            Version::Bch3 => 75,
        }
    }

    /// Get the bitstring which indicates this version.
    fn bitstring(&self) -> String {
        match self {
            Version::BchSuper => "0000".to_owned(),
            Version::Bch5 => "0001".to_owned(),
            Version::Bch4 => "0010".to_owned(),
            Version::Bch3 => "0011".to_owned(),
        }
    }

    /// Parse a version from a bitstring.
    fn from_bitstring(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "0000" => Version::BchSuper,
            "0001" => Version::Bch5,
            "0010" => Version::Bch4,
            "0011" => Version::Bch3,
            _ => return Err(Error::InvalidVersion),
        })
    }

    /// The value of the schema bits that indicate this version.
    fn schema_id(&self) -> u8 {
        match self {
            Version::BchSuper => 0,
            Version::Bch5 => 1,
            Version::Bch4 => 2,
            Version::Bch3 => 3,
        }
    }

    /// Get the number of error correcting bits for this version.
    fn ecc_bits(&self) -> u16 {
        100 - VERSION_BITS - self.data_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_version() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits(input);
        assert_eq!(bits.get_version(), Version::Bch5);
    }

    #[test]
    fn get_data() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits(input);
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn new() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn fully_corrupted() {
        let input = "0000000000000000000000000000000000000000000100000110110110111000110010101101111010011011000010000001".to_owned();
        let err = Bits::new(input).unwrap_err();
        assert_eq!(err.to_string(), "corrupt watermark");
    }

    #[test]
    fn single_bitflip() {
        let input = "0011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn single_bitflip_and_corrupted_version() {
        let input = "0011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000011".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn invalid_bitstring() {
        let err = Bits::apply_error_correction_and_schema("hello".to_string(), Version::Bch5)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidChar));
    }

    #[test]
    fn too_long_input() {
        let err =
            Bits::apply_error_correction_and_schema("0".repeat(200), Version::Bch5).unwrap_err();
        assert!(matches!(err, Error::InvalidDataLength { .. }));
    }

    #[test]
    fn corrupt() {
        let err = Bits::new("1".repeat(100)).unwrap_err();
        assert!(matches!(err, Error::CorruptWatermark));
    }

    #[test]
    fn invalid_dim() {
        let ar = ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]));
        let res: Result<Bits, _> = ar.try_into();
        assert!(matches!(res.unwrap_err(), Error::InvalidDim));
    }

    /// Packets produced by the Python reference implementation (`python/trustmark/datalayer.py`),
    /// which the C++ implementation agrees with:
    /// `(version, payload, packet, packet with the maximum correctable number of bit flips)`.
    const PYTHON_PACKETS: &[(Version, &str, &str, &str)] = &[
        (
            Version::BchSuper,
            "1111111111111111111111111111111111111111",
            "1111111111111111111111111111111111111111110101111101100001100000101111101111000111001101010010010000",
            "1101111111111101111110111111111111111111110110111101100001110000101111101111001111001001010010010000",
        ),
        (
            Version::BchSuper,
            "1010001000011000100001000011001000100001",
            "1010001000011000100001000011001000100001111101100101001100110110100010100110100001001000100110000000",
            "1010001100011000000001000010001100101001111101100111001100110111100010100110100001001000100110100000",
        ),
        (
            Version::BchSuper,
            "1111110000111110010101100111110011001111",
            "1111110000111110010101100111110011001111111011010101111101001000101100110100110000101101011101110000",
            "1111110000011110010100100111110011011111111011010110111100001001101100010100110000101101011101110000",
        ),
        (
            Version::Bch5,
            "0111110000000010110011100111110110000100100000100010111100111",
            "0111110000000010110011100111110110000100100000100010111100111011000100000010000110001000000010010001",
            "0111110000000010110011100111110110000100100000100001111100101011000100000010000110001000000000000001",
        ),
        (
            Version::Bch5,
            "1100011100010010110101000100110011101111000010101011001010110",
            "1100011100010010110101000100110011101111000010101011001010110010011010011110101010011010011010000001",
            "1100011100110010010110000100110011101111000010101011001010110010011010011110101010011010011000000001",
        ),
        (
            Version::Bch4,
            "00101100000010001010111001110001000001001100001001001101110101010111",
            "0010110000001000101011100111000100000100110000100100110111010101011110010100100111101011111110110010",
            "0110110000001001101011100111000100000100110100100100110111110101011110010100100111101011111110110010",
        ),
        (
            Version::Bch4,
            "00100100101010011000111101101011101110000011001011101101001010010001",
            "0010010010101001100011110110101110111000001100101110110100101001000100010110010001011010001100100010",
            "0010010010101001100011110110101110011000001100101110100100101001000100110110010101011010001100100010",
        ),
        (
            Version::Bch3,
            "000001000110110110100001010111100100110000110001101010011011110010100110110",
            "0000010001101101101000010101111001001100001100011010100110111100101001101100010010111000000100000011",
            "0000010001101100101000010101111001001100001100011010100110101100101001001100010010111000000100000011",
        ),
        (
            Version::Bch3,
            "000100101011000111010010000000101100011010001101000111011000011100101100111",
            "0001001010110001110100100000001011000110100011010001110110000111001011001111001000011001001110010011",
            "0001001010010001110100100100001011000111100011010001110110000111001011001111001000011001001110010011",
        ),
    ];

    #[test]
    fn encode_matches_python() {
        for (version, payload, packet, _) in PYTHON_PACKETS {
            let bits =
                Bits::apply_error_correction_and_schema(payload.to_string(), *version).unwrap();
            assert_eq!(bits.0, *packet, "{version}");
        }
    }

    #[test]
    fn decode_python_packets() {
        for (version, payload, packet, corrupted) in PYTHON_PACKETS {
            for input in [packet, corrupted] {
                let bits = Bits::new(input.to_string()).unwrap();
                assert_eq!(bits.get_version(), *version);
                assert_eq!(bits.get_data(), *payload);
            }
        }
    }

    fn decode(s: &str, options: DecodeOptions) -> Result<Decoded, Error> {
        Bits::decode_with(s, &options)
    }

    /// Replace the four schema bits of `packet`.
    fn with_schema(packet: &str, schema: &str) -> String {
        format!("{}{schema}", &packet[..96])
    }

    #[test]
    fn default_options_match_new() {
        for (_, _, packet, corrupted) in PYTHON_PACKETS {
            for input in [packet, corrupted] {
                for schema in 0..16 {
                    let s = with_schema(input, &format!("{schema:04b}"));
                    let new =
                        Bits::new(s.clone()).map(|bits| (bits.get_version(), bits.get_data()));
                    let decoded =
                        decode(&s, DecodeOptions::default()).map(|d| (d.version, d.payload));
                    match (new, decoded) {
                        (Ok(new), Ok(decoded)) => assert_eq!(new, decoded, "{s}"),
                        (Err(Error::CorruptWatermark), Err(Error::CorruptWatermark)) => {}
                        (new, decoded) => panic!("{s}: {new:?} != {decoded:?}"),
                    }
                }
            }
        }
        assert!(matches!(
            decode(&"1".repeat(100), DecodeOptions::default()),
            Err(Error::CorruptWatermark)
        ));
        assert!(matches!(
            decode(&"2".repeat(100), DecodeOptions::default()),
            Err(Error::InvalidChar)
        ));
        assert!(matches!(
            decode(&"0".repeat(99), DecodeOptions::default()),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn decoded_reports_version_and_corrected_bits() {
        for (version, payload, packet, corrupted) in PYTHON_PACKETS {
            let clean = decode(packet, DecodeOptions::default()).unwrap();
            assert_eq!(clean.corrected_bits, 0);

            let decoded = decode(corrupted, DecodeOptions::default()).unwrap();
            assert_eq!(decoded.payload, *payload);
            assert_eq!(decoded.version, *version);
            assert_eq!(decoded.schema_version, Some(*version));
            assert!(!decoded.used_fallback());
            let corrected_bits = u32::from(version.allowed_bit_flips());
            assert_eq!(decoded.corrected_bits, corrected_bits);
            assert_eq!(
                decoded.candidates[0],
                Candidate {
                    version: *version,
                    payload: payload.to_string(),
                    corrected_bits,
                }
            );
        }
    }

    #[test]
    fn without_fallback_only_the_schema_bits_version_is_accepted() {
        let (_, payload, packet, corrupted) = PYTHON_PACKETS[3];
        let strict = DecodeOptions::new().with_fallback(false);
        let decoded = decode(corrupted, strict).unwrap();
        assert_eq!(decoded.payload, payload);
        assert!(!decoded.used_fallback());

        // The BCH_5 packet with its schema bits damaged into BCH_SUPER's.
        let damaged = with_schema(packet, "0000");
        let decoded = decode(&damaged, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.version, Version::Bch5);
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.schema_version, Some(Version::BchSuper));
        assert!(decoded.used_fallback());
        assert!(matches!(
            decode(&damaged, strict),
            Err(Error::CorruptWatermark)
        ));
    }

    #[test]
    fn undefined_schema_bits() {
        let (_, payload, packet, _) = PYTHON_PACKETS[3];
        let bch5 = DecodeOptions::new().with_allowed_versions([Version::Bch5]);
        let damaged = with_schema(packet, "0101");

        let decoded = decode(&damaged, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.schema_version, None);
        assert_eq!(decoded.version, Version::Bch5);
        assert_eq!(decoded.payload, payload);
        assert!(decoded.used_fallback());
        assert_eq!(decode(&damaged, bch5).unwrap().payload, payload);

        for strict in [
            DecodeOptions::new().with_fallback(false),
            bch5.with_fallback(false),
        ] {
            assert!(matches!(
                decode(&damaged, strict),
                Err(Error::CorruptWatermark)
            ));
        }
    }

    #[test]
    fn allowed_versions() {
        let (_, payload, packet, corrupted) = PYTHON_PACKETS[3];
        let bch5 = DecodeOptions::new().with_allowed_versions([Version::Bch5]);
        assert_eq!(decode(corrupted, bch5).unwrap().payload, payload);

        // Schema bits damaged into BCH_4's: allowing only BCH_5 still finds it, unless the schema
        // bits must agree.
        let damaged = with_schema(packet, "0010");
        let decoded = decode(&damaged, bch5).unwrap();
        assert_eq!(decoded.version, Version::Bch5);
        assert!(decoded.used_fallback());
        assert!(matches!(
            decode(&damaged, bch5.with_fallback(false)),
            Err(Error::CorruptWatermark)
        ));

        for others in [
            DecodeOptions::new().with_allowed_versions([Version::BchSuper, Version::Bch4]),
            DecodeOptions::new().with_allowed_versions([]),
        ] {
            assert!(matches!(
                decode(packet, others),
                Err(Error::CorruptWatermark)
            ));
        }

        assert_eq!(
            format!("{:?}", bch5.with_fallback(false)),
            "DecodeOptions { allowed_versions: [Bch5], fallback: false, padding_check: false }"
        );
        assert_eq!(DecodeOptions::new(), DecodeOptions::default());
    }

    /// A clean BCH_5 packet from the Python reference encoder that also passes the BCH_3 check,
    /// with a different payload: `(BCH_5 payload, packet, BCH_3 payload)`. Found by decoding 2000
    /// random BCH_5 packets from the reference encoder under every version; about 9% of BCH_5
    /// packets are ambiguous like this, almost all of them with BCH_3.
    const AMBIGUOUS: (&str, &str, &str) = (
        "1100011101011110010100100100010010111100000001111000001111001",
        "1100011101011110010100100100010010111100000001111000001111001111001000010000001110111101100100110001",
        "110001110101111011010010010001001011110000000111100000111100111100100001000",
    );

    #[test]
    fn ambiguous_packet() {
        let (payload, packet, bch3_payload) = AMBIGUOUS;
        let encoded =
            Bits::apply_error_correction_and_schema(payload.to_string(), Version::Bch5).unwrap();
        assert_eq!(encoded.0, packet);

        let decoded = decode(packet, DecodeOptions::default()).unwrap();
        assert_eq!(decoded.version, Version::Bch5);
        assert_eq!(decoded.payload, payload);
        assert!(decoded.is_ambiguous());
        assert_eq!(
            decoded.candidates,
            [
                Candidate {
                    version: Version::Bch5,
                    payload: payload.to_string(),
                    corrected_bits: 0,
                },
                Candidate {
                    version: Version::Bch3,
                    payload: bch3_payload.to_string(),
                    corrected_bits: 3,
                },
            ]
        );

        // With the schema bits damaged into BCH_SUPER's or an undefined pattern, the fallback tries
        // BCH_3 before BCH_5 and returns a payload that was never embedded.
        for schema in ["0000", "0100"] {
            let damaged = with_schema(packet, schema);
            assert_eq!(Bits::new(damaged.clone()).unwrap().get_data(), bch3_payload);
            let decoded = decode(&damaged, DecodeOptions::default()).unwrap();
            assert_eq!(decoded.version, Version::Bch3);
            assert_eq!(decoded.payload, bch3_payload);
            assert!(decoded.used_fallback());
            assert!(decoded.is_ambiguous());

            let bch5 = DecodeOptions::new().with_allowed_versions([Version::Bch5]);
            assert_eq!(decode(&damaged, bch5).unwrap().payload, payload);
            assert!(matches!(
                decode(&damaged, DecodeOptions::new().with_fallback(false)),
                Err(Error::CorruptWatermark)
            ));
        }
    }

    /// Bits whose only error is in the zero padding of the payload, from the cross-language
    /// conformance vectors (`pad_bit_correction`): `(version, bits, payload)`.
    const PADDING_CORRECTIONS: &[(Version, &str, &str)] = &[
        (
            Version::Bch5,
            "1110011100011110110110100100111001000000010110111000001101101010001100100010110001101001111101000001",
            "1110011100011110110110100100111001000000010110111000001101101",
        ),
        (
            Version::Bch4,
            "1110111000100100011111000111111010110110100010100001100010101010010100011111101101100101101000100010",
            "11101110001001000111110001111110101101101000101000011000101010100101",
        ),
        (
            Version::Bch3,
            "0001000011001111011001100110000100011010101101101101100010001010000001010101110101001111100010010011",
            "000100001100111101100110011000010001101010110110110110001000101000000101010",
        ),
    ];

    #[test]
    fn padding_check() {
        for (version, bits, payload) in PADDING_CORRECTIONS {
            let decoded = decode(bits, DecodeOptions::default()).unwrap();
            assert_eq!(decoded.version, *version);
            assert_eq!(decoded.payload, *payload);
            assert_eq!(decoded.corrected_bits, 1);
            for options in [
                DecodeOptions::new().with_padding_check(true),
                DecodeOptions::strict(),
            ] {
                assert!(matches!(
                    decode(bits, options),
                    Err(Error::CorruptWatermark)
                ));
            }
        }

        // A watermark with no more bit flips than the capacity never needs a padding correction.
        for (version, payload, packet, corrupted) in PYTHON_PACKETS {
            for input in [packet, corrupted] {
                let decoded = decode(input, DecodeOptions::strict()).unwrap();
                assert_eq!(decoded.version, *version);
                assert_eq!(decoded.payload, *payload);
            }
        }
    }
}
