// Copyright 2026 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

//! Checks `Bits` against the shared packet vectors in `tests/vectors` at the repository root
//! (see the README there).
//!
//! The vectors live outside the crate, so this module only builds in a repository checkout. It
//! is compiled for `cargo test` only, never for the published package's build.

use super::*;

const VECTORS: &str = include_str!("../../../tests/vectors/packets.tsv");

/// The tab-separated fields after the record type, for every record of type `kind`.
fn records(kind: &str) -> Vec<Vec<&'static str>> {
    let records: Vec<Vec<&str>> = VECTORS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|fields| fields[0] == kind)
        .map(|fields| fields[1..].to_vec())
        .collect();
    assert!(!records.is_empty(), "no {kind} vectors");
    records
}

#[test]
fn encode_vectors() {
    for record in records("encode") {
        let [id, schema, payload, packet, _python, rust, _cpp] = record[..] else {
            panic!("malformed encode vector: {record:?}");
        };
        let result =
            Bits::apply_error_correction_and_schema(payload.to_owned(), schema.parse().unwrap());
        match rust {
            "packet" => assert_eq!(result.unwrap().0, packet, "{id}"),
            "error:length" => assert!(
                matches!(result, Err(Error::InvalidDataLength { .. })),
                "{id}: {result:?}"
            ),
            "error:char" => assert!(
                matches!(result, Err(Error::InvalidChar)),
                "{id}: {result:?}"
            ),
            other => panic!("{id}: unknown expectation {other}"),
        }
    }
}

#[test]
fn decode_vectors() {
    for record in records("decode") {
        let [id, input, schema, data, valid_under, _python, rust, _cpp, _js] = record[..] else {
            panic!("malformed decode vector: {record:?}");
        };
        let result = Bits::new(input.to_owned())
            .map(|bits| (bits.get_version().to_string(), bits.get_data()));
        if let Ok((version, decoded)) = &result {
            // Every result a decoder may report is listed in `valid_under` as `schema=data`
            let found = format!("{version}={decoded}");
            assert!(
                valid_under.split(';').any(|entry| entry == found),
                "{id}: {found} is not listed in valid_under"
            );
        }
        let original = (schema.to_owned(), data.to_owned());
        match rust {
            "valid" => assert_eq!(result.unwrap(), original, "{id}"),
            "invalid" => assert!(
                matches!(result, Err(Error::CorruptWatermark)),
                "{id}: {result:?}"
            ),
            "not_original" => assert!(!result.is_ok_and(|r| r == original), "{id}"),
            other => panic!("{id}: unknown expectation {other}"),
        }
    }
}
