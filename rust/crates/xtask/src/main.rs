// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{
    fs::{self, File},
    io::{self, Read as _},
    path::Path,
    process::ExitCode,
};

use clap::Parser;

/// Downloads larger than this are rejected instead of being written to disk.
const MAX_MODEL_BYTES: u64 = 100_000_000;

#[derive(Debug, Parser)]
enum Args {
    FetchModels,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args {
        Args::FetchModels => fetch_models(),
    };
    if let Err(err) = result {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Fetch all known models.
fn fetch_models() -> Result<(), String> {
    fetch_model("decoder_Q.onnx")?;
    fetch_model("encoder_Q.onnx")?;
    fetch_model("decoder_P.onnx")?;
    fetch_model("encoder_P.onnx")?;
    fetch_model("decoder_B.onnx")?;
    fetch_model("encoder_B.onnx")?;
    fetch_model("decoder_C.onnx")?;
    fetch_model("encoder_C.onnx")?;
    Ok(())
}

/// Fetch a single model identified by `filename`.
///
/// Models are fetched from a hardcoded CDN URL. Each download is written to a `.part` file that
/// is only renamed into place once it is complete, so models that already exist are skipped.
fn fetch_model(filename: &str) -> Result<(), String> {
    let root = "https://cai-watermark.adobe.net/watermarking/trustmark-models";
    let model_url = format!("{root}/{filename}");
    let path = Path::new("models").join(filename);
    if path.exists() {
        println!("{} already exists, skipping", path.display());
        return Ok(());
    }

    println!("fetching {model_url}");
    let part_path = Path::new("models").join(format!("{filename}.part"));
    let result = download(&model_url, &part_path).and_then(|()| {
        fs::rename(&part_path, &path)
            .map_err(|err| format!("failed to rename to {}: {err}", path.display()))
    });
    if result.is_err() {
        // Don't leave a partial download behind.
        let _ = fs::remove_file(&part_path);
    }
    result.map_err(|err| format!("failed to fetch {filename}: {err}"))
}

/// Download `url` to a new file at `path`.
fn download(url: &str, path: &Path) -> Result<(), String> {
    let response = ureq::get(url).call().map_err(|err| err.to_string())?;
    let expected_len: Option<u64> = response
        .header("Content-Length")
        .and_then(|len| len.parse().ok());
    if let Some(len) = expected_len.filter(|len| *len > MAX_MODEL_BYTES) {
        return Err(format!(
            "model is {len} bytes, larger than the {MAX_MODEL_BYTES} byte limit"
        ));
    }

    let mut file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    // Read one byte past the limit so that oversized responses are detected, not truncated.
    let mut reader = response.into_reader().take(MAX_MODEL_BYTES + 1);
    let written = io::copy(&mut reader, &mut file)
        .map_err(|err| format!("failed to download to {}: {err}", path.display()))?;
    if written > MAX_MODEL_BYTES {
        return Err(format!(
            "model is larger than the {MAX_MODEL_BYTES} byte limit"
        ));
    }
    if let Some(len) = expected_len.filter(|len| *len != written) {
        return Err(format!("expected {len} bytes but received {written}"));
    }
    file.sync_all()
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}
