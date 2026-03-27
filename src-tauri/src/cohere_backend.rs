//! Cohere Transcribe sidecar backend.
//!
//! Spawns a Python HTTP server subprocess and communicates with it over
//! localhost HTTP to perform speech-to-text transcription.
//!
//! Uses stdlib TcpStream for HTTP (not reqwest::blocking) so it is safe to
//! call from within a Tokio async context.

use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

// Paths configured via environment variables.
// COHERE_PYTHON_EXE: path to the Python interpreter in the Cohere venv
// COHERE_MODEL_PATH: path to the Cohere model directory
fn python_exe() -> String {
    std::env::var("COHERE_PYTHON_EXE").expect("COHERE_PYTHON_EXE environment variable must be set")
}

fn model_path() -> String {
    std::env::var("COHERE_MODEL_PATH").expect("COHERE_MODEL_PATH environment variable must be set")
}

const LOCALHOST: &str = "127.0.0.1";

pub struct CohereTranscribeEngine {
    _process: Child,
    port: u16,
}

#[derive(Serialize)]
struct TranscribeRequest {
    wav_path: String,
    language: String,
}

#[derive(Deserialize)]
struct TranscribeResponse {
    text: String,
}

impl CohereTranscribeEngine {
    pub fn load(sidecar_script: &Path) -> Result<Self> {
        let port = find_free_port()?;

        let process = Command::new(python_exe())
            .arg(sidecar_script)
            .arg("--model-path")
            .arg(model_path())
            .arg("--port")
            .arg(port.to_string())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn Cohere sidecar: {}", e))?;

        // Poll /health until ready (model load can take up to 2 minutes).
        let deadline = Instant::now() + Duration::from_secs(180);

        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("Cohere sidecar timed out (model load > 3 min)"));
            }
            if http_get_ok(LOCALHOST, port, "/health", Duration::from_secs(5)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }

        info!("Cohere Transcribe sidecar ready on port {}", port);
        Ok(Self {
            _process: process,
            port,
        })
    }

    pub fn transcribe(&self, audio: &[f32], language: &str) -> Result<String> {
        // Write audio to a temp WAV file (f32 IEEE float, 16kHz mono).
        let tmp = tempfile::NamedTempFile::new()?;
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        {
            let mut writer = WavWriter::create(tmp.path(), spec)?;
            for &sample in audio {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }

        let wav_path = tmp.path().to_string_lossy().to_string();
        debug!("Cohere: temp WAV path = {}, samples = {}", wav_path, audio.len());

        let req = TranscribeRequest {
            wav_path,
            language: language.to_string(),
        };
        let body = serde_json::to_string(&req)?;

        let response_body =
            http_post_json(LOCALHOST, self.port, "/transcribe", &body, Duration::from_secs(120))
                .map_err(|e| {
                    error!("Cohere sidecar HTTP error: {}", e);
                    anyhow!("Cohere sidecar HTTP error: {}", e)
                })?;

        debug!("Cohere: raw response = {}", response_body);

        let resp: TranscribeResponse = serde_json::from_str(&response_body)
            .map_err(|e| anyhow!("Cohere sidecar JSON parse error: {}", e))?;

        info!("Cohere: transcription result = {:?}", resp.text);
        Ok(resp.text)
    }
}

impl Drop for CohereTranscribeEngine {
    fn drop(&mut self) {
        let _ = self._process.kill();
    }
}

fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow!("Failed to find free port: {}", e))?;
    Ok(listener.local_addr()?.port())
}

/// Send a GET request and return true if the server responds with HTTP 200.
/// Returns false on any error (connection refused, timeout, etc.).
fn http_get_ok(host: &str, port: u16, path: &str, timeout: Duration) -> bool {
    let addr: SocketAddr = match format!("{}:{}", host, port).parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let mut stream = match TcpStream::connect_timeout(&addr, timeout) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));

    let request = format!(
        "GET {} HTTP/1.0\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        path, host, port
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200")
}

/// Send a POST request with a JSON body and return the response body as a String.
fn http_post_json(
    host: &str,
    port: u16,
    path: &str,
    json_body: &str,
    timeout: Duration,
) -> Result<String> {
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| anyhow!("Invalid address: {}", e))?;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| anyhow!("Connection failed: {}", e))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| anyhow!("set_read_timeout failed: {}", e))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| anyhow!("set_write_timeout failed: {}", e))?;

    let body_bytes = json_body.as_bytes();
    let header = format!(
        "POST {} HTTP/1.0\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        path,
        host,
        port,
        body_bytes.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|e| anyhow!("Write header failed: {}", e))?;
    stream
        .write_all(body_bytes)
        .map_err(|e| anyhow!("Write body failed: {}", e))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| anyhow!("Read response failed: {}", e))?;

    let response = String::from_utf8_lossy(&raw);

    // Verify HTTP 200; include response body in error for diagnosis
    if !response.starts_with("HTTP/1.0 200") && !response.starts_with("HTTP/1.1 200") {
        let status = response.lines().next().unwrap_or("").to_string();
        let body = response
            .find("\r\n\r\n")
            .map(|p| response[p + 4..].trim().to_string())
            .unwrap_or_default();
        return Err(anyhow!("HTTP error: {} — {}", status, body));
    }

    // Extract body after the blank line separating headers from body
    if let Some(pos) = response.find("\r\n\r\n") {
        Ok(response[pos + 4..].to_string())
    } else {
        Err(anyhow!("Invalid HTTP response: no header/body separator"))
    }
}
