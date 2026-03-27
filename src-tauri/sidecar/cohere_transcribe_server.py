"""
Cohere Transcribe HTTP sidecar server for Handy.

Usage:
    python cohere_transcribe_server.py --model-path <path> --port <port>

Protocol:
    GET  /health        -> 200 "ok" (readiness check)
    POST /transcribe    -> JSON {"wav_path": str, "language": str}
                       <- JSON {"text": str}
"""
import argparse
import ctypes
import json
import os
import sys
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model = None
processor = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def to_short_path(path: str) -> str:
    """Convert a path to Windows 8.3 short form.

    SentencePiece's C++ backend cannot handle non-ASCII characters in paths on
    Windows.  GetShortPathNameW returns the legacy 8.3 path which is always
    ASCII, allowing sentencepiece to load tokenizer.model correctly.
    Falls back to the original path on non-Windows platforms.
    """
    if sys.platform != "win32":
        return path
    _GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathName.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathName.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(512)
    ret = _GetShortPathName(path, buf, 512)
    if ret == 0:
        return path  # fallback if conversion fails
    return buf.value


class TranscribeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/transcribe":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                wav_path = body["wav_path"]
                lang_raw = body.get("language") or None
                # Cohere Transcribe requires an explicit language code;
                # fall back to "ja" when the caller requests auto-detection.
                language = "ja" if lang_raw in (None, "auto") else lang_raw

                audio_array, sr = sf.read(wav_path, dtype="float32")

                texts = model.transcribe(
                    processor=processor,
                    audio_arrays=[audio_array],
                    sample_rates=[sr],
                    language=language,
                )

                text = texts[0] if texts else ""
                result = json.dumps({"text": text}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
            except Exception as e:
                error_body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default access logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    global model, processor
    # Normalize path separators, then convert to Windows 8.3 short path so
    # SentencePiece's C++ backend can handle non-ASCII directory names.
    model_path = to_short_path(os.path.normpath(args.model_path))
    print(f"Loading model from: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    ).to(DEVICE)
    model.eval()
    print(f"READY on port {args.port}", flush=True)

    server = HTTPServer(("127.0.0.1", args.port), TranscribeHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
