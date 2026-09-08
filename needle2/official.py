"""Explicit, telemetry-free adapter for the official Needle 2 C ABI.

This is a benchmark oracle, not part of the independent runtime. It never
imports the official Python package or executes generated function calls.
The official ABI has one global session; use separate processes for concurrent
sessions. Network access happens only in the explicitly called fetch helper.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time
import urllib.request
import zipfile

HF_REPO = "Cactus-Compute/needle2"
HF_REVISION = "32e9e3a93b205f786929697446ae669cf0a84579"
ENGINE_VERSION = "2.0.4"
_SESSION_LOCK = threading.RLock()
_ACTIVE_SESSION = None
_WEIGHTS_KEEPALIVE = None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_equal(left, right) -> bool:
    """Ignore object key order, preserve array order and JSON scalar types."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
    return canonical(left) == canonical(right)


def platform_tag() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine not in ("aarch64", "arm64", "x86_64", "amd64"):
        raise RuntimeError(f"No configured official wheel for {machine}")
    arm = machine in ("aarch64", "arm64")
    system = platform.system()
    if system == "Darwin":
        return "macosx_11_0_" + ("arm64" if arm else "x86_64"), "libneedle.dylib"
    if system == "Windows":
        return ("win_arm64" if arm else "win_amd64"), "libneedle.dll"
    if system != "Linux":
        raise RuntimeError(f"Unsupported official wheel platform: {system}")
    libc, _ = platform.libc_ver()
    family = "musllinux_1_2_" if libc == "musl" else "manylinux2014_"
    return family + ("aarch64" if arm else "x86_64"), "libneedle.so"


def fetch_official_library(destination: str | Path, *, revision: str = HF_REVISION) -> Path:
    """Fetch a pinned wheel and extract only its shared library, without import."""
    tag, library_name = platform_tag()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wheel_name = f"cactus_needle-{ENGINE_VERSION}-py3-none-{tag}.whl"
    wheel = destination / wheel_name
    url = f"https://huggingface.co/{HF_REPO}/resolve/{revision}/python/{wheel_name}"
    # Include revision in metadata and never trust an unrelated cached wheel.
    metadata_path = destination / "official_library.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if (not wheel.exists() or metadata.get("url") != url
            or metadata.get("wheel_sha256") != sha256(wheel)):
        temporary = wheel.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(wheel)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read("needle/" + library_name)
    library = destination / library_name
    library.write_bytes(data)
    metadata = {"repository": HF_REPO, "revision": revision,
                "engine_version": ENGINE_VERSION, "platform_tag": tag,
                "url": url, "wheel_sha256": sha256(wheel),
                "library_sha256": sha256(library)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return library


class OfficialEngine:
    """One process-local official session loaded from a specified .cact file."""

    def __init__(self, library: str | Path, weights: str | Path, tools: list | str,
                 *, system: str = "", buffer_size: int = 1024 * 1024):
        global _ACTIVE_SESSION, _WEIGHTS_KEEPALIVE
        os.environ["NEEDLE_TELEMETRY"] = "0"
        os.environ["DO_NOT_TRACK"] = "1"
        self.library_path = Path(library).resolve()
        self.weights_path = Path(weights).resolve()
        if buffer_size < 1024:
            raise ValueError("buffer_size must be at least 1024 bytes")
        parsed = json.loads(tools) if isinstance(tools, str) else tools
        if not isinstance(parsed, list) or not all(isinstance(t, dict) for t in parsed):
            raise TypeError("tools must be a JSON array of schema objects")
        self.tools = parsed
        if not isinstance(system, str) or "\0" in system:
            raise ValueError("system must be a string without NUL characters")
        self.system = system
        self._closed = False
        self._weights = self.weights_path.read_bytes()  # needle_load may retain this pointer
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        self.lib.needle_load.restype = ctypes.c_int
        self.lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.needle_init.restype = ctypes.c_int
        self.lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                             ctypes.c_char_p, ctypes.c_int]
        self.lib.needle_complete.restype = ctypes.c_int
        self.lib.needle_reset.argtypes = []
        self.lib.needle_reset.restype = None
        with _SESSION_LOCK:
            if _ACTIVE_SESSION is not None:
                raise RuntimeError("Official ABI is global: close the existing session first")
            started = time.perf_counter()
            code = self.lib.needle_load(self._weights, len(self._weights))
            if code < 0:
                raise RuntimeError(f"needle_load failed: {code}")
            _WEIGHTS_KEEPALIVE = self._weights
            self.load_ms = 1000 * (time.perf_counter() - started)
            started = time.perf_counter()
            self.prefix_tokens = self.lib.needle_init(
                system.encode("utf-8"),
                json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                None)
            if self.prefix_tokens < 0:
                raise RuntimeError(f"needle_init failed: {self.prefix_tokens}")
            self.init_ms = 1000 * (time.perf_counter() - started)
            _ACTIVE_SESSION = self

    def _check(self):
        if self._closed or _ACTIVE_SESSION is not self:
            raise RuntimeError("Official session is closed")

    def reset(self):
        with _SESSION_LOCK:
            self._check()
            self.lib.needle_reset()

    def complete(self, query: str, max_new_tokens: int = 256) -> dict:
        if not isinstance(query, str) or "\0" in query:
            raise ValueError("query must be a string without NUL characters")
        if not 1 <= max_new_tokens <= 2**31 - 1:
            raise ValueError("max_new_tokens must be a positive C int")
        with _SESSION_LOCK:
            self._check()
            self._buffer[0] = 0
            started = time.perf_counter()
            code = self.lib.needle_complete(query.encode("utf-8"), max_new_tokens,
                                            self._buffer, len(self._buffer))
            elapsed = 1000 * (time.perf_counter() - started)
            raw = self._buffer.value.decode("utf-8", "strict")
            if code < 0:
                raise RuntimeError(f"needle_complete failed: {code}; {raw}")
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Official engine returned invalid JSON: {raw[:400]}") from exc
            if not isinstance(response, dict):
                raise RuntimeError("Official response must be a JSON object")
            return {"response": response, "wall_ms": elapsed, "return_code": code}

    def close(self):
        global _ACTIVE_SESSION
        with _SESSION_LOCK:
            if not self._closed and _ACTIVE_SESSION is self:
                self.lib.needle_reset()
                _ACTIVE_SESSION = None
            self._closed = True
            # The ABI exposes no destructor. The module keeps current weights
            # alive even after close until a later load replaces the native model.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
