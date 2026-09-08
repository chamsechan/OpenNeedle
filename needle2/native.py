"""ctypes access to the independently implemented, directly packed CQ CPU kernels.

Compilation is cached outside the source tree. ``CXX`` selects the compiler;
``NEEDLE2_NATIVE_CACHE`` selects the cache directory. ``NEEDLE2_NATIVE_LIBRARY``
selects a prebuilt OpenNeedle library and bypasses compilation.
No closed library is loaded.
"""
from __future__ import annotations

import ctypes as ct
import functools
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile

import numpy as np


@functools.lru_cache(maxsize=1)
def build_native() -> Path:
    prebuilt = os.environ.get("NEEDLE2_NATIVE_LIBRARY")
    if prebuilt is not None:
        if not prebuilt.strip():
            raise RuntimeError("NEEDLE2_NATIVE_LIBRARY must name a prebuilt OpenNeedle shared library")
        target = Path(prebuilt).expanduser().resolve()
        if not target.is_file():
            raise RuntimeError(f"Prebuilt OpenNeedle library not found: {target}")
        return target
    source = Path(__file__).resolve().parent / "csrc" / "cq.cpp"
    if not source.is_file():
        raise RuntimeError(f"Native source not found: {source}")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("CXX must name a C++17 compiler with OpenMP support")
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared", "-pthread"]
    if platform.system() == "Darwin":
        import sys
        omp_candidates = [
            (Path(sys.prefix) / "include", Path(sys.prefix) / "lib"),
            (Path("/opt/homebrew/opt/libomp/include"), Path("/opt/homebrew/opt/libomp/lib")),
            (Path("/usr/local/opt/libomp/include"), Path("/usr/local/opt/libomp/lib")),
        ]
        omp_found = False
        for inc, lib in omp_candidates:
            if (inc / "omp.h").exists() and any(lib.glob("libomp*.dylib")):
                flags += ["-Xpreprocessor", "-fopenmp", f"-I{inc}", f"-L{lib}", "-lomp"]
                omp_found = True
                break
        if not omp_found:
            flags.append("-fopenmp")
        sdk_includes = list(Path("/Library/Developer/CommandLineTools/SDKs").glob("MacOSX*.sdk/usr/include/c++/v1"))
        if sdk_includes:
            flags.append(f"-isystem{sorted(sdk_includes)[-1]}")
    else:
        flags.append("-fopenmp")
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    key = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(source.parent.glob("*.cpp"))) + repr((compiler, flags, platform.machine())).encode()).hexdigest()[:20]
    cache = Path(os.environ.get("NEEDLE2_NATIVE_CACHE", str(Path.home() / ".cache" / "needle2")))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"libneedle2-{key}{suffix}"
    if not target.exists():
        fd, temporary = tempfile.mkstemp(prefix="build-", suffix=".so", dir=cache)
        os.close(fd)
        try:
            try:
                result = subprocess.run(compiler + flags + [str(source), "-o", temporary], capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Native compiler not found: {compiler[0]}. Install a C++17/OpenMP compiler, "
                    "set CXX, or load a prebuilt library with NEEDLE2_NATIVE_LIBRARY."
                ) from exc
            if result.returncode:
                raise RuntimeError(f"Native CQ compilation failed:\n{result.stderr}")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


@functools.lru_cache(maxsize=1)
def _library():
    lib = ct.CDLL(str(build_native()))
    ptr = ct.c_void_p
    lib.needle2_native_features.restype = ct.c_char_p
    lib.needle2_cq_create.argtypes = [ptr, ptr, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ptr]
    lib.needle2_cq_create.restype = ptr
    lib.needle2_cq_destroy.argtypes = [ptr]
    lib.needle2_cq_destroy.restype = None
    lib.needle2_cq_linear.argtypes = [ptr, ptr, ptr, ct.c_int, ct.c_int]
    lib.needle2_cq_linear.restype = None
    lib.needle2_cq_linear_lookup.argtypes = [ptr, ptr, ptr, ct.c_int, ct.c_int, ct.c_int]
    lib.needle2_cq_linear_lookup.restype = None
    lib.needle2_sdot_supported.restype = ct.c_int
    lib.needle2_sdot_create.argtypes = [ptr, ptr, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ptr]
    lib.needle2_sdot_create.restype = ptr
    lib.needle2_sdot_destroy.argtypes = [ptr]
    lib.needle2_sdot_destroy.restype = None
    lib.needle2_sdot_linear.argtypes = [ptr, ptr, ptr, ct.c_int, ct.c_int]
    lib.needle2_sdot_linear.restype = None
    lib.needle2_cq_rows.argtypes = [ptr, ptr, ptr, ct.c_int]
    lib.needle2_cq_rows.restype = None
    return lib


def available() -> bool:
    try:
        _library()
        return True
    except (OSError, RuntimeError):
        return False


def features() -> str:
    return _library().needle2_native_features().decode()


def sdot_available() -> bool:
    return bool(_library().needle2_sdot_supported())


class NativeCQ:
    """A packed ``[out, in]`` matrix; owns references to all borrowed buffers.

    The codebook must already include ``1/sqrt(group_size)`` as in a CACT header.
    ``bits=5`` denotes the official signed ternary crumb storage, with 3 centroids.
    """

    def __init__(self, packed, norms, shape, bits, group_size, codebook):
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("NativeCQ requires a positive two-dimensional matrix shape")
        self.bits, self.group_size = int(bits), int(group_size)
        if self.bits not in (2, 3, 4, 5):
            raise ValueError("CQ storage bits must be 2, 3, 4, or ternary marker 5")
        if self.group_size < 8 or self.group_size & (self.group_size - 1):
            raise ValueError("CQ group size must be a power of two >= 8")
        out, dim = self.shape
        limit = np.iinfo(np.int32).max
        if max(out, dim, self.group_size) > limit or dim + self.group_size - 1 > limit:
            raise ValueError("CQ dimensions exceed the native int32 interface")
        padded = ((dim + self.group_size - 1) // self.group_size) * self.group_size
        storage_bits = 2 if self.bits == 5 else self.bits
        if padded * storage_bits > limit:
            raise ValueError("CQ row bit count exceeds the native int32 interface")
        self.packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(out, padded * storage_bits // 8)
        self.norms = np.ascontiguousarray(norms, dtype=np.float16).reshape(out, padded // self.group_size)
        self.codebook = np.ascontiguousarray(codebook, dtype=np.float32)
        if self.codebook.shape != (3 if self.bits == 5 else 1 << self.bits,):
            raise ValueError("Codebook has incorrect number of centroids")
        if not np.isfinite(self.norms).all() or np.any(self.norms < 0):
            raise ValueError("CQ norms must be finite and nonnegative")
        if not np.isfinite(self.codebook).all() or not np.all(np.diff(self.codebook) > 0):
            raise ValueError("CQ codebook must be finite and strictly sorted")
        if self.bits == 5:
            # Crumb 2 is reserved and would index beyond the ternary codebook.
            for shift in (0, 2, 4, 6):
                if np.any(((self.packed >> shift) & 3) == 2):
                    raise ValueError("Invalid ternary crumb code 2")
        self._lib = _library()
        self._handle = self._lib.needle2_cq_create(self.packed.ctypes.data, self.norms.ctypes.data,
                                                   out, dim, self.bits, self.group_size,
                                                   self.codebook.ctypes.data)
        if not self._handle:
            raise MemoryError("Could not construct native CQ matrix")

    @classmethod
    def from_record(cls, record):
        cb = record.codebook
        if record.bits == 5 and cb is None:
            cb = (np.array([-1.2240064, 0., 1.2240064], np.float64) / np.sqrt(record.group_size)).astype(np.float32)
        return cls(record.packed, record.norms, record.shape, record.bits, record.group_size, cb)

    def __del__(self):
        if getattr(self, "_sdot_handle", None):
            self._lib.needle2_sdot_destroy(self._sdot_handle)
            self._sdot_handle = None
        if getattr(self, "_handle", None):
            self._lib.needle2_cq_destroy(self._handle)
            self._handle = None

    def linear(self, x, threads: int = 1, *, lookup: int = 0):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim < 1 or x.shape[-1] != self.shape[1]:
            raise ValueError(f"Expected input last dimension {self.shape[1]}; got {x.shape}")
        if not 1 <= int(threads) <= 256:
            raise ValueError("threads must be between 1 and 256")
        if (x.size // self.shape[1]) * self.shape[0] > np.iinfo(np.int32).max:
            raise ValueError("CQ batch output count exceeds the native int32 interface")
        y = np.empty(x.shape[:-1] + (self.shape[0],), dtype=np.float32)
        if lookup not in (0, 1, 2, 3):
            raise ValueError("lookup must be 0 (NEON), 1 (byte table), 2 (nibble table), or 3 (group byte table)")
        if lookup:
            self._lib.needle2_cq_linear_lookup(self._handle, x.ctypes.data, y.ctypes.data,
                                                x.size // self.shape[1], int(threads), int(lookup))
        else:
            self._lib.needle2_cq_linear(self._handle, x.ctypes.data, y.ctypes.data,
                                        x.size // self.shape[1], int(threads))
        return y

    def linear_sdot(self, x, threads: int = 1):
        """Approximate DotProd GEMV; additionally rounds rotated inputs and centroids to INT8."""
        if self.bits not in (2, 4) or not 64 <= self.group_size <= 131072:
            raise ValueError("SDOT requires CQ2/CQ4 and group size between 64 and 131072")
        if not sdot_available():
            raise RuntimeError("SDOT requires an ARM64 CPU with DotProd support")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim < 1 or x.shape[-1] != self.shape[1] or not np.isfinite(x).all():
            raise ValueError("SDOT input must have the expected last dimension and finite values")
        if not 1 <= int(threads) <= 256:
            raise ValueError("threads must be between 1 and 256")
        if not getattr(self, "_sdot_handle", None):
            self._sdot_handle = self._lib.needle2_sdot_create(self.packed.ctypes.data, self.norms.ctypes.data,
                                                              *self.shape, self.bits, self.group_size, self.codebook.ctypes.data)
            if not self._sdot_handle:
                raise RuntimeError("could not construct SDOT matrix")
        if (x.size // self.shape[1]) * self.shape[0] > np.iinfo(np.int32).max:
            raise ValueError("SDOT batch output count exceeds the native int32 interface")
        y = np.empty(x.shape[:-1] + (self.shape[0],), dtype=np.float32)
        self._lib.needle2_sdot_linear(self._sdot_handle, x.ctypes.data, y.ctypes.data, x.size // self.shape[1], int(threads))
        return y

    def rows(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu":
            raise TypeError("row indices must be integers")
        ids_shape = ids.shape
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if np.any((ids < 0) | (ids >= self.shape[0])):
            raise IndexError("CQ row index outside matrix")
        result = np.empty(ids_shape + (self.shape[1],), dtype=np.float32)
        self._lib.needle2_cq_rows(self._handle, ids.ctypes.data, result.ctypes.data, ids.size)
        return result

    def dequantize(self):
        return self.rows(np.arange(self.shape[0], dtype=np.int64))


class _TensorDesc(ct.Structure):
    _fields_ = [("cq", ct.c_void_p), ("data", ct.c_void_p), ("rows", ct.c_int), ("cols", ct.c_int)]


class _EngineConfig(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ("vocab", "dim", "heads", "kvheads", "layers", "head_dim", "max_seq", "hada", "lanes", "window", "slots", "subdim", "tables", "taps", "dilation", "num_orders")]
    _fields_ += [("orders", ct.c_int * 4), ("num_sites", ct.c_int), ("sites", ct.c_int * 4), ("rope_theta", ct.c_float)]


class _DFAStateDesc(ct.Structure):
    _fields_ = [
        ("num_states", ct.c_int),
        ("initial_state", ct.c_int),
        ("eos_id", ct.c_int),
        ("stop_id", ct.c_int),
        ("tool_start_id", ct.c_int),
        ("tool_end_id", ct.c_int),
        ("state_types", ct.POINTER(ct.c_int)),
        ("fallback_next_states", ct.POINTER(ct.c_int)),
        ("candidate_offsets", ct.POINTER(ct.c_int)),
        ("candidate_tokens", ct.POINTER(ct.c_int)),
        ("next_states", ct.POINTER(ct.c_int)),
    ]


class NativeEngine:
    """Entire CPU token forward in C++; one independent unpadded sequence per instance.

    FP32 activation and KV arithmetic matches the public JAX decode reference.
    Set ``activation_bits=8`` for the public fake-quantized activation path.
    This does not claim bitwise equivalence with the proprietary int8 runtime.
    Calls mutate the KV/history state; do not call the same instance concurrently.
    """

    def __init__(self, archive, threads: int = 1, activation_bits: int = 0, projection_lookup: int = -1, matmul: str = "fp32", kv_cache: str = "fp32"):
        from .archive import Archive, CQ, FP16, FP32, LAYER_NAMES, MHC_NAMES
        if isinstance(archive, (str, Path)):
            archive = Archive.load(archive)
        if not 1 <= int(threads) <= 256:
            raise ValueError("threads must be between 1 and 256")
        if kv_cache not in ("fp32", "int8"):
            raise ValueError("kv_cache must be 'fp32' or 'int8'")
        self.kv_cache = kv_cache
        if matmul not in ("fp32", "sdot"):
            raise ValueError("matmul must be fp32 or sdot")
        if matmul == "sdot" and activation_bits:
            raise ValueError("SDOT already rounds rotated activations; use activation_bits=0")
        if projection_lookup not in (-1, 0, 1, 2, 3, 4):
            raise ValueError("projection_lookup must be -1 (auto), 0, 1, 2, 3, or 4")
        if activation_bits not in (0, 8):
            raise ValueError("activation_bits must be 0 (FP32) or 8")
        self._archive = archive
        self._threads, self._activation_bits = int(threads), activation_bits
        self.matmul = matmul
        self._prefill_model = None
        self.prefix_len = 0
        m = self.metadata = archive.metadata.copy()
        if m["kv_bits"] != 8:
            raise NotImplementedError("NativeEngine currently supports KV8 archives with FP32 reference KV arithmetic")
        if m["head_dim"] % 2 or m["hada_n"] < m["d_model"] or m["hada_n"] & (m["hada_n"] - 1):
            raise ValueError("invalid native head/Hadamard geometry")
        if m["engram_layers"] and (not m["engram_orders"] or m["num_engram_tables"] % len(m["engram_orders"]) or m["num_engram_tables"] * m["engram_sub_dim"] != m["d_model"]):
            raise ValueError("inconsistent engram geometry")
        d, n, layers = m["d_model"], m["mhc_lanes"], m["num_layers"]
        attn, kv = m["num_heads"] * m["head_dim"], m["num_kv_heads"] * m["head_dim"]
        expected = {"embedding": (m["vocab_size"], d)}
        shapes = [(d,), (attn, d), (kv, d), (kv, d), (m["head_dim"],),
                  (m["head_dim"],), (attn, d), (d, attn), (d,), (1,), (d,),
                  (m["hada_n"],), (m["hada_n"],), (m["hada_n"],)]
        for layer in range(layers):
            expected.update({f"layer{layer:02d}.{name}": shape for name, shape in zip(LAYER_NAMES, shapes)})
        mhc_shapes = [(layers,)] * 3 + [(layers, n)] * 2 + [(layers, n, n), (layers*n, n*d), (layers*n, n*d), (layers*n*n, n*d)]
        expected.update(dict(zip(MHC_NAMES, mhc_shapes)))
        for site in range(len(m["engram_layers"])):
            expected.update({f"engram{site}.tables": (m["num_engram_tables"]*m["engram_slots"], m["engram_sub_dim"]),
                             f"engram{site}.key_proj": (d, d), f"engram{site}.value_proj": (d, d),
                             f"engram{site}.taps": (m["engram_conv_taps"], d)})
        expected["final_norm"] = (d,)
        for name, shape in expected.items():
            rec = archive.tensors.get(name)
            if rec is None or rec.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {None if rec is None else rec.shape}")
            if rec.dtype not in (FP16, FP32, CQ):
                raise ValueError(f"{name}: expected a numeric tensor")
            is_matrix = name == "embedding" or name.startswith("mhc_phi") or name.endswith(("_proj", ".tables"))
            if rec.dtype == CQ and not is_matrix:
                raise ValueError(f"{name}: CQ is only supported for matrix tensors")
        config = _EngineConfig()
        mapping = dict(vocab="vocab_size", dim="d_model", heads="num_heads", kvheads="num_kv_heads", layers="num_layers", head_dim="head_dim", max_seq="max_seq_len", hada="hada_n", lanes="mhc_lanes", window="kv_window", slots="engram_slots", subdim="engram_sub_dim", tables="num_engram_tables", taps="engram_conv_taps", dilation="engram_conv_dilation")
        for field, key in mapping.items():
            setattr(config, field, int(m[key]))
        config.num_orders, config.num_sites = len(m["engram_orders"]), len(m["engram_layers"])
        config.orders = (ct.c_int * 4)(*m["engram_orders"])
        config.sites = (ct.c_int * 4)(*m["engram_layers"])
        config.rope_theta = m["rope_theta"]
        self._owners = []
        descriptors = []
        for name in expected:
            rec = archive.tensors[name]
            if rec.dtype == CQ and not name.startswith("mhc_"):
                q = NativeCQ.from_record(rec)
                self._owners.append(q)
                descriptors.append(_TensorDesc(q._handle, None, *rec.shape))
            else:
                a = np.ascontiguousarray(rec.dequantize(), dtype=np.float32)
                self._owners.append(a)
                rows, cols = (a.shape[0], int(np.prod(a.shape[1:]))) if a.ndim > 1 else (1, a.size)
                descriptors.append(_TensorDesc(None, a.ctypes.data, rows, cols))
        self._descriptors = (_TensorDesc * len(descriptors))(*descriptors)
        self._lib = lib = _library()
        lib.needle2_engine_create.argtypes = [ct.POINTER(_EngineConfig), ct.POINTER(_TensorDesc), ct.c_int, ct.c_int, ct.c_int]
        lib.needle2_engine_create.restype = ct.c_void_p
        lib.needle2_engine_destroy.argtypes = [ct.c_void_p]
        lib.needle2_engine_reset.argtypes = [ct.c_void_p, ct.c_int]
        lib.needle2_engine_step.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        lib.needle2_engine_step.restype = ct.c_int
        lib.needle2_engine_error.restype = ct.c_char_p
        self._handle = lib.needle2_engine_create(ct.byref(config), self._descriptors, len(descriptors), int(threads), int(activation_bits))
        if not self._handle:
            raise RuntimeError(lib.needle2_engine_error().decode())
        lib.needle2_engine_set_lookup.argtypes = [ct.c_void_p, ct.c_int]
        lib.needle2_engine_set_lookup(self._handle, int(projection_lookup))
        if matmul == "sdot":
            lib.needle2_engine_set_sdot.argtypes = [ct.c_void_p]
            lib.needle2_engine_set_sdot.restype = ct.c_int
            if lib.needle2_engine_set_sdot(self._handle):
                raise RuntimeError(lib.needle2_engine_error().decode())
        if kv_cache == "int8":
            self.set_int8_kv(True)
        self.position = 0

    def set_int8_kv(self, enabled: bool = True):
        lib = self._lib
        lib.needle2_engine_set_int8_kv.argtypes = [ct.c_void_p, ct.c_int]
        lib.needle2_engine_set_int8_kv.restype = ct.c_int
        if lib.needle2_engine_set_int8_kv(self._handle, int(bool(enabled))):
            raise RuntimeError(lib.needle2_engine_error().decode())
        self.kv_cache = "int8" if enabled else "fp32"

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.needle2_engine_destroy(self._handle)
            self._handle = None

    def reset(self, prefix_len: int = 0):
        if not 0 <= prefix_len <= self.metadata["max_seq_len"]:
            raise ValueError("prefix_len must lie between 0 and max_seq_len")
        self._lib.needle2_engine_reset(self._handle, int(prefix_len))
        self.position = 0
        self.prefix_len = int(prefix_len)

    def cache_prefix(self):
        """Snapshot the complete pinned prefix once, for reuse by this instance.

        Requires ``position == prefix_len > 0``. Copies prefix KV, token history,
        and the short Engram ring. Ordinary ``reset()`` discards the snapshot.
        """
        if self.position <= 0 or self.position != self.prefix_len:
            raise ValueError("prefix snapshot requires position == prefix_len > 0")
        lib = self._lib
        lib.needle2_engine_cache_prefix.argtypes = [ct.c_void_p]
        lib.needle2_engine_cache_prefix.restype = ct.c_int
        if lib.needle2_engine_cache_prefix(self._handle):
            raise RuntimeError(lib.needle2_engine_error().decode())

    def reset_to_prefix(self):
        """Restore this instance's explicit prefix snapshot without a forward pass."""
        lib = self._lib
        lib.needle2_engine_reset_to_prefix.argtypes = [ct.c_void_p]
        lib.needle2_engine_reset_to_prefix.restype = ct.c_int
        position = lib.needle2_engine_reset_to_prefix(self._handle)
        if position < 0:
            raise RuntimeError(lib.needle2_engine_error().decode())
        self.position = self.prefix_len = position

    def step(self, token_id: int, *, return_hidden: bool = False, compute_logits: bool = True, candidates: list[int] | None = None):
        if candidates is not None:
            return self.step_candidates(token_id, candidates, return_hidden=return_hidden)
        if not isinstance(token_id, (int, np.integer)) or not 0 <= token_id < self.metadata["vocab_size"]:
            raise ValueError("token_id must be an integer within the vocabulary")
        logits = np.empty(self.metadata["vocab_size"], dtype=np.float32) if compute_logits else None
        hidden = np.empty((self.metadata["num_layers"], self.metadata["d_model"]), dtype=np.float32) if return_hidden else None
        status = self._lib.needle2_engine_step(self._handle, int(token_id), logits.ctypes.data if logits is not None else None,
                                               hidden.ctypes.data if hidden is not None else None)
        if status:
            raise RuntimeError(self._lib.needle2_engine_error().decode())
        self.position += 1
        return (logits, hidden) if return_hidden else logits

    def step_candidates(self, token_id: int, candidates, *, return_hidden: bool = False):
        """Forward a token and project only the specified candidate token logits."""
        if not isinstance(token_id, (int, np.integer)) or not 0 <= token_id < self.metadata["vocab_size"]:
            raise ValueError("token_id must be an integer within the vocabulary")
        cands = np.ascontiguousarray(candidates, dtype=np.int32)
        if cands.ndim != 1 or len(cands) == 0:
            raise ValueError("candidates must be a nonempty 1D array of token IDs")
        if len(cands) == 1:
            hidden = self.step(token_id, compute_logits=False, return_hidden=return_hidden)
            logits = np.array([0.0], dtype=np.float32)
            return (logits, hidden) if return_hidden else logits
        logits = np.empty(len(cands), dtype=np.float32)
        hidden = np.empty((self.metadata["num_layers"], self.metadata["d_model"]), dtype=np.float32) if return_hidden else None
        lib = self._lib
        lib.needle2_engine_step_candidates.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        lib.needle2_engine_step_candidates.restype = ct.c_int
        status = lib.needle2_engine_step_candidates(self._handle, int(token_id), cands.ctypes.data, len(cands),
                                                    logits.ctypes.data, hidden.ctypes.data if hidden is not None else None)
        if status:
            raise RuntimeError(self._lib.needle2_engine_error().decode())
        self.position += 1
        return (logits, hidden) if return_hidden else logits

    def prefill(self, token_ids, *, last_only: bool = False, backend: str = "native"):
        """Append tokens; ``last_only`` skips unnecessary vocabulary projections.

        ``backend="torch"`` batches an initial prompt through the dense PyTorch
        reference and imports its state. It retains a dense model for reuse;
        call :meth:`release_prefill_model` to recover that memory after prefill.
        """
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError("prefill requires a nonempty one-dimensional integer token sequence")
        if np.any((ids < 0) | (ids >= self.metadata["vocab_size"])):
            raise ValueError("token outside vocabulary")
        if backend == "torch":
            return self._torch_prefill(ids, last_only)
        if backend != "native":
            raise ValueError("prefill backend must be 'native' or 'torch'")
        ids_c = np.ascontiguousarray(ids, dtype=np.int32)
        num_tokens = int(ids.size)
        vocab_size = int(self.metadata["vocab_size"])
        if last_only:
            logits = np.empty(vocab_size, dtype=np.float32)
        else:
            logits = np.empty((num_tokens, vocab_size), dtype=np.float32)
        lib = self._lib
        lib.needle2_engine_prefill.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_int),
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
        ]
        lib.needle2_engine_prefill.restype = ct.c_int
        status = lib.needle2_engine_prefill(
            self._handle,
            ids_c.ctypes.data_as(ct.POINTER(ct.c_int)),
            num_tokens,
            1 if last_only else 0,
            logits.ctypes.data_as(ct.POINTER(ct.c_float)),
            None,
        )
        if status:
            raise RuntimeError(self._lib.needle2_engine_error().decode())
        self.position += num_tokens
        return logits

    def _torch_prefill(self, ids, last_only):
        import torch
        from .model import NeedleModel
        if self.position:
            raise ValueError("torch prefill requires reset() before the initial prompt")
        if ids.size > self.metadata["max_seq_len"]:
            raise ValueError("prompt exceeds max_seq_len")
        if self._prefill_model is None:
            self._prefill_model = NeedleModel.from_archive(self._archive, quant_activations=bool(self._activation_bits)).eval()
        model = self._prefill_model
        tokens = torch.from_numpy(ids.astype(np.int64, copy=True))[None]
        sink = torch.arange(ids.size)[None] < self.prefix_len
        with torch.inference_mode():
            # The reference exposes hidden+cache internally; restricting the
            # tied-embedding projection to the final position saves most work.
            hidden, cache, _ = model._run(tokens, None, None, sink, False)
            if last_only:
                hidden = hidden[:, -1:]
            output = model._linear("embedding", model._aq(hidden).float(), model._w("embedding").float())
        k = np.ascontiguousarray(torch.stack(cache.keys).numpy()[:, 0].transpose(0, 2, 1, 3))
        v = np.ascontiguousarray(torch.stack(cache.values).numpy()[:, 0].transpose(0, 2, 1, 3))
        positions = np.ascontiguousarray(cache.key_positions.numpy(), dtype=np.int32)
        token_array = np.ascontiguousarray(ids, dtype=np.int32)
        lib = self._lib
        lib.needle2_engine_import.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        lib.needle2_engine_import.restype = ct.c_int
        status = lib.needle2_engine_import(self._handle, len(ids), self.prefix_len,
                                           token_array.ctypes.data, positions.ctypes.data, len(positions), k.ctypes.data, v.ctypes.data)
        if status:
            raise RuntimeError(lib.needle2_engine_error().decode())
        self.position = len(ids)
        result = output.numpy()[0].copy()
        return result[0] if last_only else result

    def release_prefill_model(self):
        """Release the optional dense prefill weights; packed decoding is unaffected."""
        self._prefill_model = None

    def generate(self, token_ids, max_new_tokens: int = 32, eos_token_id: int | None = None, prefix_len: int = 0, prefill_backend: str = "native"):
        """Reset state and greedily generate, returning prompt plus generated IDs."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        output = list(token_ids)
        if not output:
            raise ValueError("generation requires a prompt")
        if len(output) > self.metadata["max_seq_len"]:
            raise ValueError("prompt exceeds max_seq_len")
        max_new_tokens = min(max_new_tokens, self.metadata["max_seq_len"] - len(output))
        self.reset(prefix_len)
        if not max_new_tokens:
            return output
        logits = self.prefill(output, last_only=True, backend=prefill_backend)
        for i in range(max_new_tokens):
            token = int(logits.argmax())
            output.append(token)
            if token == eos_token_id or i + 1 == max_new_tokens:
                break
            logits = self.step(token)
        return output

    def decode(self, first_token: int, max_new_tokens: int = 128, grammar_dfa=None) -> list[int]:
        """Decode autoregressively in C++ until eos, max_new_tokens, or DFA terminal state."""
        if max_new_tokens <= 0:
            return []
        output_buffer = (ct.c_int * (max_new_tokens + 1))()
        count = ct.c_int(0)
        dfa_ptr = ct.byref(grammar_dfa._desc) if grammar_dfa is not None else None
        lib = self._lib
        lib.needle2_engine_decode_loop.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.POINTER(_DFAStateDesc),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
        ]
        lib.needle2_engine_decode_loop.restype = ct.c_int
        status = lib.needle2_engine_decode_loop(
            self._handle,
            int(first_token),
            int(max_new_tokens),
            dfa_ptr,
            output_buffer,
            ct.byref(count),
        )
        if status:
            raise RuntimeError(self._lib.needle2_engine_error().decode())
        generated_count = count.value
        self.position += (generated_count - 1)
        return [output_buffer[i] for i in range(generated_count)]
