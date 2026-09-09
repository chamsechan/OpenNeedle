"""Thin ownership/array adapters for the native tokenizer and schema compiler.

RefTokenizer and Python grammar remain independent test/reference backends.
"""
import ctypes as ct
from functools import lru_cache
import numpy as np
from .native import _library, _DFAStateDesc
from .tokenizer import RefTokenizer, parse_tokenizer_blob


@lru_cache(maxsize=1)
def _bindings():
    lib = _library()
    signatures = {
        'needle2_frontend_error': ([], ct.c_char_p),
        'needle2_tokenizer_create': ([ct.c_void_p, ct.c_size_t], ct.c_void_p),
        'needle2_tokenizer_free': ([ct.c_void_p], None),
        'needle2_tokenizer_encode': ([ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p, ct.c_int], ct.c_int),
        'needle2_tokenizer_decode': ([ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int], ct.c_int),
        'needle2_grammar_compile': ([ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_void_p),
        'needle2_grammar_free': ([ct.c_void_p], None),
        'needle2_grammar_dfa': ([ct.c_void_p], ct.POINTER(_DFAStateDesc)),
        'needle2_engine_decode_compiled': ([ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
                                          ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int)], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:
            raise RuntimeError('Prebuilt library lacks the native frontend; rebuild OpenNeedle.') from exc
        fn.argtypes, fn.restype = args, result
    return lib


def _error(lib):
    return lib.needle2_frontend_error().decode('utf-8', 'replace')


class NativeTokenizer(RefTokenizer):
    """Embedded .cact tokenizer: C++ BPE and text decoding; Python metadata view."""
    def __init__(self, blob):
        super().__init__(parse_tokenizer_blob(blob))
        self._lib = _bindings()
        self._handle = self._lib.needle2_tokenizer_create(blob, len(blob))
        if not self._handle:
            raise ValueError(_error(self._lib))

    @classmethod
    def from_cact(cls, path):
        from .archive import Archive
        return cls(Archive.load(path).tensors['tokenizer'].blob)

    def __del__(self):
        if getattr(self, '_handle', None):
            self._lib.needle2_tokenizer_free(self._handle)
            self._handle = None

    def encode(self, text, *, add_dummy_prefix=None):
        data = text.encode('utf-8')
        dummy = -1 if add_dummy_prefix is None else int(bool(add_dummy_prefix))
        out = np.empty(len(data)+4, dtype=np.int32)
        count = self._lib.needle2_tokenizer_encode(self._handle, data, len(data), dummy, out.ctypes.data, len(out))
        if count < 0:
            raise ValueError(_error(self._lib))
        if count > len(out):
            out = np.empty(count, dtype=np.int32)
            count = self._lib.needle2_tokenizer_encode(self._handle, data, len(data), dummy, out.ctypes.data, len(out))
            if count < 0:
                raise ValueError(_error(self._lib))
        return out[:count].tolist()

    def decode(self, ids):
        # Validate before narrowing to int32 so large values cannot wrap.
        values = list(ids)
        if any(not isinstance(t, (int, np.integer)) or not 0 <= t < len(self.pieces) for t in values):
            raise ValueError('token outside tokenizer vocabulary')
        tokens = np.ascontiguousarray(values, dtype=np.int32)
        out = ct.create_string_buffer(4*len(tokens)+256)
        count = self._lib.needle2_tokenizer_decode(self._handle, tokens.ctypes.data, len(tokens), out, len(out))
        if count < 0:
            raise ValueError(_error(self._lib))
        if count > len(out):
            out = ct.create_string_buffer(count)
            count = self._lib.needle2_tokenizer_decode(self._handle, tokens.ctypes.data, len(tokens), out, len(out))
            if count < 0:
                raise ValueError(_error(self._lib))
        return out.raw[:count].decode('utf-8')


class CompiledGrammar:
    """C++-owned grammar compiled from tool JSON; immutable and reusable."""
    def __init__(self, tokenizer, tools_json):
        self._lib = _bindings()
        data = tools_json.encode('utf-8')
        self._handle = self._lib.needle2_grammar_compile(tokenizer._handle, data, len(data))
        if not self._handle:
            raise ValueError(_error(self._lib))

    def __del__(self):
        if getattr(self, '_handle', None):
            self._lib.needle2_grammar_free(self._handle)
            self._handle = None


def decode_native(engine, logits, max_new_tokens, grammar=None):
    """Select the first token and decode in C++, with no Python grammar loop."""
    if isinstance(max_new_tokens, (bool, np.bool_)) or not isinstance(max_new_tokens, (int, np.integer)) or not 0 <= max_new_tokens <= np.iinfo(np.int32).max:
        raise ValueError('max_new_tokens must be a nonnegative int32')
    values = np.ascontiguousarray(logits, dtype=np.float32)
    if values.shape != (engine.metadata['vocab_size'],):
        raise ValueError('expected one full-vocabulary logits vector')
    if not max_new_tokens:
        return []
    if not engine.metadata['kv_window']:
        max_new_tokens = min(max_new_tokens, engine.metadata['max_seq_len']-engine.position+1)
    out = np.empty(max_new_tokens, dtype=np.int32)
    count = ct.c_int()
    lib = _bindings()
    if lib.needle2_engine_decode_compiled(engine._handle, values.ctypes.data, len(values), max_new_tokens,
            grammar._handle if grammar is not None else None, out.ctypes.data, ct.byref(count)):
        raise RuntimeError(_error(lib))
    engine.position += count.value - 1
    return out[:count.value].tolist()
