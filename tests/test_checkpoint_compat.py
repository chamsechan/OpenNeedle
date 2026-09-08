"""Restricted NumPy checkpoint loading works across the NumPy 1/2 module rename."""
import builtins
import io
from types import SimpleNamespace

import pytest

from needle2.convert import _NumpyUnpickler


@pytest.mark.parametrize("pickle_module", ["numpy.core.multiarray", "numpy._core.multiarray"])
def test_numpy1_reconstruction_fallback(monkeypatch, pickle_module):
    original_import = builtins.__import__
    marker = object()
    requested = []

    def numpy1_import(name, *args, **kwargs):
        if name == "numpy._core.multiarray":
            requested.append(name)
            raise ImportError("NumPy 1.x has no numpy._core")
        if name == "numpy.core.multiarray":
            requested.append(name)
            return SimpleNamespace(_reconstruct=marker)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", numpy1_import)
    assert _NumpyUnpickler(io.BytesIO()).find_class(pickle_module, "_reconstruct") is marker
    assert requested == ["numpy._core.multiarray", "numpy.core.multiarray"]
