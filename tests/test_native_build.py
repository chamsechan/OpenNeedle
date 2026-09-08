"""Build selection failures must not silently compile a different library."""
import pytest

from needle2 import native


@pytest.fixture(autouse=True)
def fresh_build_selection(monkeypatch, tmp_path):
    native.build_native.cache_clear()
    monkeypatch.delenv("NEEDLE2_NATIVE_LIBRARY", raising=False)
    monkeypatch.setenv("NEEDLE2_NATIVE_CACHE", str(tmp_path / "cache"))
    yield
    native.build_native.cache_clear()


def test_prebuilt_selection_does_not_invoke_compiler(monkeypatch, tmp_path):
    library = tmp_path / "library with spaces.so"
    library.write_bytes(b"selection-only fixture")
    monkeypatch.setenv("NEEDLE2_NATIVE_LIBRARY", str(library))
    monkeypatch.setenv("CXX", "/compiler-does-not-exist")
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: pytest.fail("unexpected compilation"))
    assert native.build_native() == library.resolve()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("value", ["", "missing.so", "."])
def test_invalid_prebuilt_does_not_fall_back(monkeypatch, value):
    monkeypatch.setenv("NEEDLE2_NATIVE_LIBRARY", value)
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: pytest.fail("unexpected compilation"))
    with pytest.raises(RuntimeError, match="OpenNeedle"):
        native.build_native()


def test_missing_compiler_cleans_temporary_output(monkeypatch, tmp_path):
    monkeypatch.setenv("CXX", str(tmp_path / "missing-compiler"))
    with pytest.raises(RuntimeError, match="Native compiler not found"):
        native.build_native()
    assert list((tmp_path / "cache").iterdir()) == []
