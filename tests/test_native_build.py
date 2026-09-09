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


def test_macos_uses_compiler_sdk_and_keeps_openmp(monkeypatch, tmp_path):
    """An installed CLT SDK must not override the active compiler's headers."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "omp.h").touch()
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libomp.dylib").touch()
    glob = Path.glob

    def installed_sdks(path, pattern):
        if str(path) == "/Library/Developer/CommandLineTools/SDKs":
            return iter([Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk/usr/include/c++/v1")])
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", installed_sdks)
    commands = []

    def compile_stub(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled fixture")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(native.subprocess, "run", compile_stub)
    result = native.build_native()
    assert result.suffix == ".dylib" and result.read_bytes() == b"compiled fixture"
    command, = commands
    assert not any("CommandLineTools/SDKs" in arg or arg.startswith("-isystem") for arg in command)
    assert "-Xpreprocessor" in command and "-fopenmp" in command
    assert f"-I{tmp_path / 'include'}" in command and "-lomp" in command
