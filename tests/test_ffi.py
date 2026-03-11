"""
Tests for native backend loading helpers.
"""

import importlib
import types
import warnings

from barqtrain import _ffi


def test_detect_cuda_home_prefers_environment(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda"
    cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))

    assert _ffi._detect_cuda_home() == str(cuda_home)


def test_load_cuda_backend_retries_import_after_late_install(monkeypatch):
    module = types.SimpleNamespace(__name__="barqtrain_cuda")
    original_import_module = _ffi.importlib.import_module
    state = {"count": 0}

    def fake_import_module(name):
        if name == "barqtrain_cuda":
            state["count"] += 1
            if state["count"] == 1:
                raise ImportError("not installed yet")
            return module
        return original_import_module(name)

    monkeypatch.setattr(_ffi, "_CUDA_BACKEND", None)
    monkeypatch.setattr(_ffi, "_CUDA_BUILD_FAILED", False)
    monkeypatch.setattr(_ffi, "_env_enabled", lambda name, default="1": False)
    monkeypatch.setattr(_ffi.importlib, "import_module", fake_import_module)

    assert _ffi.load_cuda_backend() is None
    assert _ffi.load_cuda_backend() is module


def test_load_rust_backend_retries_import_after_late_install(monkeypatch):
    module = types.SimpleNamespace(__name__="barqtrain_rs")
    original_import_module = _ffi.importlib.import_module
    state = {"count": 0}

    def fake_import_module(name):
        if name == "barqtrain_rs":
            state["count"] += 1
            if state["count"] == 1:
                raise ImportError("not installed yet")
            return module
        return original_import_module(name)

    monkeypatch.setattr(_ffi, "_RUST_BACKEND", None)
    monkeypatch.setattr(_ffi, "_RUST_BUILD_FAILED", False)
    monkeypatch.setattr(_ffi, "_env_enabled", lambda name, default="1": False)
    monkeypatch.setattr(_ffi.importlib, "import_module", fake_import_module)

    assert _ffi.load_rust_backend() is None
    assert _ffi.load_rust_backend() is module


def test_data_import_does_not_warn_when_rust_backend_is_missing(monkeypatch):
    monkeypatch.setattr(_ffi, "load_rust_backend", lambda: None)
    import barqtrain.data as data

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(data)

    assert not any("Rust backend unavailable" in str(w.message) for w in caught)


def test_ops_import_does_not_warn_when_cuda_backend_is_missing(monkeypatch):
    monkeypatch.setattr(_ffi, "load_cuda_backend", lambda: None)
    import barqtrain.ops as ops

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(ops)

    assert not any("CUDA backend unavailable" in str(w.message) for w in caught)
