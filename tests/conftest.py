"""Optional integration dependencies for capture-backed profiling tests."""
import importlib

import pytest

from mindie_sim.observations.kernels import PROFILER_API_REQUIREMENT


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(root / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(root / "diagnostics"))


@pytest.fixture
def profiler_api():
    for module in ("mindie_profiler.normalized", "mindie_profiler.tensor_contracts"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in {"mindie_profiler", module}:
                raise
            pytest.skip(PROFILER_API_REQUIREMENT)
