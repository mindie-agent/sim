"""Optional integration dependencies for capture-backed profiling tests."""
import importlib

import pytest

from mindie_sim.observations.kernels import PROFILER_API_REQUIREMENT


@pytest.fixture
def profiler_api():
    for module in ("mindie_profiler.normalized", "mindie_profiler.tensor_contracts"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in {"mindie_profiler", module}:
                raise
            pytest.skip(PROFILER_API_REQUIREMENT)
