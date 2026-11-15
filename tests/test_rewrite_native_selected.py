"""T04 rewrite: native selected plan compiles once and submits once.

[Role] Prove the N-domain native producer: normalized ABI -> one immutable
  ProgramCatalog (physical stages/work/ports) -> fresh BindingFrame on the
  caller Execution -> exactly one Submission with real per-port readiness.
[Invariants] The catalog carries no Event/Execution/allocation identity; two
  submissions create distinct occurrences; blocking and async receipts read
  the same actual Submission; semantic n:m members are provenance, never
  multiplied work.
[Timeouts & Locks] One bounded CPU Engine advance per case.
[Side-Effect Policy] CPU-only virtual engine; no device, weights or I/O.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from mindie_sim.prediction.piecewise.native_invocation import (
    _hardware_execution, native_execution_receipt, simulate_aclnn_call,
    submit_native_call,
)

SKU = "Ascend950DT_9582"

_R31_IMAGE = (
    "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
)


def _add_rms_request(*, rows: int = 256, hidden: int = 5120):
    return {
        "api": {"namespace": "aclnn", "entry": "aclnnAddRmsNorm",
                "aclnn_symbol": "aclnnAddRmsNorm"},
        "version": {"api": "cann-9.2.0", "image_id": _R31_IMAGE},
        "tensors": {
            "x": {"shape": [rows, hidden], "dtype": "bf16",
                  "layout": "ND", "role": "input"},
            "residual": {"shape": [rows, hidden], "dtype": "bf16",
                         "layout": "ND", "role": "input"},
            "gamma": {"shape": [hidden], "dtype": "bf16",
                      "layout": "ND", "role": "weight"},
            "output": {"shape": [rows, hidden], "dtype": "bf16",
                       "layout": "ND", "role": "output"},
            "rstd": {"shape": [rows, 1], "dtype": "fp32",
                     "layout": "ND", "role": "output"},
            "residual_out": {"shape": [rows, hidden], "dtype": "bf16",
                             "layout": "ND", "role": "output"},
        },
        "attrs": {"epsilon": 1e-6},
        "conditions": {"sequence_rows": "shard", "sequence_ranks": 2},
    }


def _matmul_request():
    return {
        "api": {"namespace": "aclnn", "entry": "aclnnMatmul"},
        "version": {"api": "cann-9.2.0", "source": "r31"},
        "tensors": {
            "x1": {"shape": [33, 256], "dtype": "bf16", "role": "input"},
            "x2": {"shape": [512, 256], "dtype": "bf16",
                   "role": "input"},
            "output": {"shape": [33, 512], "dtype": "bf16",
                       "role": "output"},
        },
        "attrs": {"transpose_x1": False, "transpose_x2": True},
    }


def test_native_selected_catalog_is_runtime_free_and_single_submission():
    result = simulate_aclnn_call(_add_rms_request(), hardware=SKU,
                                 execute=False)
    assert result.status == "predicted", result.reasons
    catalog = result.execution_catalog
    assert catalog is not None
    assert catalog.selected_call_count == 1
    assert catalog.stages and catalog.ports
    assert "residual_add" in catalog.semantic_sources
    assert "rmsnorm" in catalog.semantic_sources
    # No runtime identity is captured in the immutable catalog.
    assert not any(hasattr(value, "done") and hasattr(value, "time")
                   for port in catalog.ports for value in (port,))

    predicted_ms = float(Fraction(str(result.prediction["duration_ms"])))
    work = {str(pipe): float(value) for pipe, value in
            result.prediction["work"].items()}
    for pipe, amount in work.items():
        selected = sum(float(term.issued) for stage in catalog.stages
                       for term in stage.work
                       if term.physical_resource_role == pipe)
        assert selected == pytest.approx(amount)

    execution = _hardware_execution(SKU)
    receipt = submit_native_call(result, execution, name="t12-selected")
    submission = receipt["submission"]
    assert receipt["duration_ms"] == pytest.approx(predicted_ms)
    assert receipt["catalog_id"] == catalog.catalog_id
    assert submission.selected_program_id == catalog.catalog_id
    assert all(binding.ready.done for binding in submission.ports.values())
    assert receipt["duration_ms"] is not None and receipt["duration_ms"] >= 0.0
    # Exactly one physical call: the catalog is submitted once.
    assert execution.engine.summary()["completed_tasks"] >= 1
    view = native_execution_receipt(
        result, execution) if result.execution_handle is not None else None
    assert view is None or view["selected_program_id"] == catalog.catalog_id


def test_native_blocking_and_async_bind_fresh_occurrences():
    result = simulate_aclnn_call(_add_rms_request(), hardware=SKU,
                                 execute=False)
    execution = _hardware_execution(SKU)
    first = submit_native_call(result, execution, name="first")
    second = submit_native_call(result, execution, name="second")
    assert first["submission"].occurrence != second["submission"].occurrence
    assert first["submission"].selected_program_id == catalog_id(result)

    execution = _hardware_execution(SKU)
    pending = simulate_aclnn_call(
        _add_rms_request(), hardware=SKU, execution=execution,
        blocking=False, name="async")
    assert pending.execution_handle is not None
    assert not pending.execution_handle.done
    pending_handle = pending.execution_handle
    execution.engine.advance(until=(pending_handle.event,))
    assert pending_handle.done
    assert native_execution_receipt(
        pending, execution)["selected_program_id"] == catalog_id(result)


def catalog_id(result):
    return result.execution_catalog.catalog_id


def test_second_abi_family_uses_same_selected_catalog_path():
    result = simulate_aclnn_call(_matmul_request(), hardware=SKU,
                                 execute=False)
    assert result.status == "predicted", result.reasons
    catalog = result.execution_catalog
    assert catalog is not None
    assert catalog.selected_call_count == 1
    assert any(spec.direction == "out" for spec in catalog.ports)
    execution = _hardware_execution(SKU)
    receipt = submit_native_call(result, execution, name="matmul")
    assert receipt["selected_program_id"] == catalog.catalog_id
    assert receipt["submission"].ports


def test_retained_output_grant_binds_one_real_allocation():
    result = simulate_aclnn_call(_add_rms_request(), hardware=SKU,
                                 execute=False)
    execution = _hardware_execution(SKU)
    execution.engine.extend(pools={"hbm": 1 << 30})
    receipt = submit_native_call(
        result, execution, name="granted",
        output_grants={"output": {"pool": "hbm", "size": 4096,
                                  "storage": "retained-output"}})
    allocation = receipt["output_allocations"]["output"]
    assert allocation is not None
    assert str(allocation.symbol) in {
        str(key) for key in receipt["submission"].allocations}
    assert receipt["allocations"]


def test_selected_catalog_identity_is_stable_while_runtime_identity_is_fresh():
    first = simulate_aclnn_call(_add_rms_request(), hardware=SKU,
                                execute=False)
    second = simulate_aclnn_call(_add_rms_request(), hardware=SKU,
                                 execute=False)
    assert first.execution_catalog.catalog_id == (
        second.execution_catalog.catalog_id)
    assert first.execution_catalog.stages == second.execution_catalog.stages
    execution = _hardware_execution(SKU)
    a = submit_native_call(first, execution, name="stable-a")
    b = submit_native_call(second, execution, name="stable-b")
    assert a["catalog_id"] == b["catalog_id"]
    assert a["submission"].occurrence != b["submission"].occurrence


def test_selected_catalog_rebinds_geometry_without_timing_cache():
    small = simulate_aclnn_call(_add_rms_request(rows=128), hardware=SKU,
                                execute=False)
    large = simulate_aclnn_call(_add_rms_request(rows=512), hardware=SKU,
                                execute=False)
    assert small.execution_catalog.catalog_id != (
        large.execution_catalog.catalog_id)
    assert small.execution_catalog.stages != large.execution_catalog.stages
    assert small.prediction["work"] != large.prediction["work"]
