"""Genuine AIV HCCL all-gather packet linked through the shared native call.

Evidence: COMRUN5A signed packet (R31 image, HCCL_OP_EXPANSION_MODE=AIV from
launch config sha 19cd8b67:20, world 2, BF16 5120/rank -> 10240).  The raw
identity is rank-local: gtid5/stream26/task17/connection2/device0|1/range.
The AivKernel and hcom_allGather rows are aliases of one call, not two
charges.  Only trace comparison may read their durations; this declaration
never prices or bounds a prediction.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.native_binding import (
    bind_native_call, validate_invocation,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_op_structure

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads(
    (ROOT / "examples/source-families/hccl-comrun5a-allgather.json").read_text())


def _op(call=None):
    call = EVIDENCE["call"] if call is None else call
    return OpSpec(
        "s0.l0.tp_all_gather", "allgather", "decode", 0, 0, (),
        (5120,), (5120,), (5120,), (5120,), link_bytes=10240,
        dtype="bf16", layout="contiguous",
        metadata=(("group_axis", "tp"), ("collective", "allgather"),
                  ("hccl_observed_call", call)))


def _lowered(call=None):
    return lower((_op(call),), SimpleNamespace(kernels={}), "bf16")[0]


def test_comrun5a_binds_real_hccl_api_tensor_and_observed_native_name():
    binding = bind_native_call(_lowered())
    assert binding.api["namespace"] == "hccl"
    assert binding.api["entry"] == "torch.distributed.all_gather_into_tensor"
    assert binding.api["aclnn_symbol"] is None
    assert binding.execution_core == "aiv"
    assert binding.typed["dtype"] == "bf16"
    assert binding.typed["shapes"]["input"] == [5120]
    assert binding.typed["shapes"]["output"] == [10240]
    assert binding.typed["attrs"]["collective"] == "allGather"
    assert binding.typed["attrs"]["world_size"] == 2
    assert binding.typed["attrs"]["elements_per_rank"] == 5120
    assert binding.typed["attrs"]["expansion"] == "AIV"
    assert binding.kernels[0]["name"] == "aiv_all_gather_bfloat16_t"
    assert binding.kernels[0]["evidence"] == "measured"
    assert len(binding.observed_aliases) == 2
    assert {alias["rank_id"] for alias in binding.observed_aliases} == {
        "rank_0", "rank_1"}
    assert "hccl_detailed_variant_unestablished" in binding.unresolved
    assert "nhr_label_not_equated_to_mesh1d" in binding.unresolved
    # The detailed template is not fabricated from the NHR row label.
    assert "hccl_selection" not in dict(_lowered().metadata)
    assert "mesh" not in json.dumps(binding.typed).lower()


def test_observed_duration_never_enters_the_call_descriptor():
    binding = bind_native_call(_lowered())
    encoded = json.dumps(binding.to_dict(), sort_keys=True)
    for forbidden in ("56.348811", "56.462669", "0.013445", "0.365121",
                      "measured_span_ms", "duration_ms"):
        assert forbidden not in encoded
    # Only raw identity/range is retained for the profiler join.
    for alias in binding.observed_aliases:
        assert set(alias) >= {"global_task_id", "stream_id", "task_id",
                              "connection_id", "device_id", "start_ns", "end_ns"}
        assert "duration" not in json.dumps(alias)


def test_complete_typed_invocation_and_negative_controls():
    binding = bind_native_call(_lowered())
    validated = validate_invocation(binding, binding.prepare_args())
    assert validated["call_id"] == binding.call_id
    with pytest.raises(ModelingError, match="dtype_mismatch"):
        validate_invocation(binding, {**binding.prepare_args(), "dtype": "fp16"})
    with pytest.raises(ModelingError, match="shape_mismatch"):
        validate_invocation(binding, {
            **binding.prepare_args(),
            "shapes": {**binding.typed["shapes"], "input": [4096]}})
    with pytest.raises(ModelingError, match="attr_mismatch"):
        validate_invocation(binding, {
            **binding.prepare_args(),
            "attrs": {**binding.prepare_args()["attrs"], "ranks": 1}})


def test_two_replicas_share_one_call_and_one_owner():
    op = _lowered()
    structure = export_op_structure(
        {0: (op,), 1: (op,)}, model={}, execution={"phase": "decode"})
    records = structure["native_calls"]
    assert len(records) == 2
    assert len({record["call_id"] for record in records}) == 1
    assert {record["timing_owner"]["op_id"] for record in records} == {
        "s0.l0.tp_all_gather"}
    coverage = structure["native_call_coverage"]["totals"]
    assert coverage["bound"] == 2
    # The signed packet's installed binary is still unresolved.
    assert coverage["fully_paired"] == 0


def test_aiv_and_aicpu_are_distinct_invocations_not_name_swaps():
    aiv = EVIDENCE["call"]
    aicpu = json.loads(json.dumps(aiv))
    aicpu["typed"]["attrs"]["expansion"] = "AICPU_TS"
    aicpu["kernels"] = []
    aicpu["execution_core"] = None
    aicpu["unresolved"] = [*aicpu["unresolved"],
                           "hccl_native_c_allgather_symbol_unresolved"]
    aiv_binding = bind_native_call(_lowered(aiv))
    aicpu_binding = bind_native_call(_lowered(aicpu))
    assert aiv_binding.call_id != aicpu_binding.call_id
    assert aicpu_binding.typed["attrs"]["expansion"] == "AICPU_TS"
    assert aicpu_binding.kernels == ()
