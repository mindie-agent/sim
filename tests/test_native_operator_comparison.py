"""Compare retained native ABI rows with the actual model frontend."""
from dataclasses import replace
import json
from pathlib import Path

from mindie_sim import Case
from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import gdn_output, native_causal_conv_prefill
from tools.compare_operators import GDN_KINDS, observed_contract, predicted_contract


def test_native_gdn_comparison_uses_real_frontend_shapes_and_gate_precision(profiler_api):
    case = Case.from_mapping(json.loads(Path("examples/cases/qwen38-a5.json").read_text()))
    actual = {row.kind: row for row in kernel_rows(
        "tests/fixtures/observations/gdn-native-kernels.csv")}
    ops = {op.family: op for op in target_operators(
        case, phase="prefill", tokens=16384) if op.family in GDN_KINDS}
    assert set(ops) == set(GDN_KINDS)
    for family, kind in GDN_KINDS.items():
        assert predicted_contract(ops[family].to_dict()) == observed_contract(actual[kind])
    # Matching only q/k/v dimensions would have admitted the old BF16 gate
    # branch, although the actual native ABI receives FP32 gate and beta.
    wrong_gate = replace(actual["ChunkGatedDeltaRuleFwdPrepare"],
                         input_dtypes=("DT_BF16",) * 5 + ("INT64",) * 2)
    assert observed_contract(wrong_gate) != predicted_contract(ops["gdn_prepare"].to_dict())


def test_equal_gdn_tensor_abi_does_not_imply_equal_sequence_execution():
    params = dict(heads=3, key_heads=1, dk=128, dv=128, chunk=64,
                  dtype="bf16", gate_dtype="fp32", use_exp2=True, state_v_first=True)
    left = gdn_output.make_op(dict(params, lengths=(1, 127)))
    right = gdn_output.make_op(dict(params, lengths=(17, 111)))
    assert predicted_contract(left.to_dict()) == predicted_contract(right.to_dict())
    geometry = Geometry(1, {"vector": 2})
    left_work = gdn_output.build(left, geometry).work
    right_work = gdn_output.build(right, geometry).work
    # Same 128-token ABI and three chunks, different executed tail tiles.
    assert abs(left_work.matrix["bf16"] - right_work.matrix["bf16"]) == 786432


def test_native_conv_matches_active_descriptors_without_claiming_cache_capacity(profiler_api):
    case = Case.from_mapping(json.loads(Path("examples/cases/qwen38-a5.json").read_text()))
    row = next(kernel_rows("tests/fixtures/observations/native-causal-conv-kernel.csv"))
    op = next(op for op in target_operators(case, phase="prefill", tokens=16384)
              if op.family == "causal_conv1d_fn")
    assert predicted_contract(op.to_dict()) == observed_contract(row)
    meta = dict(op.metadata)
    assert row.input_shapes[3] == (648, 6, 10240)
    assert meta["state_rows"] != 648
    assert meta["state_history_slots"] == 6


def test_native_conv_cache_capacity_does_not_change_active_work():
    # FN touches width-1 history slots; speculative storage beyond that is
    # capacity, not additional per-token copy work.
    base = dict(lengths=(7,), dim=16, width=4, cache_indices=(1,),
                has_initial_state=(True,), state_rows=2)
    narrow = native_causal_conv_prefill.source_ledger(dict(base, state_history_slots=3))
    extended = native_causal_conv_prefill.source_ledger(dict(base, state_history_slots=6))
    assert extended["work"] == narrow["work"]
