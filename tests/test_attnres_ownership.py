"""Independent physical-owner and arithmetic counterexamples for AttnRes."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.family_attnres import attn_res_point, build, recipes
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError


def _case(aiv, *, split=False, hidden=64):
    rates = {"memory": 1e30, "vector": 1e9, "mte3": 1e30}
    units = {"vector": aiv, "mte3": aiv}
    if split:
        rates.update(mte2_aic=1e30, mte2_aiv=1e30)
        units.update(mte2_aic=max(1, aiv // 2), mte2_aiv=aiv)
    else:
        rates["mte2"] = 1e30
        units["mte2"] = aiv
    return parse_case({
        "machine": {"num_devices": 1, "device": {
            "memory_bytes": 2**30, "compute_units": max(1, aiv // 2),
            "compute_peaks": {"bf16": 1e14}, "rates": rates,
            "resource_units": units, "buffers": {"ub": 196608},
            "buffer_units": {"ub": aiv}}},
        "model": {"name": "owner-control", "layers": 1, "hidden_size": hidden,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": hidden,
                  "intermediate_size": 2 * hidden, "vocab_size": 16,
                  "activation_dtype": "bf16"},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    })


def _point(case, *, tokens=1, valid=1, norm=True):
    hidden = case.model.hidden_size
    return attn_res_point(
        {"prefix": "point", "stage": 0, "layer": 0, "model": case.model,
         "tokens": tokens, "mode": "decode"},
        prefix_id="prefix", addend_id=None, bank_ids=tuple(f"bank{i}" for i in range(valid)),
        valid_blocks=valid, write_slot=None, output_norm=norm, mix=True,
        materialize=False, tokens_local=tokens, bank_capacity=valid,
        block_token_stride=valid * hidden).ops[0]


def _prepared(case, op):
    return recipes()["attn_res_fwd"].prepare(op, Geometry.from_devices(case.machine.dies))[0]


def test_one_token_can_only_use_one_physical_aiv_and_split_read_pipe():
    durations = []
    for count in (2, 64):
        case = _case(count, split=True)
        op = _point(case)
        result = verify_operator(case, op, build, input_ready_ms=7)["result"]
        prepared = _prepared(case, op)
        # One row on one owner; fixed chip throughput is divided among AIVs.
        expected = 1000 * prepared.vector_ops / (1e9 / count)
        assert result["duration_ms"] == pytest.approx(expected, rel=1e-9)
        assert result["serviced"]["mte2_aiv@r0:s0"] == 640
        assert not any("mte2_aic" in name for name in result["serviced"])
        assert result["service_before_input"] == {}
        assert result["live_tasks"] == 0 and not any(result["remaining_buffers"].values())
        durations.append(result["duration_ms"])
    assert durations[1] / durations[0] == pytest.approx(32)


def test_source_vector_arithmetic_cannot_omit_score_or_current_prefix_row():
    case = _case(8, hidden=512)
    op = _prepared(case, _point(case, valid=3, norm=False))
    # Per source row: square, two multiplies and two reductions for score,
    # then weighted multiply/add. Excludes every cast and special function.
    ordinary_arithmetic_lower_bound = 4 * (7 * 512 - 2) + 512
    assert op.vector_ops >= ordinary_arithmetic_lower_bound
    assert op.flops == 0 and op.matrix_flops == ()


def test_contiguous_tail_ownership_and_explicit_source_dtype():
    case = _case(64, hidden=64)
    op = _prepared(case, _point(case, tokens=65, valid=2))
    view = dict(op.metadata)["attn_res_view"]
    # Host ceil division leaves owners33..63 idle, not evenly fractional rows.
    assert view["tokens_per_core"] == 2 and view["used_cores"] == 33
    assert view["core_ranges"][-1] == (64, 1)
    assert op.mte2_bytes == 2 * 65 * 64 * 4 + 4 * 64 * 33
    wrong_dtype = replace(case, model=replace(case.model, activation_dtype="fp16"))
    with pytest.raises(ModelingError, match="attn_res_source_requires_bf16"):
        _point(wrong_dtype)
