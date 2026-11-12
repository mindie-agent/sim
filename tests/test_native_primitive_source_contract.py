"""Source-tiling and declared-argument contracts for standalone primitives.

These tests exercise the shared primitive implementation, not a calibration
corpus or a measured target.  They pin the source quantities that changed
normal pricing: the CVDiff query-tile start selected by head dimension, and
the causal-convolution call's declared bias argument.
"""
from __future__ import annotations

import pytest

from mindie_sim.prediction import attention_prefill
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_gdn import make_call
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.operators import OpSpec

SOURCE_REFERENCES = "\n".join(
    " ".join(reference) for reference in attention_prefill.SOURCE_REFERENCES
)


def _prefill_op(*, heads: int, kv_heads: int, dim: int, query: int,
                context: int = 0) -> OpSpec:
    metadata = {
        "requests": ((query, context),),
        "query_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "query_dtype": "bf16",
        "kv_dtype": "bf16",
        "causal": True,
        "window": 0,
        "cp": 1,
    }
    length = query + context
    shape = (query, length, heads, dim)
    return OpSpec("gqa_prefill", "attention", "prefill", 0, 0, (),
                  shape, shape, shape, shape, recipe="flash_attention_prefill",
                  dtype="bf16", metadata=(("gqa_prefill", metadata),))


def test_source_query_tile_follows_cv_diff_head_size_thresholds():
    assert attention_prefill.source_query_tile(256) == 64
    assert attention_prefill.source_query_tile(200) == 64
    assert attention_prefill.source_query_tile(128) == 128
    assert attention_prefill.source_query_tile(8) == 128
    assert "6415-6443" in SOURCE_REFERENCES
    assert "PP_BLOCK_BUFFER_SIZE" in SOURCE_REFERENCES


def test_default_prefill_uses_head_dim_tiling_and_explicit_option_wins():
    query = 2048
    wide = _prefill_op(heads=4, kv_heads=1, dim=256, query=query)
    narrow = _prefill_op(heads=8, kv_heads=1, dim=128, query=query)

    default = attention_prefill.StagedGQAPrefill()
    wide_report = default.report(wide)
    narrow_report = default.report(narrow)
    assert wide_report["query_tile"] == 64
    assert narrow_report["query_tile"] == 128
    # Same query rows and the same heads*dim product, but the source selects
    # a different resident query panel for 256-wide and 128-wide heads.
    assert wide_report["tasks"] == 4 * (query // 64)
    assert narrow_report["tasks"] == 8 * (query // 128)

    explicit = attention_prefill.StagedGQAPrefill(query_tile=32)
    assert explicit.report(narrow)["query_tile"] == 32
    assert explicit.report(narrow)["tasks"] == 8 * (query // 32)


def _conv_case():
    from mindie_sim.prediction.structure import parse_case

    raw = {
        "machine": {
            "name": "synthetic-conv", "num_devices": 1,
            "device": {
                "memory_bytes": 1 << 30,
                "compute_units": 4,
                "compute_peaks": {"bf16": 8e12, "fp32": 4e12},
                "rates": {
                    "memory": 1e9, "vector": 1e9, "mte1": 1e9,
                    "mte2": 1e9, "mte3": 1e9,
                    **{role: 1e9 for role in (
                        "reg_data_writes", "reg_ub_accesses", "reg_ub_stores",
                        "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes",
                        "reg_shuffle_lanes")},
                },
                "resource_units": {
                    role: 8 for role in (
                        "vector", "reg_data_writes", "reg_ub_accesses",
                        "reg_ub_stores", "reg_alu_lanes", "reg_special_lanes",
                        "reg_reduce_lanes", "reg_shuffle_lanes")},
                "service_modes": {
                    "ub_copy": {"resource": "vector", "rate": 1e9},
                    "reg_issue_slots": {"resource": "vector", "rate": 1e9},
                },
                "buffers": {"l1": 1 << 22, "l0a": 1 << 21, "l0b": 1 << 21,
                            "l0c": 1 << 22, "ub": 1 << 22},
                "buffer_units": {"l1": 4, "l0a": 4, "l0b": 4, "l0c": 4, "ub": 4},
            },
        },
        "model": {"name": "geometry", "layers": 1, "hidden_size": 16,
                  "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4,
                                   "output_tokens": 1}]},
        "runtime": {},
    }
    return parse_case(raw)


def _conv_recipe(call):
    from mindie_sim.prediction.operators import native_causal_conv_prefill

    return RecipeBook({op.recipe: PlannedRecipe(
        native_causal_conv_prefill.build) for op in call.ops})


def test_conv_call_honors_declared_bias_absence_without_strict_bias_requirement():
    params = {
        "batch": 1, "tokens": 8192, "conv_width": 2048, "kernel": 4,
        "input_dtype": "bf16", "activation": "silu", "initial_state": False,
    }
    with_bias = make_call("conv_prefill", {**params, "bias": True})
    without_bias = make_call("conv_prefill", {**params, "bias": False})
    assert dict(with_bias.ops[0].metadata)["bias"] is True
    assert dict(without_bias.ops[0].metadata)["bias"] is False

    case = _conv_case()
    biased = call_costs(case, with_bias.ops, recipes=_conv_recipe(with_bias),
                        invocations=1)["duration_ms"]
    unbiased = call_costs(case, without_bias.ops,
                          recipes=_conv_recipe(without_bias),
                          invocations=1)["duration_ms"]
    # Bias selects the generic RunSeq branch; no-bias retains the source
    # rolling/register branch.  The two declarations are not interchangeable.
    assert unbiased < biased
    without = graph_costs(case, without_bias.ops,
                          recipes=_conv_recipe(without_bias),
                          isolated=True)["timeline"][0]
    assert without["isolated_ms"] == pytest.approx(unbiased)
