"""T13: source-proven BF16 materialization for INT8xINT8 projections.

[Role]
  Separate matrix input bytes from the materialized result/consumer bytes.
  The five Qwen3.6 QuantBatchMatmulV3 families and GDN Conv1d materialize
  BF16 even though the GEMM consumes INT8.  A role without the source
  declaration keeps the prior input-width fallback.

[Invariants]
  * qkvz/gate_up/down/A0 qkv/A0 o outputs and GDN conv output are BF16 bytes;
    qk_norm/rotary/activation/preserved consumers use those BF16 bytes.
  * Declaring output_format=int8 for the same role is an explicit control and
    is not silently rewritten to BF16.
  * No declaration preserves the prior act_bytes fallback.
  * B17/T12 scale/role work is not re-derived here.

[Side-Effect Policy] Pure compiled geometry; no I/O or measurements.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"


def base_case():
    return parse_case(json.loads(CASE.read_text()))


def declared_quant(q, output_format="bf16"):
    overrides = dict(q.category_overrides)
    for category in ("gdn", "dense_mlp", "attn.qkv", "attn.o"):
        values = dict(overrides.get(category, ()))
        values["output_format"] = output_format
        overrides[category] = tuple(values.items())
    return replace(q, category_overrides=overrides,
                   runtime_weight_scale_format="bf16", runtime_weight_scale_bytes=2.0,
                   runtime_activation_scale_format="fp32",
                   runtime_int8_weight_metadata="scale_only")


def pipeline(q):
    case = base_case()
    model = replace(case.model, quant=q)
    return case, assemble_model(
        model, supply=PipeSupply.from_hardware(case.machine),
        policy=ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                                    split_target=1000, max_splits=4),
        features=FeatureConfig.from_runtime(case.runtime, model))


def find(pipe, kind):
    try:
        return pipe.instantiated.find(kind)
    except Exception:
        pass
    for layer in pipe.instantiated.layers:
        try:
            return layer.find(kind)
        except Exception:
            continue
    raise AssertionError(kind)


def materials(pipe, kind, rows):
    inst = find(pipe, kind)
    geom = WorkloadGeometry((RequestGeometry(rows, rows, phase="prefill",
                                             query_ranges=((0, rows),)),))
    evaluation = pipe._subgraph(inst).evaluate(geom)
    return {op.name: {key: float(value) for key, value in op.materialization}
            for op in evaluation.operators}


@pytest.mark.parametrize("rows", (2, 4096, 4097, 4098))
def test_source_roles_materialize_bf16_and_consumers_use_it(rows):
    case = base_case()
    q = declared_quant(case.model.quant)
    _, pipe = pipeline(q)
    a3 = materials(pipe, "A3", rows)
    f0 = materials(pipe, "F0", rows)
    a0 = materials(pipe, "A0", rows)
    assert a3["qkvz_projection"]["qkvz_projection_output"] == rows * 8192 * 2
    assert a3["ba_projection"]["ba_projection_output"] == rows * 48 * 2
    assert a3["o_projection"]["o_projection_output"] == rows * 5120 * 2
    assert a3["conv"]["conv_output"] == rows * 5120 * 2
    assert a3["prepare"]["prepare_output"] == rows * 5120 * 2
    assert a3["gated_norm"]["gated_norm_output"] == rows * 3072 * 2
    assert f0["gate_up"]["ffn_activation"] == rows * 17408 * 2
    assert f0["activation"]["activation_intermediate"] == rows * 8704 * 2
    assert f0["down"]["down_output"] == rows * 5120 * 2
    assert a0["qkv_projection"]["qkv_activation"] == rows * 7168 * 2
    assert a0["o_projection"]["o_projection_output"] == rows * 5120 * 2
    # Preserved consumers read the BF16 result, not the INT8 GEMM operand.
    assert a0["qk_norm"]["qk_norm_output"] == rows * (12 + 2) * 256 * 2
    assert a0["rotary"]["rotary_output"] == rows * (12 + 2) * 256 * 2


def test_explicit_int8_result_and_absent_declaration_are_not_rewritten():
    case = base_case()
    q_int8 = declared_quant(case.model.quant, output_format="int8")
    _, pipe = pipeline(q_int8)
    a3 = materials(pipe, "A3", 4096)
    assert a3["qkvz_projection"]["qkvz_projection_output"] == 4096 * 8192
    assert a3["conv"]["conv_output"] == 4096 * 5120
    # No source output declaration: preserve the prior input-width fallback.
    _, plain = pipeline(case.model.quant)
    p3 = materials(plain, "A3", 4096)
    assert p3["qkvz_projection"]["qkvz_projection_output"] == 4096 * 8192
    assert p3["conv"]["conv_output"] == 4096 * 5120
