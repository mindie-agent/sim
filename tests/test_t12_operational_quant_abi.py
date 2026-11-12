"""T12: actual operational W8A8 role labels and scale ABI.

[Role]
  Carry source role names into the physical simulator and expose the actual
  QuantBatchMatmulV3 scale separation: per-N BF16 weight scale, per-M FLOAT
  dynamic activation scale.

[Invariants]
  * GDN a/b and GDN o use the source overrides gdn.beta_alpha / gdn.o and are
    BF16 x BF16; qkvz stays INT8 x INT8.
  * Dense F0, attention qkv/o and GDN qkvz stay INT8 x INT8; lm_head BF16.
  * W8 dynamic INT8 weight metadata is one BF16 scale per output channel;
    activation metadata is one FLOAT scale per row.
  * Same-family alternate declarations without these source fields keep the
    prior storage/ABI; no global all-W8A8 two-byte substitution.

[Side-Effect Policy] Pure catalog operator construction; no I/O.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"


def qwen_case():
    return parse_case(json.loads(CASE.read_text()))


def op(case, *, m, n, k, category, quant=None):
    return gemm_op(
        op_id="t12", deps=(), stage=0, layer=0, requests=(), m=m, n=n, k=k,
        quant=quant or case.model.quant, category=category, phase="decode",
        family="gemm", recipe="t12")


def meta(op):
    return dict(op.metadata)


def test_gdn_source_roles_reach_physical_operator():
    case = qwen_case()
    q = case.model.quant
    assert q.for_category("gdn.beta_alpha").weight_format == "bf16"
    assert q.for_category("gdn.o").weight_format == "bf16"
    assert q.for_category("gdn").weight_format == "int8"
    assert q.for_category("attn.qkv").weight_format == "int8"
    assert q.for_category("attn.o").weight_format == "int8"
    assert q.for_category("dense_mlp").weight_format == "int8"
    assert q.format_for("lm_head") == "bf16"

    gdn_ba = op(case, m=4096, n=48, k=5120, category="gdn.beta_alpha")
    gdn_o = op(case, m=4096, n=5120, k=3072, category="gdn.o")
    gdn_qkvz = op(case, m=4096, n=8192, k=5120, category="gdn")
    assert meta(gdn_ba)["weight_format"] == "bf16"
    assert meta(gdn_ba)["packed_weight_bytes"] == 48 * 5120 * 2
    assert meta(gdn_o)["weight_format"] == "bf16"
    assert meta(gdn_o)["packed_weight_bytes"] == 5120 * 3072 * 2
    assert meta(gdn_qkvz)["weight_format"] == "int8"
    assert meta(gdn_qkvz)["packed_weight_bytes"] == 8192 * 5120 + 8192 * 8  # storage scales/zero before runtime ABI
    assert not any(k in meta(gdn_ba) for k in ("weight_scale_format",))


def _pipeline(case):
    from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
    from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
    from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    return assemble_model(case.model, supply=PipeSupply.from_hardware(case.machine),
                          policy=ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                                                      split_target=1000, max_splits=4),
                          features=features)


def _find(pipe, kind):
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


def test_actual_subgraph_theta_uses_role_specific_quant():
    case = qwen_case()
    pipe = _pipeline(case)
    a3 = _find(pipe, "A3")
    a0 = _find(pipe, "A0")
    f0 = _find(pipe, "F0")
    o0 = _find(pipe, "O0")
    q = lambda instance, name: dict(instance.operator(name).theta.quant)
    assert q(a3, "ba_projection")["weight_format"] == "bf16"
    assert q(a3, "ba_projection")["weight_payload"] == 48 * 5120 * 2
    assert q(a3, "o_projection")["weight_format"] == "bf16"
    assert q(a3, "o_projection")["weight_payload"] == 3072 * 5120 * 2
    assert q(a3, "qkvz_projection")["weight_format"] == "int8"
    assert q(a0, "qkv_projection")["weight_format"] == "int8"
    assert q(a0, "qkv_projection")["act_format"] == "int8"
    assert q(a0, "o_projection")["weight_format"] == "int8"
    assert q(a0, "o_projection")["act_format"] == "int8"
    assert q(f0, "gate_up")["weight_format"] == "int8"
    assert q(f0, "gate_up")["act_format"] == "int8"
    assert q(f0, "down")["weight_format"] == "int8"
    assert q(f0, "down")["act_format"] == "int8"
    assert q(a3, "qkvz_projection")["act_format"] == "int8"
    assert q(o0, "lm_head")["weight_format"] == "bf16"


def test_wrong_generic_role_and_absent_override_are_not_substituted():
    case = qwen_case()
    q = case.model.quant
    # The wrong generic role keeps the INT8 family policy; it is not silently
    # equated to the source-specific beta/alpha or o role.
    wrong = op(case, m=4096, n=48, k=5120, category="gdn", quant=q)
    assert meta(wrong)["weight_format"] == "int8"
    # Without the source declaration there is no categorical BF16 substitution.
    plain = replace(q, category_overrides=())
    plain_gdn_o = op(case, m=4096, n=5120, k=3072, category="gdn.o", quant=plain)
    assert meta(plain_gdn_o)["weight_format"] == "int8"


def runtime_quant(q):
    return replace(
        q, runtime_weight_scale_format="bf16", runtime_weight_scale_bytes=2.0,
        runtime_activation_scale_format="fp32",
        runtime_int8_weight_metadata="scale_only")


def test_normal_gemm_scale_separation_and_bytes():
    case = qwen_case()
    q = runtime_quant(case.model.quant)
    gate = op(case, m=4096, n=17408, k=5120, category="dense_mlp", quant=q)
    m = meta(gate)
    assert m["weight_scale_format"] == "bf16"
    assert m["weight_scale_bytes"] == 2.0
    assert m["weight_scale_storage_format"] == "fp32"
    assert m["weight_scale_storage_bytes"] == 4.0
    assert m["activation_scale_format"] == "fp32"
    assert m["activation_scale_bytes"] == 4096 * 4
    assert m["packed_weight_bytes"] == 17408 * 5120 + 17408 * 2
    assert m["weight_bytes"] == 17408 * 5120 + 17408 * 2
    assert gate.mte2_bytes == 4096 * 5120 + m["weight_bytes"] + 4096 * 4
    assert gate.hbm_bytes == 4096 * 5120 + m["weight_bytes"] + 4096 * 17408 * 2 + 4096 * 4
    assert m["int8_weight_metadata"] == "scale_only"


def test_runtime_scale_abi_reaches_normal_compiled_theta():
    case = qwen_case()
    model = replace(case.model, quant=runtime_quant(case.model.quant))
    from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
    from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
    from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
    features = FeatureConfig.from_runtime(case.runtime, model)
    pipe = assemble_model(model, supply=PipeSupply.from_hardware(case.machine),
                          policy=ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                                                      split_target=1000, max_splits=4),
                          features=features)
    a3, a0, f0, o0 = (_find(pipe, k) for k in ("A3", "A0", "F0", "O0"))
    q = lambda instance, name: dict(instance.operator(name).theta.quant)
    assert q(a3, "ba_projection")["weight_format"] == "bf16"
    assert q(a3, "ba_projection")["weight_scale"] == 0
    assert q(a3, "o_projection")["weight_format"] == "bf16"
    assert q(a3, "o_projection")["weight_total"] == 3072 * 5120 * 2
    assert q(a3, "qkvz_projection")["weight_scale"] == 8192 * 2
    assert q(f0, "gate_up")["weight_scale"] == 17408 * 2
    assert q(f0, "gate_up")["weight_total"] == 17408 * 5120 + 17408 * 2
    assert q(f0, "down")["weight_scale"] == 5120 * 2
    assert q(a0, "qkv_projection")["weight_scale"] == 7168 * 2
    assert q(a0, "o_projection")["weight_scale"] == 5120 * 2
    assert q(o0, "lm_head")["weight_format"] == "bf16"
    assert q(o0, "lm_head")["weight_total"] == 124160 * 5120 * 2


def test_other_formats_keep_storage_scale_abi_without_declaration():
    from mindie_sim.catalog.quant import quant_preset
    case = qwen_case()
    w4 = quant_preset("w4a8_dynamic")
    m = meta(op(case, m=4096, n=5120, k=8704, category="dense_mlp", quant=w4))
    assert m["weight_scale_format"] == "fp32"
    assert m["weight_scale_bytes"] == 4.0
    fp8 = quant_preset("w8a8_mxfp8")
    m8 = meta(op(case, m=4096, n=5120, k=8704, category="dense_mlp", quant=fp8))
    assert m8["weight_scale_format"] == "fp8_e8m0"
    assert m8["weight_scale_bytes"] == 1.0
    # A W8 declaration without the source runtime ABI keeps storage FP32.
    raw = meta(op(case, m=4096, n=17408, k=5120, category="dense_mlp"))
    assert raw["weight_scale_format"] == "fp32"
    assert raw["weight_scale_bytes"] == 4.0
    ds = parse_case(json.loads((ROOT / "examples/cases/deepseek-v4-flash-0731.json").read_text()))
    assert ds.model.quant.scale_format == "fp8_e8m0"
    assert ds.model.quant.scale_bytes == 1.0
    assert ds.model.quant.runtime_weight_scale_format == ""
    ds_meta = meta(op(case, m=4096, n=5120, k=8704, category="attn.qkv",
                      quant=ds.model.quant))
    assert ds_meta["weight_scale_format"] == "fp8_e8m0"
    assert ds_meta["weight_scale_bytes"] == 1.0


def test_mixed_q_and_group_m_counts_use_actual_rows_not_padded_m4098():
    case = qwen_case()
    q = runtime_quant(case.model.quant)
    for m in (4096, 4097, 4098):
        g = op(case, m=m, n=17408, k=5120, category="dense_mlp", quant=q)
        assert g.local_shape[0] == m
        assert g.flops == 2 * m * 17408 * 5120
        assert meta(g)["activation_scale_bytes"] == m * 4
        assert g.padded_shape[0] == ((m + 15) // 16) * 16
        assert g.hbm_bytes == (
            m * 5120 + meta(g)["packed_weight_bytes"] + m * 17408 * 2 + m * 4)
    pipe = _pipeline(case)
    kinds = {}
    for layer in pipe.instantiated.layers:
        for instance in layer.subgraphs:
            kinds[instance.kind] = kinds.get(instance.kind, 0) + 1
    assert kinds.get("A3") == 48
    assert kinds.get("A0") == 16
    assert kinds.get("F0") == 64
    assert kinds.get("O0", 0) == 0  # O0 is a global output instance
    assert _find(pipe, "O0").instance_id.startswith("O0:")
