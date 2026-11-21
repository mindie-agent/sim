"""A5 vector-family normalization and quantization chain.

[Role] Prove that native AddRmsNorm/AddRmsNormDynamicQuant/MxQuant selection
  preserves residual, rstd, payload and scale materialization, and that the
  group-32 MX quantization phase uses the declared source register ledger.
[Invariants] INT8 is one FP32 scale per complete row; MX keeps an E8M0 scale
  per 32-value group plus the even-group tail pad; an unclassified fused work
  composition is rejected rather than silently repriced. No captured duration
  or model-name table selects a kernel.
[Timeouts & Locks] Pure metadata arithmetic on small synthetic shapes.
[Side-Effect Policy] No device, weight load or repository mutation.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.operators.vector.family_ascend_norm import lower_op
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.prediction.operators.vector.dynamic_quant_schedule import quant_timeline
from mindie_sim.prediction.operators.vector.mx_dynamic_quant import group32_work, producer_contract
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]


def _selected(op, **kwargs):
    return lower_op(op, **kwargs)[-1]


def _band(width, rows):
    groups = (width + 31) // 32
    padded = (groups + 1) // 2 * 2
    return groups, padded, (rows, padded // 2, 2)


def _norm(family, rows, width, *, quant=None, vector=None, read=None, write=None,
          returns=(), extra=(), dtype="bf16", bias=False):
    elements = rows * width
    meta = [
        ("norm_has_weight", True), ("norm_has_bias", bias),
        ("norm_eps", 1e-6), ("norm_gamma_role", "ffn_norm"),
        ("norm_gamma_runtime_bytes", width * 2),
        ("norm_gamma_storage_bytes", width * 2),
        ("norm_gamma_runtime_format", "bf16"),
        ("norm_gamma_storage_format", "bf16"),
    ]
    if quant == "int8":
        meta += [
            ("output_format", "int8"),
            ("activation_group_size", width),
            ("activation_scale_format", "fp32"),
            ("activation_scale_shape", (rows,)),
            ("activation_scale_bytes", rows * 4),
        ]
        payload, scale = rows * width, rows * 4
    elif quant == "mx":
        groups, padded, shape = _band(width, rows)
        meta += [
            ("output_format", "fp8_e4m3"),
            ("activation_group_size", 32),
            ("activation_scale_format", "fp8_e8m0"),
            ("activation_scale_shape", shape),
            ("activation_scale_bytes", rows * padded),
        ]
        payload, scale = rows * width, rows * padded
    else:
        payload = scale = 0
    if read is None:
        read = (rows * width * 2 + width * 2
                if family != "add_rmsnorm_quant" else (2 * rows + 2) * width * 2)
        if bias and not any(key == "norm_has_bias" for key, _ in meta):
            read += width * 2
    if write is None:
        if family == "add_rmsnorm_quant":
            write = rows * width * 2 + payload + scale
        elif quant:
            write = payload + scale
        else:
            # The fused residual norm writes both the normalized output and
            # the residual state; rstd is added by the native selection.
            write = 2 * rows * width * 2
    if vector is None:
        vector = (6 if family == "add_rmsnorm_quant" else 5) * elements + (
            4 * elements + 2 * rows * ((width + 31) // 32 if quant == "mx" else 1)
            if quant else 0)
    if family == "add_rmsnorm_quant":
        returns = (*returns, "residual", "activation_scales")
    elif quant:
        returns = (*returns, "activation_scales")
    return vector_op(
        op_id="body.norm", deps=("residual",), stage=0, layer=0, requests=("r",),
        elements=elements, ops_per_element=0, dtype=dtype, family=family,
        phase="decode", recipe=("add_rmsnorm_quantize" if family == "add_rmsnorm_quant"
                                else "rmsnorm_quantize" if quant else "add_rmsnorm"),
        shape=(rows, width), read_bytes_override=read, write_bytes_override=write,
        vector_ops_override=vector, returns=returns,
        metadata=tuple(meta) + tuple(extra),
    )


def test_a5_mx_plain_norm_quant_uses_source_register_ledger():
    rows, width = 4, 5120
    groups, padded, shape = _band(width, rows)
    op = _norm("rmsnorm_quant", rows, width, quant="mx")
    selected = _selected(op)
    meta = dict(selected.metadata)
    ledger = group32_work(width)["lane_ops"]["total"]
    elements = rows * width
    assert meta["native_norm_kind"] == "RmsNormDynamicMxQuant"
    assert selected.vector_ops == 5 * elements + rows * ledger
    assert meta["native_norm_vector_ops"] == 5 * elements
    assert meta["native_norm_quant_vector_ops"] == rows * ledger
    assert meta["native_norm_quant_work_source"] == (
        "mx_dynamic_quant.group32_work:r64_alg0")
    assert meta["native_norm_quant_scale_shape"] == shape
    assert meta["native_norm_quant_scale_bytes"] == rows * padded
    assert meta["native_norm_quant_scale_layout"] == "NCL"
    assert selected.mte2_bytes == rows * width * 2 + width * 2
    assert selected.mte3_bytes == rows * width + rows * padded
    assert selected.hbm_bytes == selected.mte2_bytes + selected.mte3_bytes
    assert "activation_scales" in selected.returns


@pytest.mark.parametrize("family", ("rmsnorm_quant", "add_rmsnorm_quant"))
def test_mx_norm_quant_beta_is_rejected_not_relabelled(family):
    # The declared MX tensor contract is x, residual-or-none, gamma.  A beta
    # request is an unmodelled input; the selector must not drop it or sign a
    # different kernel.  Bias-free requests still use the source kind.
    declared = _norm(family, 2, 3584, quant="mx", bias=True)
    with pytest.raises(ModelingError, match="mx_quant_beta_not_in_declared_abi"):
        lower_op(declared)
    caller_beta = _norm(family, 2, 3584, quant="mx")
    with pytest.raises(ModelingError, match="mx_quant_beta_not_in_declared_abi"):
        lower_op(caller_beta, has_bias=True)
    selected = _selected(caller_beta)
    assert dict(selected.metadata)["native_norm_kind"] == (
        "RmsNormDynamicMxQuant" if family == "rmsnorm_quant"
        else "AddRmsNormDynamicMxQuant")


def test_unquantized_add_rms_norm_declares_and_bills_residual_once():
    rows, width = 2, 5120
    selected = _selected(_norm("add_rmsnorm", rows, width))
    meta = dict(selected.metadata)
    residual = rows * width * 2
    normalized = rows * width * 2
    rstd = rows * 4
    assert residual == 20480
    assert meta["native_norm_residual_output"] is True
    assert meta["native_norm_residual_bytes"] == residual
    assert meta["native_norm_normalized_output_bytes"] == normalized
    assert meta["native_norm_residual_billing"] == "included_in_mte3"
    assert meta["native_norm_residual_format"] == "bf16"
    assert meta["native_norm_rstd_format"] == "fp32"
    assert meta["native_norm_normalized_format"] == "bf16"
    assert selected.returns.count("residual") == 1
    assert "rstd" in selected.returns
    # mte3 already contained both stores; the new ledger must not add them a
    # second time.  The two stores plus rstd are one complete billing.
    assert selected.mte3_bytes == residual + normalized + rstd
    # A predeclared external consumer output survives and is not duplicated.
    existing = _selected(_norm("add_rmsnorm", rows, width,
                               returns=("residual", "x")))
    assert existing.returns.count("residual") == 1
    assert "x" in existing.returns

def test_a5_mx_add_norm_quant_keeps_residual_and_tail_scale():
    rows, width = 3, 96
    groups, padded, shape = _band(width, rows)
    assert (groups, padded, shape) == (3, 4, (3, 2, 2))
    op = _norm("add_rmsnorm_quant", rows, width, quant="mx")
    selected = _selected(op)
    meta = dict(selected.metadata)
    ledger = group32_work(width)["lane_ops"]["total"]
    elements = rows * width
    assert meta["native_norm_kind"] == "AddRmsNormDynamicMxQuant"
    assert selected.vector_ops == 6 * elements + rows * ledger
    assert selected.mte2_bytes == (2 * rows + 2) * width * 2
    assert selected.mte3_bytes == rows * width * 2 + rows * width + rows * padded
    assert meta["native_norm_residual_bytes"] == rows * width * 2
    assert meta["native_norm_quant_scale_shape"] == (3, 2, 2)
    assert {"residual", "activation_scales"} <= set(selected.returns)


def test_a5_int8_row_quant_keeps_beta_work_and_complete_row_scale():
    rows, width = 2, 6144
    op = _norm("add_rmsnorm_quant", rows, width, quant="int8", bias=True)
    selected = _selected(op, has_bias=True)
    meta = dict(selected.metadata)
    assert meta["native_norm_kind"] == "AddRmsNormDynamicQuant"
    assert selected.mte2_bytes == (2 * rows + 2) * width * 2
    assert selected.mte3_bytes == rows * width * 3 + rows * 4
    assert selected.vector_ops == 6 * rows * width + 4 * rows * width + 2 * rows
    assert meta["native_norm_quant_scale_shape"] == (rows,)
    assert meta["native_norm_quant_scale_bytes"] == rows * 4
    assert meta["native_norm_rstd_bytes"] == 0
    assert {"residual", "activation_scales"} <= set(selected.returns)


def test_a5_source_declared_three_input_add_rms_norm_preserves_rstd():
    op = _norm("add_rmsnorm", 2, 5120, extra=(("norm_native_api", "AddRmsNorm"),))
    selected = _selected(op)
    meta = dict(selected.metadata)
    assert meta["native_norm_kind"] == "AddRmsNorm"
    assert selected.mte3_bytes == 2 * 2 * 5120 * 2 + 2 * 4
    assert meta["native_norm_rstd_bytes"] == 2 * 4
    assert "rstd" in selected.returns
    assert "residual" in selected.returns
    assert meta["native_norm_residual_bytes"] == 2 * 5120 * 2
    assert meta["native_norm_residual_billing"] == "included_in_mte3"
    assert meta["native_implementation"]["conditions"]["beta"] is False
    assert meta["native_implementation"]["conditions"]["api"] == "AddRmsNorm"


def test_a5_gemma_caller_source_class_selects_three_input_api():
    gemma = _norm("add_rmsnorm", 1, 3584, extra=(
        ("rms_norm_source_class", "AscendGemmaRMSNorm"),))
    plain = _norm("add_rmsnorm", 1, 3584, extra=(
        ("rms_norm_source_class", "AscendRMSNorm"),))
    assert dict(_selected(gemma).metadata)["native_norm_kind"] == "AddRmsNorm"
    assert dict(_selected(plain).metadata)["native_norm_kind"] == "AddRmsNormBias"
    biased = _norm("add_rmsnorm", 1, 3584, bias=True, extra=(
        ("rms_norm_source_class", "AscendGemmaRMSNorm"),))
    assert dict(_selected(biased, has_bias=True).metadata)[
        "native_norm_kind"] == "AddRmsNormBias"
    with pytest.raises(ModelingError, match="add_rms_norm_has_no_beta_slot"):
        lower_op(_norm("add_rmsnorm", 1, 3584, extra=(
            ("norm_native_api", "AddRmsNorm"), ("norm_has_bias", True))),
            has_bias=True)


def test_a5_per_block_int8_scale_is_not_reinterpreted_as_per_token():
    rows, width = 2, 5120
    op = _norm("add_rmsnorm_quant", rows, width, quant="int8")
    bad = op.metadata + (
        ("activation_group_size", 128),
        ("activation_scale_shape", (rows, width // 128)),
        ("activation_scale_bytes", rows * width // 128 * 4),
    )
    with pytest.raises(ModelingError, match="complete_row_fp32_scale"):
        lower_op(vector_op(
            op_id=op.op_id, deps=op.deps, stage=0, layer=0, requests=("r",),
            elements=rows * width, ops_per_element=0, dtype="bf16",
            family=op.family, phase="decode", recipe=op.recipe, shape=(rows, width),
            read_bytes_override=op.mte2_bytes, write_bytes_override=op.mte3_bytes,
            vector_ops_override=op.vector_ops, returns=op.returns, metadata=bad))


def test_a5_mx_odd_group_scale_cannot_use_floor_width_layout():
    rows, width = 3, 96
    op = _norm("add_rmsnorm_quant", rows, width, quant="mx")
    bad = op.metadata + (
        ("activation_scale_shape", (rows, width // 64, 2)),
        ("activation_scale_bytes", rows * width // 64 * 2),
    )
    with pytest.raises(ModelingError, match="scale_materialization_mismatch"):
        lower_op(vector_op(
            op_id=op.op_id, deps=op.deps, stage=0, layer=0, requests=("r",),
            elements=rows * width, ops_per_element=0, dtype="bf16",
            family=op.family, phase="decode", recipe=op.recipe, shape=(rows, width),
            read_bytes_override=op.mte2_bytes, write_bytes_override=op.mte3_bytes,
            vector_ops_override=op.vector_ops, returns=op.returns, metadata=bad))


def test_a5_plain_int8_norm_quant_has_no_established_kernel():
    with pytest.raises(ModelingError, match="plain_int8_quant_kernel_not_established"):
        lower_op(_norm("rmsnorm_quant", 1, 5120, quant="int8"))


def test_mx_producer_contract_carries_group_and_tail_materialization():
    for width in (31, 32, 33, 96, 100, 255, 5120):
        contract = producer_contract(width)
        groups = (width + 31) // 32
        padded = (groups + 1) // 2 * 2
        assert contract["scale_groups"] == groups
        assert contract["padded_scale_groups"] == padded
        assert contract["scale_shape"] == (padded // 2, 2)
        assert contract["scale_bytes"] == padded
        assert contract["write_bytes"] == width + padded
        assert contract["vector_ops_per_row"] == group32_work(width)["lane_ops"]["total"]
    alg1 = producer_contract(96, scale_alg=1)
    assert alg1["vector_ops"] is None and alg1["vector_ops_per_row"] is None
    assert alg1["scale_shape"] == (2, 2)
    with pytest.raises(ModelingError, match="positive_width"):
        producer_contract(0)


def test_a5_real_mx_norm_quant_chain_matches_consumer_scale():
    raw = json.loads((ROOT / "examples/cases/qwen38-a5.json").read_text())
    options = {"implementation": "vllm_ascend_native_norm",
               "options": {"has_bias": False, "external_gamma_shift": True}}
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {}).update({
        "add_rmsnorm": options, "add_rmsnorm_quant": options,
        "rmsnorm_quant": options})
    case = parse_case(raw)
    ops = target_operators(case, phase="decode", tokens=4, context=2048)
    producers = [
        op for op in ops
        if op.family in {"rmsnorm_quant", "add_rmsnorm_quant"}
        and dict(op.metadata).get("native_norm_kind") in {
            "RmsNormDynamicMxQuant", "AddRmsNormDynamicMxQuant"}]
    assert producers
    by_id = {op.op_id: op for op in ops}
    checked = 0
    for producer in producers:
        meta = dict(producer.metadata)
        if producer.local_shape != (4, 5120):
            continue
        consumers = [op for op in ops if producer.op_id in op.deps]
        assert len(consumers) == 1
        consumer = consumers[0]
        cm = dict(consumer.metadata)
        assert cm["activation_scale_shape"] == meta["native_norm_quant_scale_shape"]
        assert cm["activation_scale_bytes"] == meta["native_norm_quant_scale_bytes"]
        assert cm["activation_scale_format"] == "fp8_e8m0"
        assert by_id[consumer.op_id].family in {"gate_up_projection", "qkv_projection"}
        checked += 1
    assert checked >= 1


def test_dynamic_quant_engine_program_covers_unsampled_rows_and_tail():
    rates = {"vector": 1e6, "mte2": 1e6, "mte3": 1e6, "hbm": 1e6}
    units = {"vector": 4, "mte2": 4, "mte3": 4}
    for row_len, mode in ((1234, "full_db"), (200000, "large")):
        report = quant_timeline(
            rates=rates, units=units, rows=7, row_len=row_len,
            source_bytes=2, y_bytes=1, scale_bytes=4, ub_bytes=64 * 1024,
            aiv=4)
        assert report["mode"] == mode
        assert report["launched"] == 4
        assert report["rows_per_head"] == 2
        assert report["rows_per_tail"] == 1
        assert report["vector_ops"] > 0
        assert report["mte2_bytes"] > 0 and report["mte3_bytes"] > 0
        assert report["mte2_bytes"] + report["mte3_bytes"] == report["hbm_bytes"]
        assert report["makespan_ms"] >= report["head_ms"] >= 0.0
        assert report["makespan_ms"] >= report["tail_ms"] >= 0.0


def test_a5_fused_residual_mx_norm_quant_lowers_through_production_pipeline():
    raw = json.loads((ROOT / "examples/cases/qwen38-a5.json").read_text())
    case = parse_case(raw)
    rows, width = 4, 5120
    elements = rows * width
    residual = vector_op(
        op_id="residual", deps=("x", "saved"), stage=0, layer=0,
        requests=("r",), elements=elements, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="decode", recipe="residual_add",
        read_bytes_override=2 * elements * 2, write_bytes_override=elements * 2)
    normalized = norm(
        name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
        tokens=rows, width=width, model=case.model, mode="decode")
    # The retained A5 Gemma shift is an external producer; this test isolates
    # the fused residual path so the existing residual_norm pass can bind it.
    normalized = replace(normalized, metadata=tuple(
        (key, 0.0 if key == "norm_weight_offset" else value)
        for key, value in normalized.metadata))
    gate_up = gemm_op(
        op_id="gate_up", deps=("norm",), stage=0, layer=0, requests=("r",),
        m=rows, n=64, k=width, quant=case.model.quant, category="dense_mlp",
        phase="decode", recipe="dense_gate_up")
    runtime = replace(
        case.runtime,
        lowering_passes=("activation_quantization", "norm_quant", "residual_norm"),
        kernels={**case.runtime.kernels,
                 "lowerings": {"add_rmsnorm_quant": {
                     "implementation": "vllm_ascend_native_norm",
                     "options": {"has_bias": False,
                                 "external_gamma_shift": False}}}})
    ops = lower_implementations([residual, normalized, gate_up], runtime, "bf16")
    fused = next(op for op in ops if op.family == "add_rmsnorm_quant")
    meta = dict(fused.metadata)
    groups, padded, shape = _band(width, rows)
    assert meta["native_norm_kind"] == "AddRmsNormDynamicMxQuant"
    assert meta["native_norm_quant_scale_shape"] == shape
    assert meta["native_norm_quant_scale_bytes"] == rows * padded
    assert meta["native_norm_residual_bytes"] == elements * 2
    assert fused.vector_ops == 6 * elements + rows * group32_work(width)["lane_ops"]["total"]
    assert fused.mte2_bytes == (2 * rows + 1) * width * 2
    assert fused.mte3_bytes == 3 * elements + rows * padded
    assert {"residual", "activation_scales"} <= set(fused.returns)


@pytest.mark.parametrize("width", (31, 33, 96, 100, 255, 4097))
def test_mx_norm_quant_work_generalizes_to_unsampled_widths(width):
    rows = 7
    selected = _selected(_norm("rmsnorm_quant", rows, width, quant="mx"))
    meta = dict(selected.metadata)
    groups, padded, shape = _band(width, rows)
    assert selected.vector_ops == (
        5 * rows * width + rows * group32_work(width)["lane_ops"]["total"])
    assert meta["native_norm_quant_scale_shape"] == shape
    assert meta["native_norm_quant_scale_bytes"] == rows * padded
    assert meta["native_norm_quant_groups"] == groups
    assert selected.mte3_bytes == rows * width + rows * padded


def test_mx_declared_row_ledger_replaces_the_guard_slot():
    rows, width = 4, 5120
    op = _norm("rmsnorm_quant", rows, width, quant="mx")
    declared = 1234
    metadata = op.metadata + (("mx_vector_ops_per_row", declared),)
    guard = vector_op(
        op_id=op.op_id, deps=op.deps, stage=0, layer=0, requests=("r",),
        elements=rows * width, ops_per_element=0, dtype="bf16",
        family=op.family, phase="decode", recipe=op.recipe, shape=(rows, width),
        read_bytes_override=op.mte2_bytes, write_bytes_override=op.mte3_bytes,
        vector_ops_override=5 * rows * width + rows * width, metadata=metadata)
    selected = _selected(guard)
    meta = dict(selected.metadata)
    assert selected.vector_ops == 5 * rows * width + rows * declared
    assert meta["native_norm_quant_work_source"] == "declared_mx_vector_ops_per_row"
    assert meta["native_norm_quant_vector_ops"] == rows * declared


def test_mx_unknown_work_composition_is_rejected():
    rows, width = 4, 5120
    op = _norm("rmsnorm_quant", rows, width, quant="mx")
    odd = vector_op(
        op_id=op.op_id, deps=op.deps, stage=0, layer=0, requests=("r",),
        elements=rows * width, ops_per_element=0, dtype="bf16",
        family=op.family, phase="decode", recipe=op.recipe, shape=(rows, width),
        read_bytes_override=op.mte2_bytes, write_bytes_override=op.mte3_bytes,
        vector_ops_override=op.vector_ops + 7, metadata=op.metadata)
    with pytest.raises(ModelingError, match="work_composition_unknown"):
        lower_op(odd)


def test_repeated_selection_does_not_double_count_beta_or_mx_ledger():
    int8 = _norm("add_rmsnorm_quant", 2, 5120, quant="int8")
    first = _selected(int8, has_bias=True)
    second = _selected(first, has_bias=True)
    assert second.vector_ops == first.vector_ops
    assert second.mte2_bytes == first.mte2_bytes
    assert dict(second.metadata)["native_norm_extra_vector_ops"] == 2 * 5120
    mx = _norm("add_rmsnorm_quant", 2, 5120, quant="mx")
    once = _selected(mx)
    twice = _selected(once)
    assert twice.vector_ops == once.vector_ops
    assert dict(twice.metadata)["native_norm_quant_vector_ops"] == 2 * 64000


def test_unquantized_add_rms_norm_does_not_invent_explicitly_absent_residual():
    rows, width = 2, 5120
    # A non-empty return list without "residual" is an explicit absence, such
    # as an O0 final norm.  Do not add the output or a virtual byte; the
    # remaining normalized store and rstd are still billed exactly once.
    op = _norm("add_rmsnorm", rows, width, returns=("norm_output",),
               write=rows * width * 2)
    selected = _selected(op)
    meta = dict(selected.metadata)
    assert "residual" not in selected.returns
    assert meta["native_norm_residual_output"] is False
    assert meta["native_norm_residual_bytes"] == 0
    assert meta["native_norm_residual_billing"] == "absent"
    assert selected.mte3_bytes == rows * width * 2 + rows * 4


def test_unquantized_add_rms_norm_empty_returns_does_not_invent_unbacked_residual():
    rows, width = 2, 5120
    # Empty returns are ambiguous, so the residual is added only when mte3
    # really covers both stores.  A one-store partial producer must not grow a
    # virtual 20480 B residual output in the report.
    op = _norm("add_rmsnorm", rows, width, write=rows * width * 2)
    selected = _selected(op)
    meta = dict(selected.metadata)
    assert "residual" not in selected.returns
    assert meta["native_norm_residual_output"] is False
    assert meta["native_norm_residual_bytes"] == 0
    assert meta["native_norm_residual_billing"] == "absent"
    assert meta["native_norm_residual_format"] is None
    assert meta["native_norm_store_consistent"] is True


def test_repeated_unquantized_add_rms_norm_does_not_rewrite_rstd_or_residual():
    first = _selected(_norm("add_rmsnorm", 2, 5120))
    second = _selected(first)
    meta = dict(second.metadata)
    assert second.mte3_bytes == first.mte3_bytes
    assert meta["native_norm_rstd_bytes"] == 2 * 4
    assert meta["native_norm_residual_bytes"] == 2 * 5120 * 2
    assert second.returns.count("rstd") == 1
    assert second.returns.count("residual") == 1
