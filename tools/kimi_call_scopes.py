"""Standalone Kimi call-scope analysis helpers.

[Role]
  Build a descriptor-conditioned GroupedMatmulSituQuant semantic reference and
  a source-conditioned ChunkKdaFwd stage/call ledger.  This is analysis only;
  the runtime, catalog, schema and scoreboard do not import it.

[Invariants]
  * Descriptor shapes/dtypes select geometry; measured durations, occurrence
    counts and capture activity never enter either result.
  * Situ is an explicit gate-up -> SiTU -> output-quantization DAG.  The matrix
    is an aggregate resource envelope because group_list values are unknown;
    all declared weight banks are charged, with no invented equal-row split.
  * The dummy down consumer only requests the output quantizer.  Down itself is
    removed; the generated ``down.quantize`` op is retained.
  * The KDA ledger is SOURCE-conditioned stage membership/formula evidence.  It
    is not native timing, a historical launch reconstruction or a replacement
    for the current semantic member program.

[Timeouts & Locks]
  Pure constant-time arithmetic over a bounded descriptor list; no I/O, jobs,
  devices, network or locks.

[Side-Effect Policy]
  Constructs immutable operators and JSON-compatible ledger data only.  It
  writes no files and mutates no global state.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math

from mindie_sim.catalog.ops import activation_scales, gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.gated_activation import gated_activation
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.schema.common import ModelingError, dtype_bytes

SITU_INPUT, SITU_DOWN = "situ.input", "situ.down"
SITU_QUANT = SITU_DOWN + ".quantize"
_MATRIX_RECIPE = "matrix_envelope"
_REFERENCE = "GroupedMatmulSituQuant"
_REFERENCE_STATUS = "declared_semantic_reference_not_native_fused_traffic_or_abi"
_SOURCE_REVISION = "ae923905948f360f4b7ab06df52d0789a0c9316e"
_KDA_SOURCES = tuple({"name": name, "path": path, "sha256": digest, "lines": lines}
        for name, path, digest, lines in (
        ("aclnn_chunk_kda_fwd", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp", "4db63cf9151d52d8b6d101abc05f3af1f42eaa88435483662dc9cb5d72370fa2", [[650, 669], [770, 815]]),
        ("chunk_kda_fwd_kernel", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp", "0ea787730736f4ed92a35e3bafa6e54417c84716fc6aed381a339259abfa74d0", [[15, 70]]),
        ("chunk_kda_fwd_common", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h", "e98c17e528d673a9b6c2f0bbbfce83746bcccf0afaeb413ea0ac41a2809e74f8", [[164, 184], [225, 240]]),
        ("chunk_kda_fwd_prepare", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h", "b808a1a1c6e21c0a3098433764675a1f507b04bc8f77d22eaffedd2c4de866c7", [[2865, 2933]]),
        ("chunk_kda_fwd_post_wu", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_post_wu.h", "89cfa406c18465b8d34c527791d1611ad3bc423253fe613d8f9a70920656860d", [[1710, 1754], [2001, 2020]]),
        ("chunk_kda_fwd_finalize", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_finalize.h", "f760d3c30f37b58fc2cf28678951f83f3e5a734859c4e058d8e0f0078a9aac01", [[969, 995]]),
        ("chunk_kda_fwd_tiling_impl", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h", "e7542eff7113d97ce069b7ec4f57b01809f4894f1e3b5e801add5f557991fe59", [[12, 39]]),
        ("chunk_kda_fwd_tiling", "/Users/maoxx241/code/vllm-ascend-v41-private/csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp", "72148ca10b26ba4f34919c11e973c3bbcd696cfbc9a938c3cc3bf724e41f813c", [[208, 259]]),
        ("generic_recurrent", "/Users/maoxx241/code/sim-rewrite-20261027/mindie_sim/prediction/recurrent.py", "face2ef1b7f1ed121c2ae9b3cb686fa79773b695f46fa0edaf577c856ca3df5f", [[100, 143], [200, 237]]),
        ("a5_kimi_k3_fixture", "/Users/maoxx241/code/profiler/tests/fixtures/fusion_contracts/a5_kimi_k3.csv", "00beb09f90dbb57c12474b6cc0d0db724223b5c25b724d9b13b222da0aec378b", [[45, 45], [71, 72], [80, 80]]),
))


def _mapping(value, reason):
    if not isinstance(value, Mapping):
        raise ModelingError(reason)
    return value


def _descriptors(value):
    value = _mapping(value, "kimi_scope_descriptors_invalid")
    if "descriptors" in value and "input_shapes" not in value:
        value = _mapping(value["descriptors"], "kimi_scope_descriptors_invalid")
    for key in ("input_shapes", "output_shapes", "input_dtypes", "output_dtypes"):
        if key not in value or not isinstance(value[key], (list, tuple)):
            raise ModelingError(f"kimi_scope_descriptor_missing:{key}")
    if len(value["input_shapes"]) != len(value["input_dtypes"]):
        raise ModelingError("kimi_scope_input_descriptor_count_mismatch")
    if len(value["output_shapes"]) != len(value["output_dtypes"]):
        raise ModelingError("kimi_scope_output_descriptor_count_mismatch")
    return value


def _shape(value, reason):
    if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ModelingError(reason)
    return tuple(int(item) for item in value)


def _quant(value):
    if isinstance(value, QuantSpec):
        return value
    raw = dict(_mapping(value, "kimi_scope_quant_invalid"))
    for key in ("weight_block", "exclusions", "source", "notes"):
        if isinstance(raw.get(key), list):
            raw[key] = tuple(raw[key])
    try:
        return QuantSpec(**raw)
    except TypeError as exc:
        raise ModelingError("kimi_scope_quant_invalid") from exc


def _situ_geometry(descriptors, quant):
    desc = _descriptors(descriptors)
    ins, outs, idt, odt = (desc[key] for key in (
        "input_shapes", "output_shapes", "input_dtypes", "output_dtypes"))
    if len(ins) != 5 or len(outs) != 2:
        raise ModelingError("kimi_situ_descriptor_inventory_mismatch")
    input_shape, input_scales, weight_shape, weight_scales, groups = (
        _shape(value, f"kimi_situ_input_{index}_geometry_invalid")
        for index, value in enumerate(ins))
    output_shape, output_scales = (
        _shape(value, f"kimi_situ_output_{index}_geometry_invalid")
        for index, value in enumerate(outs))
    if len(input_shape) != 2 or len(output_shape) != 2 or len(weight_shape) != 3:
        raise ModelingError("kimi_situ_matrix_rank_invalid")
    m, k = input_shape
    experts, gate_n, packed_k = weight_shape
    if m < 1 or k < 1 or experts < 1 or gate_n < 2 or packed_k != math.ceil(k / 2):
        raise ModelingError("kimi_situ_matrix_geometry_invalid")
    if output_shape[0] != m or output_shape[1] != gate_n // 2:
        raise ModelingError("kimi_situ_output_must_be_half_gate_up_width")
    width = output_shape[1]
    if groups != (experts,) or idt[4] not in {"INT64", "DT_INT64"}:
        raise ModelingError("kimi_situ_group_list_contract_invalid")
    expected = ({"DT_FLOAT8_E4M3FN", "DT_FLOAT8_E4M3"}, {"DT_FLOAT8_E8M0", "DT_FLOAT8_E8M0FN"},
                {"DT_FLOAT4_E2M1", "DT_FLOAT4_E2M1FN"}, {"DT_FLOAT8_E8M0", "DT_FLOAT8_E8M0FN"})
    if any(idt[index] not in allowed for index, allowed in enumerate(expected)):
        raise ModelingError("kimi_situ_input_representation_mismatch")
    if odt[0] not in expected[0] or odt[1] not in expected[1]:
        raise ModelingError("kimi_situ_output_representation_mismatch")
    if quant.format_for("routed_expert") not in {"mxfp4", "fp4", "fp4_e2m1"}:
        raise ModelingError("kimi_situ_quant_must_declare_fp4_weights")
    act_group, expert_group = int(quant.activation_group_size or 0), int(quant.expert_block or 0)
    if act_group < 1 or expert_group < 1:
        raise ModelingError("kimi_situ_quant_groups_must_be_positive")
    if input_scales[0] != m or math.prod(input_scales[1:]) != math.ceil(k / act_group):
        raise ModelingError("kimi_situ_input_scale_geometry_mismatch")
    if weight_scales[:2] != (experts, gate_n) or math.prod(weight_scales[2:]) != math.ceil(k / expert_group):
        raise ModelingError("kimi_situ_weight_scale_geometry_mismatch")
    if output_scales[0] != m or math.prod(output_scales[1:]) != math.ceil(width / act_group):
        raise ModelingError("kimi_situ_output_scale_geometry_mismatch")
    for label, cols, shape in (("input", k, input_scales), ("output", width, output_scales)):
        _, scale_format, scale_shape, scale_bytes = activation_scales(
            quant, "fp8_e4m3", m, cols)
        if (scale_format != "fp8_e8m0" or tuple(scale_shape) != shape
                or scale_bytes != math.prod(shape)):
            raise ModelingError(f"kimi_situ_{label}_scale_declaration_mismatch")
    return {
        "m": m, "k": k, "experts": experts, "gate_up_n": gate_n, "output_width": width,
        "activation_group": act_group, "expert_group": expert_group,
        "input_payload_bytes": m * k, "input_scale_bytes": m * math.ceil(k / act_group),
        "weight_payload_bytes": experts * gate_n * math.ceil(k / 2),
        "weight_scale_bytes": experts * gate_n * math.ceil(k / expert_group),
        "gate_bf16_bytes": m * gate_n * 2, "situ_bf16_bytes": m * width * 2,
        "output_payload_bytes": m * width, "output_scale_bytes": m * math.ceil(width / act_group),
        "input_scale_shape": input_scales, "output_scale_shape": output_scales,
    }


def _plan(case):
    if not case.model.layer_plans:
        raise ModelingError("kimi_situ_reference_requires_case_layer_plan")
    plan = case.model.layer_plans[0]
    if plan.ffn_activation != "situ":
        raise ModelingError("kimi_situ_reference_requires_case_plan_ffn_activation_situ")
    return plan


def _op_inventory(ops):
    return [{key: (list(op.deps) if key == "deps" else getattr(op, key))
             for key in ("op_id", "family", "recipe", "deps", "flops", "vector_ops",
                         "hbm_bytes", "mte2_bytes", "mte3_bytes", "dtype", "layout")}
            for op in ops]


def _validate_situ_operands(gate, quantize, geometry):
    """The executed category overrides must agree with the descriptor ledger."""
    gate_meta, output_meta = dict(gate.metadata), dict(quantize.metadata)
    if (gate_meta["weight_format"] not in {"mxfp4", "fp4", "fp4_e2m1"}
            or gate_meta["packed_weight_bytes"] != (
                geometry["weight_payload_bytes"] + geometry["weight_scale_bytes"])):
        raise ModelingError("kimi_situ_effective_weight_declaration_mismatch")
    for label, meta in (("input", gate_meta), ("output", output_meta)):
        if (meta.get("activation_scale_format") != "fp8_e8m0"
                or tuple(meta.get("activation_scale_shape", ())) != geometry[label + "_scale_shape"]
                or meta.get("activation_scale_bytes") != geometry[label + "_scale_bytes"]):
            raise ModelingError(f"kimi_situ_effective_{label}_scale_declaration_mismatch")
    if quantize.mte3_bytes != geometry["output_payload_bytes"] + geometry["output_scale_bytes"]:
        raise ModelingError("kimi_situ_effective_quantized_output_bytes_mismatch")


def situ_reference(case, descriptors, quant):
    """Return the declared grouped gate-up + SiTU + output-quantization DAG."""
    case, quant = parse_case(case), _quant(quant)
    plan, g = _plan(case), _situ_geometry(descriptors, quant)
    m, n, k, width = g["m"], g["gate_up_n"], g["k"], g["output_width"]
    declaration = (
        ("semantic_reference", _REFERENCE), ("reference_status", _REFERENCE_STATUS),
        ("group_list_status", "values_absent_from_descriptors"),
        ("weight_bank_policy", "all_declared_weight_banks_charged"),
        ("matrix_recipe", _MATRIX_RECIPE),
        ("matrix_layout_basis", "aggregate_unknown_group_list_no_equal_rows_no_tiled_groups"),
        ("down_projection_body", "omitted"), ("down_quantizer_op_id", SITU_QUANT))
    gate = gemm_op(
        op_id="situ.gate_up", deps=(SITU_INPUT,), stage=0, layer=plan.index, requests=(),
        m=m, n=n, k=k, quant=quant, category="routed_expert", phase="prefill",
        family="gate_up_projection", recipe=_MATRIX_RECIPE, weight_count=g["experts"],
        weight_rows=n, weight_cols=k, activation_dtype="fp8_e4m3", output_dtype="bf16",
        metadata=declaration)
    activation = gated_activation(
        plan=plan, op_id="situ.activation", deps=(gate.op_id,), rows=m, width=width,
        dtype="bf16", stage=0, requests=(), phase="prefill", recipe="situ")
    activation = replace(activation, metadata=activation.metadata + declaration)
    dummy = gemm_op(
        op_id=SITU_DOWN, deps=(activation.op_id,), stage=0, layer=plan.index, requests=(),
        m=m, n=width, k=width, quant=quant, category="routed_expert", phase="prefill",
        family="down_projection", recipe="dummy_output_consumer",
        activation_dtype="fp8_e4m3", output_dtype="bf16",
        metadata=(("quantization_request_only", True),))
    quantize = next(op for op in activation_quantization((activation, dummy), "bf16")
                    if op.op_id == SITU_QUANT)
    quantize = replace(quantize, metadata=quantize.metadata + declaration + (
        ("quantizer_role", "situ_output_quantization"), ("dummy_consumer", SITU_DOWN),
        ("dummy_consumer_in_returned_work", False)))
    _validate_situ_operands(gate, quantize, g)
    ops = (gate, activation, quantize)
    if any(op.plan_required for op in ops):
        raise ModelingError("kimi_situ_reference_must_not_claim_source_boundary")
    gate_bf16, situ_bf16 = int(g["gate_bf16_bytes"]), int(g["situ_bf16_bytes"])
    ledger = {
        "schema": "kimi-call-scopes/situ-semantic-reference/1",
        "reference": _REFERENCE, "reference_status": _REFERENCE_STATUS,
        "boundary": "gate_up_matrix -> SiTU -> output_quantization",
        "group_list": {"values": "absent", "weight_banks": g["experts"],
                       "policy": "charge every declared weight bank; do not invent equal rows or a tiled expert distribution"},
        "matrix_recipe": {"id": _MATRIX_RECIPE, "kind": "resource_envelope",
                          "reason": "unknown group_list; aggregate logical FLOPs and all weight banks"},
        "activation": {"kind": "situ", "beta": plan.activation_situ_beta,
                       "linear_beta": plan.activation_situ_linear_beta,
                       "parameter_source": "case.model.layer_plans[0]",
                       "scalar_expression_ops": int(activation.vector_ops)},
        "geometry": {"m": m, "logical_k": k, "gate_up_n": n,
                     "weight_banks": g["experts"], "output_width": width,
                     "activation_group": g["activation_group"], "expert_group": g["expert_group"]},
        "work": {"gate_up_flops": int(gate.flops), "situ_scalar_ops": int(activation.vector_ops),
                 "output_quant_vector_ops": int(quantize.vector_ops)},
        "traffic": {
            "input_payload_bytes": g["input_payload_bytes"],
            "input_scale_bytes": g["input_scale_bytes"],
            "weight_payload_bytes": g["weight_payload_bytes"],
            "weight_scale_bytes": g["weight_scale_bytes"],
            "gate_bf16_output_bytes": gate_bf16, "situ_bf16_output_bytes": situ_bf16,
            "output_payload_bytes": g["output_payload_bytes"],
            "output_scale_bytes": g["output_scale_bytes"],
            "total_op_hbm_bytes": int(sum(op.hbm_bytes for op in ops)),
            "mte2_bytes": int(sum(op.mte2_bytes for op in ops)),
            "mte3_bytes": int(sum(op.mte3_bytes for op in ops))},
        "workspaces": {
            "policy": "preserve_workspaces", "gate_bf16_bytes": gate_bf16,
            "situ_bf16_bytes": situ_bf16,
            "intermediate_bf16_read_write_bytes": 2 * gate_bf16 + 2 * situ_bf16,
            "note": "Gate and SiTU BF16 outputs remain materialized; the output quantizer reads the SiTU result."},
        "inventory": {"external_inputs": [SITU_INPUT], "ops": _op_inventory(ops),
                      "retained_quantizer": SITU_QUANT, "omitted_op": SITU_DOWN,
                      "omitted_op_note": "dummy down consumer requested the quantizer but is not returned work"},
        "limitations": [
            "Declared semantic reference only; not a fused native kernel, ABI or traffic identity.",
            "Group-list values and per-expert row counts are absent; the envelope does not assert equal rows or a native grouped schedule.",
            "Input FP8 payload and E8M0 scales are already inputs; no input quantizer is added.",
            "BF16 gate/SiTU workspaces are retained analytically and are not an instruction-level schedule."],
    }
    recipes = RecipeBook.from_mapping({"recipes": {_MATRIX_RECIPE: {"kind": "envelope"}}})
    return {"ops": ops, "recipes": recipes, "ledger": ledger}


def _dtype_name(value):
    return str(value).strip().upper()


def _kda_storage_dtype(value):
    name = _dtype_name(value)
    if name in {"DT_BF16", "BF16"}:
        return "bf16"
    if name in {"DT_FLOAT", "FLOAT", "DT_FLOAT32", "FLOAT32"}:
        return "fp32"
    raise ModelingError(f"kimi_kda_output_dtype_unsupported:{value}")


def _kda_geometry(descriptors, chunk_size):
    desc = _descriptors(descriptors)
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size not in (64, 128)):
        raise ModelingError("kimi_kda_source_chunk_size_must_be_64_or_128")
    ins, idt = desc["input_shapes"], desc["input_dtypes"]
    q, k, v, gate = (_shape(ins[index], "kimi_kda_qkvg_shape_invalid") for index in range(4))
    if len(q) != 4 or q != k or q != gate or v[:3] != q[:3] or len(v) != 4:
        raise ModelingError("kimi_kda_qkvg_geometry_mismatch")
    batch, tokens, heads, key_dim = q
    value_dim = v[3]
    if batch != 1 or min(tokens, heads, key_dim, value_dim) < 1:
        raise ModelingError("kimi_kda_geometry_must_be_positive_single_sequence")
    if any(_dtype_name(idt[index]) not in {"DT_BF16", "BF16"} for index in range(4)):
        raise ModelingError("kimi_kda_qkvg_must_be_bf16")
    beta = _shape(ins[4], "kimi_kda_beta_shape_invalid")
    a_log = _shape(ins[5], "kimi_kda_a_log_shape_invalid")
    dt_bias = _shape(ins[6], "kimi_kda_dt_bias_shape_invalid")
    initial = _shape(ins[7], "kimi_kda_initial_state_shape_invalid")
    if (math.prod(beta) != batch * tokens * heads or a_log != (heads,)
            or dt_bias != (heads * key_dim,) or initial != (batch, heads, key_dim, value_dim)):
        raise ModelingError("kimi_kda_gate_or_state_geometry_mismatch")
    if _dtype_name(idt[4]) not in {"DT_BF16", "BF16", "DT_FLOAT", "FLOAT", "DT_FLOAT32"}:
        raise ModelingError("kimi_kda_beta_dtype_unsupported")
    if any(_dtype_name(idt[index]) not in {"DT_FLOAT", "FLOAT", "DT_FLOAT32"} for index in (5, 6, 7)):
        raise ModelingError("kimi_kda_gate_and_state_parameters_must_be_fp32")
    if len(ins) == 10:
        cu, chunk_index_dtype = _shape(ins[8], "kimi_kda_cu_seqlens_shape_invalid"), idt[9]
        chunk_indices = _shape(ins[9], "kimi_kda_chunk_indices_shape_invalid")
        if cu != (2,) or _dtype_name(idt[8]) not in {"DT_INT64", "INT64"}:
            raise ModelingError("kimi_kda_cu_seqlens_contract_invalid")
    elif len(ins) == 9:
        cu, chunk_indices, chunk_index_dtype = None, _shape(ins[8], "kimi_kda_chunk_indices_geometry_invalid"), idt[8]
    else:
        raise ModelingError("kimi_kda_input_inventory_mismatch")
    if _dtype_name(chunk_index_dtype) not in {"DT_INT64", "INT64"}:
        raise ModelingError("kimi_kda_chunk_indices_must_be_int64")
    total_chunks = math.ceil(tokens / chunk_size)
    if chunk_indices != (2 * total_chunks,):
        raise ModelingError("kimi_kda_chunk_indices_geometry_mismatch")
    pairs = (tokens // chunk_size) * chunk_size * chunk_size + (tokens % chunk_size) ** 2
    labels = ("attn_out", "final_state", "gk", "aqk", "akk", "w", "u", "qg",
              "kg", "v_new", "h", "qg_scaled", "u_seed")
    odt = desc["output_dtypes"]
    if len(desc["output_shapes"]) != 13 or len(odt) != 13:
        raise ModelingError("kimi_kda_stage_output_contract_must_have_13_tensors")
    outputs = []
    for label, value, dtype in zip(labels, desc["output_shapes"], odt):
        shape, storage_dtype = _shape(value, "kimi_kda_output_shape_invalid"), _kda_storage_dtype(dtype)
        outputs.append({"name": label, "shape": list(shape), "dtype": storage_dtype,
                        "bytes": int(math.prod(shape) * dtype_bytes(storage_dtype))})
    dense_fastpath = (cu is None and _dtype_name(idt[0]) in {"DT_BF16", "BF16"}
                      and chunk_size == 64 and key_dim == 128 and value_dim == 128
                      and tokens % chunk_size == 0)
    return {"batch": batch, "tokens": tokens, "heads": heads, "dk": key_dim,
            "dv": value_dim, "chunk": chunk_size, "total_chunks": total_chunks,
            "pairs": pairs, "cu_seqlens_present": cu is not None,
            "dense_fastpath": dense_fastpath, "split_stages": total_chunks > 1 and not dense_fastpath,
            "launches": 4 if total_chunks > 1 and not dense_fastpath else 1,
            "outputs": outputs}


def _stage(name, index, components, members, products, unresolved, generic):
    return {"stage_index": index, "name": name,
            "exposes_same_13_output_descriptors": True,
            "source_components": components, "current_semantic_members": members,
            "recognized_product_work": products,
            "recognized_product_flops": int(sum(item["flops"] for item in products)),
            "unresolved_work": unresolved, "generic_member_work": generic,
            "mapping_status": "semantic_projection_only_not_source_phase_timing"}


def _kda_stages(g):
    t, h, dk, dv, c, pairs = g["tokens"], g["heads"], g["dk"], g["dv"], g["chunk"], g["pairs"]
    prepare, qg_h, aqk_v = 4 * pairs * h * dk, 2 * t * h * dk * dv, 2 * pairs * h * dv
    generic_chunk, generic_wy = prepare, 2 * pairs * h * (dk + dv)
    generic_scan = 4 * t * h * dk * dv
    generic_output = qg_h + aqk_v
    cubes = (t // c) * c ** 3 + (t % c) ** 3
    stages = [
        _stage("GATE_PREPARE", 0,
               ["gate cumsum (raw G BF16; computeGateInPrepare false)", "QK and KK score products",
                "causal mask/triangular inversion/prepare seeds"],
               ["gate", "chunk_products", "triangular_solve"],
               [{"name": "QK", "formula": "2*pairs*heads*dk", "flops": int(2 * pairs * h * dk)},
                {"name": "KK", "formula": "2*pairs*heads*dk", "flops": int(2 * pairs * h * dk)}],
               ["causal-mask and triangular-inversion scalar work", "gate cumsum traffic and soft-clip parameter work",
                "prepare seed/output read-write traffic"],
               [{"semantic_member": "chunk_products", "generic_flops": int(generic_chunk)},
                {"semantic_member": "triangular_solve", "generic_flops": int(2 * cubes * h // 3)}]),
        _stage("POST_WU", 1, ["w=akk*wSeed", "u=akk*uSeed", "kg finalization"], ["wy_transform"], [],
               ["source FLOP formula for w/u/kg transforms is not established here", "elementwise and finalization traffic"],
               [{"semantic_member": "wy_transform", "generic_flops": int(generic_wy)}]),
        _stage("FWD_H", 2, ["GDNFwdHKernel<T,float,float,float,...>", "recurrent H/vNew/finalState updates"],
               ["recurrent_scan"], [],
               ["sequence recurrence work and resource ownership", "state/output traffic and final-state write"],
               [{"semantic_member": "recurrent_scan", "generic_flops": int(generic_scan)}]),
        _stage("FINALIZE", 3, ["qgScaled@h", "aqk@vNew", "output accumulation/cast/store"],
               ["recurrent_output"],
               [{"name": "qgScaled@h", "formula": "2*tokens*heads*dk*dv", "flops": int(qg_h)},
                {"name": "aqk@vNew", "formula": "2*pairs*heads*dv", "flops": int(aqk_v)}],
               ["output accumulation, cast and store traffic", "final-state/h export traffic"],
               [{"semantic_member": "recurrent_output", "generic_flops": int(generic_output)}]),
    ]
    mismatch = {
        "summary": "QK and KK products are in prepare; qgScaled@h and aqk@vNew are in finalize; source phase traffic and ownership are not inferred from arithmetic equality",
        "status": "recognized_products_aligned",
        "flops_shifted_from_prepare_to_output": 0,
        "generic_chunk_products_flops": int(generic_chunk), "source_prepare_flops": int(prepare),
        "generic_recurrent_output_flops": int(generic_output), "source_finalize_flops": int(qg_h + aqk_v),
        "combined_flops_equal": generic_chunk + generic_output == prepare + qg_h + aqk_v,
        "combined_flops": int(prepare + qg_h + aqk_v),
        "prior_to_revision": {
            "fixed_by": "10c2e531",
            "flops_previously_shifted_from_prepare_to_output": int(2 * pairs * h * dk),
            "total_work_changed": False,
        },
        "interpretation": "equal combined FLOPs do not justify relabeling semantic member timings as source phase timings",
    }
    evidence = {"GATE_PREPARE": ["chunk_kda_fwd_common:164-184", "chunk_kda_fwd_prepare:2865-2933"],
                "POST_WU": ["chunk_kda_fwd_post_wu:1710-1754", "chunk_kda_fwd_post_wu:2001-2020"],
                "FWD_H": ["chunk_kda_fwd_common:225-240"], "FINALIZE": ["chunk_kda_fwd_finalize:969-995"]}
    for stage in stages:
        stage["evidence"] = evidence[stage["name"]]
    return stages, mismatch, {"source_prepare_flops": int(prepare),
        "source_finalize_flops": int(qg_h + aqk_v), "source_finalize_qg_h_flops": int(qg_h),
        "source_finalize_aqk_v_flops": int(aqk_v)}


def kda_stage_ledger(descriptors, *, chunk_size=64):
    """Return source-conditioned ChunkKdaFwd call/stage membership evidence."""
    g = _kda_geometry(descriptors, chunk_size)
    stages, mismatch, recognized = _kda_stages(g)
    if not g["split_stages"]:
        stages = [_stage("FULL", -1, ["single source call; split-stage membership not selected"],
                         ["current complete KDA semantic core"], [],
                         ["stage-specific work is not exposed by this branch",
                          "native timing and historical launch count are not reconstructed"], [])]
        mismatch = {"summary": "split-stage source ledger is inactive for these descriptors",
                    "flops_shifted_from_prepare_to_output": 0, "generic_chunk_products_flops": None,
                    "source_prepare_flops": None, "generic_recurrent_output_flops": None,
                    "source_finalize_flops": None, "combined_flops_equal": None,
                    "combined_flops": None,
                    "interpretation": "generic semantic members are not source stage timings"}
    output_bytes = int(sum(item["bytes"] for item in g["outputs"]))
    known_workspace = int(2 * g["tokens"] * g["heads"] * g["chunk"] * 4)
    ledger = {
        "schema": "kimi-call-scopes/kda-source-stage-ledger/1",
        "source_condition": {
            "pinned_revision": _SOURCE_REVISION,
            "predicate": "splitStages = IsAscend950 && totalChunks > 1 && !dense_fastpath",
            "formula_evidence": ["aclnn_chunk_kda_fwd:650-669,770-815",
                                 "chunk_kda_fwd_tiling:208-259", "chunk_kda_fwd_tiling_impl:12-39"],
            "soc_assumption": "pinned source branch evaluated for Ascend950; descriptors do not carry SoC identity",
            "total_chunks": g["total_chunks"], "total_chunks_gt_1": g["total_chunks"] > 1,
            "cu_seqlens_present": g["cu_seqlens_present"],
            "dense_fastpath": {"predicate": "cu_seqlens is absent && q is BF16 && chunk==64 && dk==128 && dv==128 && tokens%chunk==0",
                               "selected": g["dense_fastpath"]},
            "split_stages": g["split_stages"], "launches": g["launches"],
            "pinned_capture_note": "captured cu_seqlens[2] is a present descriptor; its values are not used to infer the stage count",
            "stage_scalar_present": False},
        "geometry": {"tokens": g["tokens"], "heads": g["heads"], "dk": g["dk"], "dv": g["dv"],
                     "chunk_size": g["chunk"], "total_chunks": g["total_chunks"], "pairs": g["pairs"]},
        "stage_ledger": stages, "recognized_product_work": recognized,
        "semantic_phase_mismatch": mismatch,
        "storage": {
            "output_contract": {"descriptor_count": len(g["outputs"]),
                                "same_13_tensor_outputs_for_every_stage": True,
                                "tensors": g["outputs"], "unique_output_footprint_bytes": output_bytes},
            "workspace": {"known_allocations": [{"name": "prepare_two_fp32_scores",
                                                  "formula": "2*tokens*heads*chunk*4",
                                                  "bytes": known_workspace}],
                          "known_workspace_bytes": known_workspace,
                          "unresolved": ["solve/score scratch and queue slots", "stage-specific buffer offsets",
                                         "compiled tiling workspace total"]},
            "traffic": {"status": "unresolved_source_stage_traffic", "total_bytes": None,
                        "reason": "output/workspace storage is not per-stage read/write traffic; the same 13 descriptors are reused across stage launches and must not be multiplied by launch count"}},
        "non_claims": ["not native timing or a reconstructed historical number of calls",
                       "aggregate measured durations and native launch count are not divided by stage count",
                       "semantic member timings are not relabeled as source stage timings",
                       "output and workspace footprint is storage, not total traffic"],
        "sources": [dict(item) for item in _KDA_SOURCES],
        "limitations": [
            "Descriptors carry no group_list values, stage scalar, timing samples or occurrence counts.",
            "The four-launch count is source-conditioned by the pinned predicate and the present cu_seqlens descriptor, not recovered from native events.",
            "POST_WU and FWD_H formulas/traffic remain unresolved; no zero is a claim that those stages are free.",
            "Unknown group/row distributions remain approximated by the generic semantic member program and must not be relabeled as source phases."],
    }
    return ledger
