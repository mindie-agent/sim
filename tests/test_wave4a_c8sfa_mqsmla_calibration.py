"""Wave 4A: C8 sparse flash attention and MQSMLA cross-SKU calibration.

[Role] Prove the two arch35 sparse-attention programs conserve native work and
  reproduce the paired ``Ascend950PR_958b`` / ``Ascend950DT_9582`` silicon
  measurements from declaration geometry alone.
[Invariants] Every latency is the Engine's own makespan over declared ``Work``;
  no fitted parallel exponent, per-model multiplier or envelope division.  The
  legacy 512-key three-slot arch22 schedule keeps its own admission and numbers.
[Timeouts & Locks] Tier 1 in-memory plans only; no device, capture or locks.
[Side-Effect Policy] Read-only over catalog cases and builder-local records.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs
from mindie_sim.prediction.family_mqsmla import (
    FORMATS, RECIPE as MQSMLA_RECIPE, declared_quant_mode, quant_mode_of,
)
from mindie_sim.prediction.indexed_attention import owner_row_tiles
from mindie_sim.prediction.indexed_quant_attention import (
    C8_KEY_TILE, C8_PACKED_ROW_BYTES, C8_TILE_SLOTS,
    SCHEDULE_ARCH22, SCHEDULE_ARCH35, build as build_packed, prefix_partition,
)
from mindie_sim.prediction.latent_dcp import PAIRED_SFA_CALLS, paired_sfa_calls
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mqsmla import (
    KV_QUANT_BF16, KV_QUANT_FP8, KV_QUANT_MXFP8, KV_QUANT_ROW_BYTES,
    METADATA_PROGRAM_MS, quant_external_fill,
)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


ROOT = Path(__file__).resolve().parents[1]
GLM_CASE = (ROOT / "examples/cases/high-concurrency-20261030/glm78/"
            "a3_glm53_w4a8_tp16_tpot88_c8.json")
V41_CASE = (ROOT / "examples/cases/high-concurrency-20261030/v41-dp8/"
            "a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1.json")
SKUS = (("PR", "Ascend950PR_958b"), ("DT", "Ascend950DT_9582"))
C8_PROGRAM = {"kind": "indexed_attention_packed_int8", "scratch_policy": "offchip",
              "compact": True, "operand_modes": True,
              "schedule": SCHEDULE_ARCH35}
#: Paired ``(PR, DT)`` cold-L2 silicon anchors of the two sparse MLA kernels.
C8_TARGETS = (
    # label, sequences, query, context, topk, heads, prefix, PR us, DT us, tol
    ("glm_s6_h64_k2048", 88, 6, 131072, 2048, 64, 124518, 1076.78, 1027.37, 0.10),
    ("glm_s1_h64_k2048", 88, 1, 131072, 2048, 64, 124518, 234.90, 215.38, 0.12),
    ("kimi_h8_k256", 96, 6, 16384, 256, 8, 15564, 128.14, 128.81, 0.12),
    ("kimi_h16_k256", 96, 6, 16384, 256, 16, 15564, 130.32, 129.58, 0.12),
    ("kimi_h48_k256", 96, 6, 16384, 256, 48, 15564, 155.59, 136.15, 0.12),
)
MQSMLA_MODES = {
    KV_QUANT_BF16: {"window": {"dtype": "bf16"}, "compressed_kv": {"dtype": "bf16"}},
    KV_QUANT_FP8: {
        "window": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "bf16"},
        "compressed_kv": {"dtype": "fp8_e4m3", "group_size": 32,
                          "scale_dtype": "bf16"}},
    KV_QUANT_MXFP8: {
        "window": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "uint8"},
        "compressed_kv": {"dtype": "fp8_e4m3", "group_size": 32,
                          "scale_dtype": "uint8"}},
}
MQSMLA_TARGETS = (
    # label, sequences, kv_quant_mode, prefix, layer, PR us, DT us
    ("v41_swa_bf16", 160, KV_QUANT_BF16, 0, None, 146.50, 85.90),
    ("v41_swa_fp8", 160, KV_QUANT_FP8, 0, 0, 140.86, 106.49),
    ("v41pro_swa_fp8", 80, KV_QUANT_FP8, 0, 0, 77.10, 63.26),
    ("v41_csa_fp8", 160, KV_QUANT_FP8, 124518, 2, 445.29, 419.16),
    ("v41_csa_mxfp8", 160, KV_QUANT_MXFP8, 124518, 2, 631.99, 636.99),
    ("v41pro_csa_fp8", 80, KV_QUANT_FP8, 124518, 2, 231.94, 219.44),
    ("v41pro_csa_mxfp8", 80, KV_QUANT_MXFP8, 124518, 2, 329.20, 333.89),
)


def _load(path):
    return json.loads(path.read_text())


def _c8_case(sku):
    raw = _load(GLM_CASE)
    raw["machine"].update({"sku": sku, "num_devices": 1, "dies_per_package": 1,
                           "ranks_per_node": 1})
    raw["runtime"].update({"tp": 1, "dp": 1, "ep": 1, "pp": 1, "max_num_seqs": 256})
    raw["model"]["layers"] = 1
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"},
                                              "indexed_attention": dict(C8_PROGRAM)}}
    return raw


def _c8_op(sequences, topk, heads, prefix, paired_calls=1):
    shape = {"topk": topk, "compression_ratio": 1, "window": 0, "heads": heads,
             "latent": 512, "rope": 64, "cp": 1, "query_dtype": "bf16",
             "cache_dtype": "int8", "sequences": tuple(sequences),
             "prefix_cache_mode": "shared" if prefix else "private"}
    if paired_calls != 1:
        shape["paired_calls"] = paired_calls
    if prefix:
        shape["prefix_tokens"] = tuple([prefix] * len(sequences))
    rows = sum(query for query, _ in sequences)
    return OpSpec(
        op_id="s0.l0.attention.0", family="attention", phase="decode", stage=0,
        layer=0, requests=("r",), semantic_shape=(rows, heads, 576),
        local_shape=(rows, heads, 576), padded_shape=(rows, heads, 576),
        launch_shape=(rows, heads, 576), flops=0.0, vector_ops=0.0,
        hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0, deps=(),
        recipe="latent_attention_batch",
        metadata=(("program_kind", "indexed_attention"),
                  ("indexed_attention_geometry", shape)))


def _c8_prediction(sequences, query, context, topk, heads, prefix, paired_calls=1):
    out = {}
    for name, sku in SKUS:
        case = parse_case(_c8_case(sku))
        op = _c8_op([(query, context)] * sequences, topk, heads, prefix,
                    paired_calls)
        row = operator_costs(case, (op,), input_ready_ms=0)["operators"][0]
        out[name] = (row["duration_ms"] * 1000, row)
    return out


def _mqsmla_case(sku, mode, prefix, sequences):
    raw = _load(V41_CASE)
    raw["machine"].update({"sku": sku, "num_devices": 1, "dies_per_package": 1,
                           "ranks_per_node": 1})
    raw["runtime"].update({"tp": 1, "dp": 1, "ep": 1, "pp": 1,
                           "max_num_seqs": 256, "max_model_len": 262144})
    raw["model"].setdefault("metadata", {})["cache_representations"] = deepcopy(
        MQSMLA_MODES[mode])
    raw["runtime"]["kernels"].setdefault("lowerings", {})[
        "mqsmla_attention"] = {"implementation": MQSMLA_RECIPE}
    if prefix:
        raw["runtime"]["prefix_cache_mode"] = "shared"
        raw["workload"] = {"requests": [{
            "id": "r", "prompt_tokens": 131072, "output_tokens": 1,
            "prefix_tokens": prefix, "prefix_cache_key": "k"}]}
    return raw


def _mqsmla_prediction(sequences, mode, prefix, query=6, context=131072, layer=2):
    out = {}
    for name, sku in SKUS:
        raw = _mqsmla_case(sku, mode, prefix, sequences)
        ops = target_graphs(raw, scope="body", phase="decode", tokens=query,
                            context=context, sequences=sequences)[0]
        op = next(o for o in ops if o.recipe == MQSMLA_RECIPE and o.layer == layer)
        row = operator_costs(parse_case(raw), (op,), input_ready_ms=0)["operators"][0]
        out[name] = (row["duration_ms"] * 1000, row)
    return out


def _mqsmla_geometry(sequences, mode, prefix, window, ratio, topk, heads=64):
    shape = {"sequences": tuple(sequences), "window": window, "ratio": ratio,
             "topk": topk, "heads": heads, "width": 512,
             "metadata_support_fd": False, "implementation": MQSMLA_RECIPE,
             "addressing": "scalar", "kv_quant_mode": mode,
             "packed_row_bytes": KV_QUANT_ROW_BYTES[mode]}
    if prefix:
        shape.update({"prefix_cache_mode": "shared",
                      "prefix_tokens": tuple([prefix] * len(sequences))})
    else:
        shape["prefix_cache_mode"] = "private"
    return shape


def _mqsmla_direct(sequences, mode, prefix, window, ratio, topk):
    """Price one declared packed-KV call without a serving-cache admission.

    A BF16 latent cache keeps the plain attention path, so its tile program is
    exercised through the same source recipe rather than a packed binding.
    """
    out = {}
    for name, sku in SKUS:
        raw = _mqsmla_case(sku, mode, prefix, sequences)
        case = parse_case(raw)
        shape = _mqsmla_geometry(sequences, mode, prefix, window, ratio, topk)
        rows = sum(query for query, _ in sequences)
        op = OpSpec(
            op_id="s0.l2.mqsmla", family="attention", phase="decode", stage=0,
            layer=2, requests=("r",), semantic_shape=(rows, 64, 512),
            local_shape=(rows, 64, 512), padded_shape=(rows, 64, 512),
            launch_shape=(rows, 64, 512), flops=0.0, vector_ops=0.0,
            hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0, deps=(),
            recipe=MQSMLA_RECIPE, metadata=(("mqsmla_geometry", shape),))
        row = operator_costs(case, (op,), input_ready_ms=0)["operators"][0]
        out[name] = (row["duration_ms"] * 1000, row)
    return out


def _plan(row):
    return dict(row["operator"]["metadata"])["program_execution"]


@pytest.mark.parametrize("label,sequences,query,context,topk,heads,prefix,pr,dt,tol",
                         C8_TARGETS, ids=[row[0] for row in C8_TARGETS])
def test_c8_sfa_paired_silicon_targets(label, sequences, query, context, topk,
                                       heads, prefix, pr, dt, tol):
    result = _c8_prediction(sequences, query, context, topk, heads, prefix)
    assert result["PR"][0] == pytest.approx(pr, rel=tol), (label, result["PR"][0])
    assert result["DT"][0] == pytest.approx(dt, rel=tol), (label, result["DT"][0])


def test_c8_schedule_declares_tile128_alias_rows_and_conserves_native_work():
    case = parse_case(_c8_case(SKUS[0][1]))
    op = _c8_op([(6, 131072)] * 88, 2048, 64, 124518)
    row = operator_costs(case, (op,), input_ready_ms=0)["operators"][0]
    plan = _plan(row)
    assert plan["kernel_schedule"] == SCHEDULE_ARCH35
    assert plan["c8_schedule"]["key_tile"] == C8_KEY_TILE == 128
    assert plan["c8_schedule"]["tile_slots"] == C8_TILE_SLOTS == 2
    assert plan["c8_schedule"]["waves"] == 17
    assert plan["c8_schedule"]["head_tile"] == 64
    assert plan["c8_schedule"]["key_tile_owners"] == "one_1c2v_group_per_query_row"
    assert plan["packed_cache"]["row_bytes"] == C8_PACKED_ROW_BYTES == 656
    assert plan["packed_cache"]["key_value_alias"] is True
    # Declared Q/O, the one cold shared-prefix union, and the private rows each
    # query token gathers for itself (``S`` tokens never share a private row).
    qo_bytes = 528 * 64 * (576 + 512) * 2
    assert row["operator"]["hbm_bytes"] == (
        qo_bytes + 124518 * 656 + 528 * 102 * 656)
    assert row["operator"]["flops"] == 2 * 64 * 2048 * (576 + 512) * 528
    # Every declared FLOP is serviced by the Engine at the BF16 cube rate.
    peak = case.machine.dies[0].peak("bf16")
    assert row["serviced"]["matrix@r0:s0"] == pytest.approx(
        row["operator"]["flops"] / peak * 1000)


def test_c8_legacy_key512_three_slot_schedule_is_untouched():
    geometry = Geometry.from_devices(parse_case(_c8_case(SKUS[0][1])).machine.dies)
    op = _c8_op([(6, 4096)], 2048, 64, 0)
    legacy = build_packed(op, geometry, scratch_policy="offchip", compact=True,
                          operand_modes=True, schedule=SCHEDULE_ARCH22)
    assert legacy.metadata["kernel_schedule"] == SCHEDULE_ARCH22
    assert legacy.metadata["packed_cache"]["row_bytes"] == 656
    assert legacy.metadata["vector_buffer_bytes_per_aiv"] > 0
    arch35 = build_packed(op, geometry, scratch_policy="offchip", compact=True,
                          operand_modes=True, schedule=SCHEDULE_ARCH35,
                          tile_size=128, slots=2)
    assert arch35.metadata["kernel_schedule"] == SCHEDULE_ARCH35
    # The arch35 planner bills both aliased reads of every packed row.
    assert arch35.work.resources["mte2_aiv"] > legacy.work.resources["mte2_aiv"]
    assert arch35.work.matrix["bf16"] == pytest.approx(legacy.work.matrix["bf16"])


def test_c8_private_mode_exposes_every_selected_row():
    sequences = ((6, 16384),) * 96
    shared = {"sequences": sequences, "topk": 256, "prefix_cache_mode": "shared",
              "prefix_tokens": tuple([15564] * 96)}
    private = {"sequences": sequences, "topk": 256, "prefix_cache_mode": "private"}
    shared_unique, private_rows, shared_selected = prefix_partition(
        shared, sequences, 256)
    # One physical shared union, plus each request's own few selected rows.
    assert shared_unique == 15564
    assert shared_selected == 243
    assert private_rows == 96 * 13
    # Without a shared pool nothing is cold-filled: every row is its own read.
    assert prefix_partition(private, sequences, 256) == (0, 96 * 6 * 256, 0)


def test_c8_owner_rows_carry_one_group_and_their_selected_tiles():
    sequences = ((6, 131072),) * 88
    parts = tuple(owner_row_tiles(0, sequences, 32, 2048, 128))
    assert len(parts) == 17
    assert all(len(tiles) == 16 for _sequence, tiles in parts)
    assert all(tiles[0][1] and tiles[-1][2] for _sequence, tiles in parts)
    assert parts[0][1][0] == (128, True, False)


@pytest.mark.parametrize("label,sequences,mode,prefix,layer,pr,dt", MQSMLA_TARGETS,
                         ids=[row[0] for row in MQSMLA_TARGETS])
def test_mqsmla_paired_silicon_targets(label, sequences, mode, prefix, layer, pr, dt):
    result = (_mqsmla_prediction(sequences, mode, prefix, layer=layer)
              if layer is not None else
              _mqsmla_direct([(6, 131072)] * sequences, mode, prefix, 128, 0, 0))
    assert result["PR"][0] == pytest.approx(pr, rel=0.10), (label, result["PR"][0])
    assert result["DT"][0] == pytest.approx(dt, rel=0.10), (label, result["DT"][0])


def test_mqsmla_quant_mode_is_admitted_and_memoized_with_the_tile_dag():
    observed = {}
    for mode in (KV_QUANT_FP8, KV_QUANT_MXFP8):
        result = _mqsmla_prediction(160, mode, 124518)
        plan = _plan(result["PR"][1])
        assert plan["mqsmla_kv_quant_mode"] == mode
        assert plan["mqsmla_packed_row_bytes"] == KV_QUANT_ROW_BYTES[mode]
        assert plan["mqsmla_schedule"].startswith("arch35_overlapped_tile_dag")
        assert plan["mqsmla_metadata_program_ms"] == METADATA_PROGRAM_MS
        observed[mode] = (plan["mqsmla_packed_row_bytes"],
                          plan["mqsmla_dequant_ops_per_element"],
                          result["PR"][0], result["DT"][0])
    assert observed[KV_QUANT_FP8][0] == 608
    assert observed[KV_QUANT_MXFP8][0] == 584
    # The UE8M0 group unpack costs more vector work than the FP8 tile scale.
    assert observed[KV_QUANT_MXFP8][1] > observed[KV_QUANT_FP8][1]
    assert observed[KV_QUANT_MXFP8][2] > observed[KV_QUANT_FP8][2]
    assert observed[KV_QUANT_MXFP8][3] > observed[KV_QUANT_FP8][3]


def test_mqsmla_swa_window_is_private_and_shared_topk_union_is_cold_fill():
    swa = {"sequences": ((6, 131072),) * 160, "window": 128, "ratio": 0, "topk": 0}
    external, shared_rows = quant_external_fill(swa, KV_QUANT_FP8)
    assert shared_rows == 0
    assert external == (160 * 128 + 160 * 6) * 608
    csa = {"sequences": ((6, 131072),) * 160, "window": 128, "ratio": 2,
           "topk": 512, "prefix_cache_mode": "shared",
           "prefix_tokens": tuple([124518] * 160)}
    external, shared_rows = quant_external_fill(csa, KV_QUANT_FP8)
    assert shared_rows == 124518 // 2
    assert external == (160 * 128 + 160 * 6 + 124518 // 2 + 160 * 26) * 608
    assert quant_external_fill(dict(csa, prefix_cache_mode="private"),
                               KV_QUANT_FP8)[0] > external


def test_mqsmla_mixed_fp8_fp4_declaration_stays_on_the_legacy_row_widths():
    raw = _load(V41_CASE)
    raw["model"].setdefault("metadata", {})["cache_representations"] = deepcopy(FORMATS)
    model = parse_case(raw).model
    assert quant_mode_of(model, 576) is None
    ops = target_graphs(raw, scope="body", phase="decode", tokens=6,
                        context=2048, sequences=2)[0]
    att = next(o for o in ops if o.layer == 2 and o.family == "attention")
    assert declared_quant_mode(att) is None


def test_paired_dcp_sfa_declares_two_sequential_half_head_calls():
    assert PAIRED_SFA_CALLS == 2
    assert paired_sfa_calls(96) == (48, 48)
    with pytest.raises(ModelingError, match="paired_sfa_requires_even_gathered_heads"):
        paired_sfa_calls(45)
    # One half-head call is the calibrated unit; the pair is two of them plus
    # the declared peer exchange, so it must cost strictly more than one call.
    single = _c8_prediction(96, 6, 16384, 256, 48, 15564)
    assert single["PR"][0] == pytest.approx(155.59, rel=0.12)
    assert single["DT"][0] == pytest.approx(136.15, rel=0.12)
    paired = _c8_prediction(96, 6, 16384, 256, 48, 15564, paired_calls=2)
    plan = _plan(paired["PR"][1])
    assert plan["c8_schedule"]["paired_calls"] == 2
    assert plan["c8_schedule"]["external_fill_bytes"] == _plan(
        single["PR"][1])["c8_schedule"]["external_fill_bytes"]
    assert paired["PR"][0] > single["PR"][0]
    assert paired["PR"][0] == pytest.approx(2 * 155.59, rel=0.20)


def test_c8_unknown_schedule_is_rejected_instead_of_silently_reinterpreted():
    geometry = Geometry.from_devices(parse_case(_c8_case(SKUS[0][1])).machine.dies)
    op = _c8_op([(6, 4096)], 2048, 64, 0)
    with pytest.raises(ModelingError, match="schedule_unknown"):
        build_packed(op, geometry, scratch_policy="offchip", schedule="arch99")
    with pytest.raises(ModelingError, match="tile128_and_two_slots"):
        build_packed(op, geometry, scratch_policy="offchip",
                     schedule=SCHEDULE_ARCH35, tile_size=512, slots=3)
