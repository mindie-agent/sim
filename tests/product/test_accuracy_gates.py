"""Catalog stack parity, heldout accuracy, and composite physical scaling.

[Role] Compare declared, exported, and executed operator selections and
  validate body predictions against retained measurements and physical references.
[Invariants] Heldout accuracy stays within 10%. Analytical byte, FLOP, wire,
  and duration checks conserve work across parallelism and speculation.
[Timeouts & Locks] CPU-only catalog sweeps are sharded by hardware profile.
[Side-Effect Policy] Reads checked-in cases and scoreboards. Test homes are isolated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mindie_sim import predict
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.weights import cache_row_bytes_by_layer, weight_breakdown
from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import analyze_network_efficiency, extract_hardware_peaks, solve_sla_sizing
from mindie_sim.prediction.scheduler import DraftPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.model_catalog import list_models, load_model
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.operators.paged_mla.family_mqsmla import FORMATS as MQSMLA_FORMATS
from mindie_sim.prediction.operators.paged_mla.family_mqsmla import RECIPE as MQSMLA_RECIPE
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.structure import export_structure

pytestmark = pytest.mark.tier0

SKUS = ("a2", "a3", "a5")
OLDER_HARDWARE = SKUS[:2]
MQSMLA_NATIVE = "aclnnMixedQuantSparseFlashMla"
SHARED_KV_NATIVE = "SparseAttnSharedkv"
SPARSE_MLA_NATIVE = "SparseFlashMla"
MIXED_QUANT_NATIVE = "MixedQuantSparseFlashMla"
V41_CACHE_KEYS = ("window", "compressed_kv", "index_k")
PACKED_ROW_BYTES = {"window": 544, "compressed_kv": 320, "index_k": 68}
REGION_WIDTH = {"window": 512, "compressed_kv": 512, "index_k": 128}
DEFAULT_CONTEXT = 2048
DEFAULT_MAX_MODEL_LEN = 32768
CATALOG_MODELS = tuple(entry["key"] for entry in list_models())

def _packed_case(**overrides):
    options = dict(tp=2, dp=4, ep=8, concurrency=4, max_model_len=DEFAULT_MAX_MODEL_LEN)
    return compose_case("deepseek-v4.1-flash", "a5", **(options | overrides))

def attention_natives(raw_case, *, phase="decode", tokens=1,
                      context=DEFAULT_CONTEXT, sequences=1, scope="body"):
    """Native attention kinds from the real structure export schema."""
    report = export_structure(parse_case(raw_case), phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope)
    return {native for node in report["nodes"] if node["family"] == "attention"
        for native in node["implementation"].get("native_kinds", ())}

def attention_recipes(raw_case, *, context=DEFAULT_CONTEXT, sequences=1):
    graphs = target_graphs(parse_case(raw_case), phase="decode", tokens=1, context=context,
        sequences=sequences, scope="body")
    return {op.recipe for op in graphs[0] if op.family == "attention"}

def selection_summary(raw_case):
    return dict(parse_case(raw_case).runtime.metadata)["operator_selection_summary"]

@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_packed_cache_selects_mixed_quant_mla_across_tensor_parallel(tp: int):
    dp = max(1, 8 // tp)
    raw = _packed_case(tp=tp, dp=dp, ep=8, concurrency=dp)
    declared = raw["model"]["metadata"]["cache_representations"]
    assert tuple(declared) == V41_CACHE_KEYS
    index = {"dtype": "uint8", "packed_elements": 2, "group_size": 32, "scale_dtype": "uint8"}
    assert declared == {**MQSMLA_FORMATS, "index_k": index}
    parsed = parse_case(raw)
    actual = {region: cache_format(parsed.model, region).row_bytes(width)
              for region, width in REGION_WIDTH.items()}
    assert actual == PACKED_ROW_BYTES
    assert actual["index_k"] < REGION_WIDTH["index_k"]
    assert attention_recipes(raw, sequences=dp) == {MQSMLA_RECIPE}
    assert attention_natives(raw, sequences=dp) == {MQSMLA_NATIVE}

def test_mixed_quant_mla_preserves_idle_replicas_and_verify_rows():
    raw = _packed_case(tp=2, dp=4, ep=8, concurrency=3, mtp_proposal_tokens=2, mtp_acceptance_length=2.0)
    parsed = parse_case(raw)
    graphs = target_graphs(parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
        sequences=(3, 1, 0, 0), scope="body", padded_idle=True)
    assert {op.recipe for op in graphs[0] if op.family == "attention"} == {MQSMLA_RECIPE}
    verify = target_graphs(parsed, phase="decode", tokens=1 + 2, context=DEFAULT_CONTEXT,
        sequences=(1, 1, 1, 0), scope="body")
    assert {op.recipe for op in verify[0] if op.family == "attention"} == {MQSMLA_RECIPE}
    result = predict(raw, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
                     sequences=(1, 1, 1, 0), scope="body")
    assert result["duration_ms"] > 0

def test_expert_quantization_preserves_packed_cache_binding():
    fp8 = _packed_case(quant="fp8")
    default = _packed_case()
    assert (fp8["model"]["metadata"]["cache_representations"]
            == default["model"]["metadata"]["cache_representations"])
    assert fp8["model"]["quant"]["expert_weight_format"] != "mxfp4"
    assert attention_natives(fp8, sequences=4) == {MQSMLA_NATIVE}
    assert attention_recipes(fp8, sequences=4) == {MQSMLA_RECIPE}

def test_attention_head_tiling_rejection_has_explicit_reason():
    raw = _packed_case(tp=16, dp=1, ep=1, concurrency=1, sp=False)
    summary = selection_summary(raw)
    reasons = {note["reason"] for note in summary["degraded_fallbacks"]}
    assert "mqsmla_local_heads_outside_source_tp_tiling" in reasons
    assert all(note["family"] != "mqsmla_decode" for note in summary["selected_native"])
    assert any("mqsmla_decode->analytical" in warning for warning in summary["selection_warnings"])
    assert attention_natives(raw) == {MIXED_QUANT_NATIVE}

def test_explicit_bf16_cache_override_preserves_declared_dtype():
    from mindie_sim.compose.operator_defaults import (CACHE_OVERRIDE_REASON, apply_compose_v41_a5_defaults,)
    model = load_model("deepseek-v4.1-flash")
    model_dict = {"metadata": {"cache_representations": {"window": {"dtype": "bf16"},
        "compressed_kv": {"dtype": "bf16"},}}}
    lowerings: dict = {}
    notes = apply_compose_v41_a5_defaults(
        model_dict=model_dict, lowerings=lowerings, backend="ascend", arch="a5",
        layer_plans=model.layer_plans, activation_dtype=model.activation_dtype, tp=2, cp=1,
        runtime_metadata={})
    assert CACHE_OVERRIDE_REASON in {note.get("reason") for note in notes}
    assert "mqsmla_attention" not in lowerings
    assert model_dict["metadata"]["cache_representations"]["window"] == {"dtype": "bf16"}
    raw = _packed_case()
    raw["model"]["metadata"]["cache_representations"] = {"window": {"dtype": "bf16"},
        "compressed_kv": {"dtype": "bf16"},}
    parsed = parse_case(raw)
    _effective, selection = effective_body_declaration(parsed, "decode", tokens=1)
    reasons = {note["reason"] for note in selection["operator_selection_summary"]["degraded_fallbacks"]}
    assert any("cache_representation_contract_required" in reason for reason in reasons)
    assert attention_natives(parsed) == {SPARSE_MLA_NATIVE}

@pytest.mark.parametrize("hardware", OLDER_HARDWARE)
def test_unpacked_cache_selects_sparse_mla(hardware: str):
    raw = compose_case("deepseek-v4.1-flash", hardware, tp=4, dp=2, ep=8, concurrency=2,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert attention_natives(raw, sequences=2) == {SPARSE_MLA_NATIVE}

def test_shared_kv_respects_hardware_domain():
    a3 = compose_case("deepseek-v4-flash-0731", "a3", tp=8, dp=1, ep=8, concurrency=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert attention_natives(a3) == {SHARED_KV_NATIVE}
    a5 = compose_case("deepseek-v4-flash-0731", "a5", tp=8, dp=1, ep=8, concurrency=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    a5_natives = attention_natives(a5)
    assert a5_natives and a5_natives <= {SPARSE_MLA_NATIVE, MIXED_QUANT_NATIVE}

def test_quantized_cache_selects_sparse_mla_by_dtype():
    raw = compose_case("deepseek-v4-flash-0731", "a5", quant="w8a8", tp=8, dp=1, ep=8,
        concurrency=1, max_model_len=DEFAULT_MAX_MODEL_LEN)
    parsed = parse_case(raw)
    cache_dtype = parsed.model.quant.kv_dtype
    natives = attention_natives(parsed)
    expected = (MIXED_QUANT_NATIVE if cache_dtype not in {"bf16", "fp16"} else SPARSE_MLA_NATIVE)
    assert natives == {expected}

def test_paged_mla_source_kernels_cover_both_generations():
    kimi_a5 = compose_case("kimi-k3", "a5", tp=8, dp=1, ep=8, concurrency=4, max_model_len=65536)
    assert attention_recipes(kimi_a5, context=32768, sequences=4) == {"ascend_paged_mla_decode_arch35"}
    assert attention_recipes(kimi_a5, context=126000, sequences=4) == {"ascend_paged_mla_decode_arch35"}
    dsr1_a2 = compose_case("deepseek-r1", "a2", tp=8, dp=1, ep=8, concurrency=4, max_model_len=8192)
    assert attention_recipes(dsr1_a2, context=1024, sequences=4) == {"ascend_paged_mla_decode_arch22"}
    assert "ascend_paged_mla_decode_arch22" not in attention_recipes(dsr1_a2, context=256, sequences=4)

def test_windowed_gqa_layers_keep_analytical_execution_with_a_reason():
    for key in ("gemma-4-26b-a4b", "gemma-4-31b"):
        raw = compose_case(key, "a5", tp=8, dp=1, ep=1, concurrency=4, max_model_len=DEFAULT_MAX_MODEL_LEN)
        assert "flash_attention_decode" not in attention_recipes(
            raw, context=DEFAULT_CONTEXT, sequences=4), key
        _effective, selection = effective_body_declaration(parse_case(raw), "decode", tokens=1)
        reasons = {note["reason"] for note in selection["operator_selection_summary"]["degraded_fallbacks"]}
        assert "windowed_noncausal_gqa_domain" in reasons, key
    dense = compose_case("qwen3-32b", "a5", tp=8, dp=1, ep=1, concurrency=4,
        max_model_len=DEFAULT_MAX_MODEL_LEN)
    assert attention_recipes(dense, sequences=4) == {"flash_attention_decode"}
    lowerings = parse_case(dense).runtime.kernels.get("lowerings") or {}
    assert lowerings.get("add_rmsnorm") == "vllm_ascend_native_norm"

def test_low_rank_hyperconnections_export_both_roles():
    raw = compose_case("qwen3.8-flash-next", "a5", tp=4, dp=1, ep=4, concurrency=2, max_model_len=8192)
    report = export_structure(parse_case(raw), phase="decode", tokens=1, context=1024)
    roles = {group["role"] for group in report["subgraphs"]}
    assert {"hc_pre", "hc_post"} <= roles

@pytest.mark.parametrize("hardware", SKUS)
def test_all_catalog_models_compose_and_lower_on_every_sku(hardware: str):
    """Every catalog entry composes and lowers; drafts attach to a target."""
    from mindie_sim.catalog.draft import optional_draft_spec
    checked = 0
    drafts = 0
    for key in CATALOG_MODELS:
        raw = None
        for experts in (8, 1):
            try:
                raw = compose_case(key, hardware, tp=8, dp=1, ep=experts, concurrency=1,
                    max_model_len=DEFAULT_MAX_MODEL_LEN)
                break
            except Exception as exc:  # dense models reject ep > 1
                if experts == 1:
                    raise AssertionError(f"{key}/{hardware}: {exc}") from exc
        parsed = parse_case(raw)
        if parsed.model.is_standalone_draft:
            auxiliary = parsed.model.auxiliary
            assert auxiliary.dspark is not None or auxiliary.mtp is not None
            kind = "dspark" if auxiliary.dspark is not None else "mtp"
            assert optional_draft_spec(parsed.model, kind) is not None
            drafts += 1
            continue
        graphs = target_graphs(parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT,
            sequences=1, scope="body")
        assert graphs, (key, hardware)
        report = export_structure(parsed, phase="decode", tokens=1, context=DEFAULT_CONTEXT, scope="body")
        assert report["nodes"], (key, hardware)
        checked += 1
    assert checked + drafts == len(CATALOG_MODELS)
    assert checked > 0

def test_entry_points_agree_on_the_selected_operator_stack():
    raw = _packed_case()
    parsed = parse_case(raw)
    geometry = dict(phase="decode", tokens=1, context=DEFAULT_CONTEXT, sequences=4)
    declared = dict(parsed.runtime.metadata)["operator_selection_summary"]
    assert {note["binding"] for note in declared["selected_native"]} >= {MQSMLA_RECIPE}
    effective, selection = effective_body_declaration(parsed, "decode", tokens=1)
    body_selected = {note["binding"] for note in selection["operator_selection_summary"]["selected_native"]}
    assert MQSMLA_RECIPE in body_selected
    bodies = target_graphs(parsed, scope="body", **geometry)
    models = target_graphs(parsed, scope="model", **geometry)
    body_recipes = {op.recipe for op in bodies[0] if op.family == "attention"}
    model_recipes = {op.recipe for op in models[0] if op.family == "attention"}
    assert body_recipes == model_recipes == {MQSMLA_RECIPE}
    assert attention_natives(parsed, **geometry) == {MQSMLA_NATIVE}
    resolved = {record["family"]: record["selection"] for record in models.default_selection["selections"]}
    assert resolved["mqsmla_decode"] == "default_installed"
    assert models.scope == "model" and bodies.scope == "body"
    assert models[0].scope == "model" and bodies[0].scope == "body"

REL_TOL = 0.10
ROOT = Path(__file__).resolve().parents[2]
VERIFICATION = ROOT / "data/verification/decode-body-current-20261030.json"
CASE_ROOT = ROOT / "examples/cases/high-concurrency-20261030"

class _HeldoutTimingMismatch(AssertionError):
    """Raised strictly when a heldout step prediction misses the 10% p50 gate."""


A3_SPECIALIST_PENDING = pytest.mark.xfail(
    strict=True,
    raises=_HeldoutTimingMismatch,
    reason="Pending Tier-1/Tier-2 A3-03 GroupedMatmul + A3-05 MC2 specialist closure "
           "after Vol.8 I-01/I-02 (mindie_sim/backends/profiles/ascend.json:8613) "
           "removed 1.2T/32B-cyc error cancellation",
)
HELDOUT_CASES = (
    pytest.param("v41-new6/a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32.json",
                 "a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32:q8_b9_m72", marks=A3_SPECIALIST_PENDING),
    pytest.param("glm-source-accepted-26/glm53flash-a3-tp16-c4.json",
                 "glm53flash-a3-tp16-c4:q6_b4_m24", marks=A3_SPECIALIST_PENDING),
    pytest.param("glm-source-accepted-26/a3_glm53flash_tp16_aclgraph_c16.json",
                 "a3_glm53flash_tp16_aclgraph_c16:q6_b16_m96"),
)
HELDOUT_IDS = [row.values[1].split(":")[0] for row in HELDOUT_CASES]

from _support.axes import COMPOSITE_ACCURACY_CAPTURES as CAPTURES  # noqa: E402

PIPELINE = (("kimi-k3", 2), ("glm-5.3", 4), ("deepseek-v4-pro", 4), ("qwen3.5-397b-a17b", 2))

MC2_PLAIN_REF_MS = 0.36357595397457626
MC2_SCHEDULED_REF_MS = 0.5379446716058904

def _rel(actual: float, measured: float) -> float:
    return abs(actual - measured) / measured

def _case(model, hardware, *, quant=None, tp=1, dp=1, ep=1, pp=1, pcp=1, sp=False,
          rpn=None, layers=2, experts=8, topk=2, mtp=0, accept=1.0, shared=False, recipe=None):
    kw = dict(model=model, hardware=hardware, tp=tp, dp=dp, ep=ep, pp=pp, pcp=pcp, sp=sp,
        layers=layers, n_routed_experts=experts, num_experts_per_tok=topk, prompt_tokens=64,
        output_tokens=8, concurrency=1, max_model_len=131072, enable_shared_expert_dp=shared,
        multistream_overlap_shared_expert=False, enable_fused_mc2=0, mtp_proposal_tokens=mtp)
    if quant is not None: kw["quant"] = quant
    if rpn is not None: kw["ranks_per_node"] = rpn
    if mtp: kw["mtp_acceptance_length"] = accept
    case = compose_case(**kw)
    if recipe:
        case["runtime"]["moe_recipe"] = recipe
        case["runtime"]["mc2_chunks"] = 4
    return case

def _row_case(row, **overrides):
    _, model, hardware, quant, _, tp, dp, ep, rpn, layers, experts, phase, *_ = row
    options = dict(quant=quant, tp=tp, dp=dp, ep=ep, layers=layers, n_routed_experts=experts,
                   num_experts_per_tok=2, prompt_tokens=64, output_tokens=8, concurrency=1,
                   max_model_len=131072, multistream_overlap_shared_expert=False,
                   engram_cpu_offload=False, enable_shared_expert_dp=False, mtp_proposal_tokens=0)
    if phase == "decode": options["exclude_optimizations"] = ("cann_megamoe_fusion",)
    if rpn is not None: options["ranks_per_node"] = rpn
    return compose_case(model, hardware, **(options | overrides))

def _step(case, phase, tokens, context, sequences):
    return predict(case, scope="body", phase=phase, tokens=tokens, context=context, sequences=sequences)

def _conserve(eff, body_ms, *, collective_extra_ms=0.0):
    ops = eff["operators"]
    summed = sum(op["duration_ms"] * op["occurrences"] for op in ops)
    accounted = summed + collective_extra_ms
    assert summed <= accounted <= body_ms + 1e-12
    assert max(0.0, body_ms - accounted) <= max(1e-6, 0.002 * body_ms)
    assert eff["operator_sum_ms"] == pytest.approx(summed, rel=1e-12)
    assert eff["graph_duration_ms"] == pytest.approx(body_ms, rel=1e-12)
    assert sum(op["link_bytes"] * op["occurrences"] for op in ops) == pytest.approx(eff["total_link_bytes"], abs=1e-6)
    assert sum(op["hbm_bytes"] * op["occurrences"] for op in ops) == pytest.approx(eff["total_hbm_bytes"], rel=1e-9)

def _bucket(ops, kind):
    if kind == "shared_expert":
        return [op for op in ops if "shared" in op["op_id"]]
    if kind in {"expert_dispatch", "expert_combine"}:
        return [op for op in ops if op["recipe"] == kind]
    if kind == "tp_allreduce":
        return [op for op in ops if op["family"] == "allreduce"]
    assert kind == "gdn"
    return [op for op in ops if op["family"] in {"short_convolution", "gdn_gating", "recurrent_vector"}]

def _scoreboard():
    return json.loads(VERIFICATION.read_text(encoding="utf-8"))

def _scoreboard_row(case_id):
    row = next((candidate for candidate in _scoreboard()["rows"] if candidate["case_id"] == case_id), None)
    assert row is not None, case_id
    return row

def _metadata_key(value, key):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                return child
            found = _metadata_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _metadata_key(child, key)
            if found is not None:
                return found
    return None

def _evaluation_relation(case, row):
    declared = _metadata_key((case.get("runtime") or {}).get("metadata") or {}, "evaluation_relation")
    if declared is not None:
        return str(declared)
    origin = row.get("calibration_origin")
    if origin is None:
        return "unknown"
    return "in_sample" if "in_sample" in str(origin) else "heldout_capture_not_in_sample"

def _declared_capture_path(case):
    metadata = (case.get("runtime") or {}).get("metadata") or {}
    capture = metadata.get("capture_directory")
    if capture:
        return str(capture)
    return str((metadata.get("kernel_source") or {}).get("path") or "")

def _explicit_capture_provenance(case, path):
    """Name the external campaign that owns one retained capture path."""
    for marker in ("heldout_engine_20261030", "glm5_3_flash_w8a8_heldout"):
        if marker in path:
            return marker
    capture = _declared_capture_path(case)
    return capture if capture and capture in path else ""

def _apply_captured_parallel(case, geometry):
    """Declare sequence parallelism observed by the retained capture geometry."""
    factor = int(geometry.get("sequence_parallel_factor", 1) or 1)
    if factor <= 1:
        return case
    kernels = case["runtime"].setdefault("kernels", {})
    if "sequence_parallel" not in kernels:
        from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION
        kernels["sequence_parallel"] = dict(DECLARATION, padding="right_to_tp")
        case["runtime"].setdefault("metadata", {})["sp"] = True
    return case

@pytest.mark.parametrize("relative,case_id", HELDOUT_CASES, ids=HELDOUT_IDS)
def test_heldout_real_npu_body_p50(relative, case_id):
    case = json.loads((CASE_ROOT / relative).read_text(encoding="utf-8"))
    row = _scoreboard_row(case_id)
    assert row["display_status"] == "compared"
    path = row["source_identity"]["path"]
    campaign = _explicit_capture_provenance(case, path)
    assert "kernel_details.csv" in path
    assert campaign and campaign in path and "in_sample" not in campaign.lower()
    assert row["comparison"]["metric_boundary"] == "target_body_device_graph"
    assert _evaluation_relation(case, row) == "unknown"
    geometry = row["geometry"]
    measured = row["comparison"]["observed_ms"]["p50"]
    case = _apply_captured_parallel(case, geometry)
    live = predict(case, scope="body", phase=geometry["phase"], tokens=geometry["tokens_per_sequence"],
                   context=geometry["context_tokens"], sequences=geometry["sequences_per_replica"])
    dur = live["duration_ms"]
    assert live["scope"] == "body" and math.isfinite(dur) and dur > 0 and math.isfinite(measured) and measured > 0
    if _rel(dur, measured) > REL_TOL:
        raise _HeldoutTimingMismatch(f"{relative}: live {dur:.6f} ms vs measured {measured:.6f} ms")

def test_heldout_cases_preserve_unique_ids_and_unmarked_glm_c16():
    assert HELDOUT_IDS == ["a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32", "glm53flash-a3-tp16-c4", "a3_glm53flash_tp16_aclgraph_c16"]
    assert len(HELDOUT_CASES) == 3 and HELDOUT_CASES[2].marks == ()
def test_missing_evaluation_provenance_is_unknown():
    assert _evaluation_relation({}, {}) == _evaluation_relation({}, {"calibration_origin": None}) == "unknown"
    assert _evaluation_relation({}, {"calibration_origin": "observed"}) == "heldout_capture_not_in_sample"
    assert _evaluation_relation({}, {"calibration_origin": "retained_operator_service_in_sample"}) == "in_sample"

@pytest.mark.parametrize("marker", ("heldout_engine_20261030", "glm5_3_flash_w8a8_heldout"))
def test_named_heldout_campaign_marker_is_explicit(marker):
    path = f"/captures/{marker}/run/kernel_details.csv"
    assert _explicit_capture_provenance({}, path) == marker

def test_declared_evaluation_relation_is_preserved():
    path = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp8.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    expected = "retrospective_cross_TP_frozen_training_validation"
    assert case["runtime"]["metadata"]["calibration"]["evaluation_relation"] == expected
    assert _evaluation_relation(case, {}) == expected

@pytest.mark.parametrize("row", CAPTURES, ids=[row[0] for row in CAPTURES])
def test_composite_matrix_matches_physical_roofline(row):
    name, model, hardware, quant, sku, tp, dp, ep, rpn, layers, experts, phase, tokens, ctx, seq, reference = row
    case = _row_case(row)
    machine = case["machine"]
    assert machine["sku"] == sku
    assert machine["num_devices"] == tp * dp
    multi_node = machine["num_devices"] > machine["ranks_per_node"]
    assert multi_node is (name.endswith("-mn") or name.endswith("-mn-fp8"))
    got = _step(case, phase, tokens, ctx, seq)["duration_ms"]
    eff = analyze_network_efficiency(
        case, phase=phase, tokens=tokens, context=ctx, sequences=seq, scope="body")
    hbm_bw_bytes_per_ms = (extract_hardware_peaks(case)["hbm_peak_bytes_per_s"] / 1000.0)
    assert eff["total_roofline_ideal_ms"] <= got <= eff["operator_sum_ms"] * 1.05
    assert got >= eff["total_hbm_bytes"] / hbm_bw_bytes_per_ms
    assert got == pytest.approx(reference, abs=5e-9)
    _conserve(eff, got)

def test_multi_node_ep16_exceeds_single_node_hccl_latency():
    rows = {row[0]: row for row in CAPTURES}
    named_sn = _row_case(rows["a3-9362-ep16-sn"])
    named_mn = _row_case(rows["a2-ep16-mn"])
    assert _step(named_mn, "decode", 1, 64, 1)["duration_ms"] > _step(
        named_sn, "decode", 1, 64, 1)["duration_ms"]
    single = _row_case(rows["a3-9362-ep16-sn"])
    multi = _row_case(rows["a3-9362-ep16-sn"], ranks_per_node=8)
    assert single["machine"]["sku"] == multi["machine"]["sku"] == "Ascend910_9362"
    assert single["machine"]["num_devices"] == multi["machine"]["num_devices"] == 16
    assert (single["machine"]["ranks_per_node"], multi["machine"]["ranks_per_node"]) == (16, 8)
    single_ms = _step(single, "decode", 1, 64, 1)["duration_ms"]
    multi_ms = _step(multi, "decode", 1, 64, 1)["duration_ms"]
    single_eff = analyze_network_efficiency(
        single, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    multi_eff = analyze_network_efficiency(
        multi, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    assert multi_ms > single_ms
    assert multi_eff["total_link_bytes"] == pytest.approx(single_eff["total_link_bytes"])
    assert multi_eff["total_link_ideal_ms"] > single_eff["total_link_ideal_ms"]

def test_shared_expert_dp_mc2_conserves_weights_and_hccl_bytes():
    common = dict(model="qwen3.5-35b-a3b", hardware="Ascend910_9362", quant="w8a8", tp=4, dp=1, ep=4, layers=2)
    plain = _case(**common)
    scheduled = _case(**common, recipe="mc2")
    shared = _case(**common, shared=True, recipe="mc2")
    assert shared["runtime"]["metadata"]["enable_shared_expert_dp"] is True
    assert scheduled["runtime"]["moe_recipe"] == "mc2"
    plain_eff = analyze_network_efficiency(plain, scope="body", phase="decode", tokens=8, context=128, sequences=1)
    mc2_eff = analyze_network_efficiency(scheduled, scope="body", phase="decode", tokens=8, context=128, sequences=1)
    shared_step = _step(shared, "decode", 8, 128, 1)
    assert _rel(plain_eff["graph_duration_ms"], MC2_PLAIN_REF_MS) <= REL_TOL
    assert _rel(mc2_eff["graph_duration_ms"], MC2_SCHEDULED_REF_MS) <= REL_TOL
    assert _rel(plain_eff["graph_duration_ms"], MC2_SCHEDULED_REF_MS) > REL_TOL
    assert plain_eff["total_link_bytes"] == pytest.approx(mc2_eff["total_link_bytes"])
    assert plain_eff["total_link_bytes"] == pytest.approx(233496.0)
    plain_step = _step(plain, "decode", 8, 128, 1)
    tp_graph_ms = sum(row["elapsed_ms"] for row in plain_step["timeline"] if row["family"] == "allreduce")
    tp_operator_ms = sum(row["total_ms"] for row in plain_eff["operators"] if row["family"] == "allreduce")
    _conserve(plain_eff, plain_step["duration_ms"], collective_extra_ms=tp_graph_ms - tp_operator_ms)
    mc2_ops = mc2_eff["operators"]
    mc2_summed = sum(op["duration_ms"] * op["occurrences"] for op in mc2_ops)
    assert mc2_summed == pytest.approx(mc2_eff["operator_sum_ms"], rel=1e-9)
    assert mc2_eff["operator_sum_ms"] > mc2_eff["graph_duration_ms"]
    assert (sum(op["link_bytes"] * op["occurrences"] for op in mc2_ops)
            == pytest.approx(mc2_eff["total_link_bytes"], abs=1e-6))
    assert (sum(op["hbm_bytes"] * op["occurrences"] for op in mc2_ops)
            == pytest.approx(mc2_eff["total_hbm_bytes"], rel=1e-9))
    assert any(str(row.get("stream", "")).startswith("mc2") for row in shared_step["timeline"])
    for kind in ("shared_expert", "expert_dispatch", "expert_combine", "tp_allreduce", "gdn"):
        assert _bucket(mc2_eff["operators"], kind), kind
    dispatch = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "expert_dispatch"))
    combine = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "expert_combine"))
    reduce = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "tp_allreduce"))
    assert dispatch > 0 and combine > 0 and reduce > 0
    assert dispatch + combine + reduce == pytest.approx(mc2_eff["total_link_bytes"])
    off = weight_breakdown(parse_case(plain).model, parse_case(plain).runtime.parallel, enable_shared_expert_dp=False)
    on = weight_breakdown(parse_case(shared).model, parse_case(shared).runtime.parallel, enable_shared_expert_dp=True)
    assert off["routed_expert"] == pytest.approx(12607488)
    assert on["routed_expert"] == pytest.approx(off["routed_expert"])
    assert off["shared_expert"] == pytest.approx(1582080)
    assert on["shared_expert"] == pytest.approx(6303744)
    assert shared_step["duration_ms"] != mc2_eff["graph_duration_ms"]

def _sla_baseline(case, *, context, output_budget):
    result = solve_sla_sizing(case, phase="decode", context=context, output_budget=output_budget,
        max_sequences_per_dp=1, scenarios=[{"id": "baseline"}], scope="body")
    return result, result["scenarios"][0]

def _assert_automatic_speculative_scaling(model, hardware, *, quant, tp, dp, ep,
                                          proposal, acceptance, context=256, output=16):
    common = dict(model=model, hardware=hardware, tp=tp, dp=dp, ep=ep, layers=2,
                  prompt_tokens=context, output_tokens=output, concurrency=1)
    if quant is not None:
        common["quant"] = quant
    off_case = compose_case(**common, mtp_proposal_tokens=0, mtp_acceptance_length=1.0)
    on_case = compose_case(**common, mtp_proposal_tokens=proposal, mtp_acceptance_length=acceptance)
    parsed_off, parsed_on = parse_case(off_case), parse_case(on_case)
    off_policy = DraftPolicy.for_request(parsed_off, parsed_off.workload.requests[0])
    on_policy = DraftPolicy.for_request(parsed_on, parsed_on.workload.requests[0])
    assert off_policy.proposal == 0
    assert on_policy.proposal == proposal
    assert on_policy.mean == pytest.approx(acceptance)
    assert on_policy.mode in {"mtp", "dspark"}
    off, off_row = _sla_baseline(off_case, context=context, output_budget=output)
    on, on_row = _sla_baseline(on_case, context=context, output_budget=output)
    assert off["tokens_per_seq"] == 1
    assert on["tokens_per_seq"] == 1 + proposal
    assert off["emitted_tokens_per_seq"] == pytest.approx(1.0)
    assert on["emitted_tokens_per_seq"] == pytest.approx(acceptance)
    single_token_step_ms = off_row["step_ms_at_b1"]
    verify_step_ms = on_row["step_ms_at_b1"]
    tpot_effective_ms = verify_step_ms / acceptance
    assert verify_step_ms > single_token_step_ms
    assert tpot_effective_ms < single_token_step_ms
    off_eff = analyze_network_efficiency(off_case, phase="decode", tokens=off["tokens_per_seq"],
        context=context, sequences=1, scope="body")
    on_eff = analyze_network_efficiency(on_case, phase="decode", tokens=on["tokens_per_seq"],
        context=context, sequences=1, scope="body")
    assert on_eff["total_hbm_bytes"] > off_eff["total_hbm_bytes"]
    assert on_eff["operator_sum_ms"] > off_eff["operator_sum_ms"]

def test_mtp_and_dspark_scale_verify_rows_and_effective_tpot():
    _assert_automatic_speculative_scaling("kimi-k3", "Ascend950DT_9582", quant=None, tp=2, dp=1, ep=2,
        proposal=3, acceptance=3.0)
    _assert_automatic_speculative_scaling("qwen3.5-35b-a3b", "Ascend910_9392", quant="w8a8", tp=2, dp=2, ep=4,
        proposal=7, acceptance=7.0)
    kimi = _case("kimi-k3", "Ascend950DT_9582", tp=2, dp=1, ep=2, layers=2, mtp=3, accept=3.0)
    assert kimi["runtime"]["metadata"]["draft_mode"] == "dspark"
    disabled = _case("kimi-k3", "Ascend950DT_9582", tp=2, dp=1, ep=2, layers=2)
    assert "draft_mode" not in disabled["runtime"]["metadata"]

def test_pp_pcp_ep_sp_scaling_matches_roofline_and_bytes():
    curves = {}
    for model, layers in PIPELINE:
        case = _case(model, "a3", tp=2, dp=1, pp=2, pcp=2, ep=2, sp=True, layers=layers, experts=4, topk=1)
        short = _step(case, "prefill", 128, 0, 1)["duration_ms"]
        longer = _step(case, "prefill", 1024, 0, 1)["duration_ms"]
        assert longer > short
        curves[model] = case
    qwen = curves["qwen3.5-397b-a17b"]
    by_tokens = [(_step(qwen, "prefill", tokens, 0, 1)["duration_ms"], tokens) for tokens in (128, 1024, 4096, 32768)]
    assert [ms for ms, _ in by_tokens] == sorted(ms for ms, _ in by_tokens)
    by_conc = [by_tokens[0][0]]
    for sequences in (4, 16):
        by_conc.append(_step(qwen, "prefill", 128, 0, sequences)["duration_ms"])
    assert by_conc == sorted(by_conc) and by_conc[0] < by_conc[-1]
    short_eff = analyze_network_efficiency(qwen, scope="body", phase="prefill", tokens=128, context=0, sequences=1)
    long_eff = analyze_network_efficiency(qwen, scope="body", phase="prefill", tokens=32768, context=0, sequences=1)
    assert _rel(short_eff["graph_duration_ms"], by_tokens[0][0]) <= REL_TOL
    assert _rel(long_eff["graph_duration_ms"], by_tokens[-1][0]) <= REL_TOL
    assert long_eff["total_roofline_ideal_ms"] > short_eff["total_roofline_ideal_ms"]
    assert long_eff["total_hbm_bytes"] > short_eff["total_hbm_bytes"]
    assert long_eff["total_link_bytes"] > short_eff["total_link_bytes"]
    assert short_eff["graph_duration_ms"] + 1e-9 >= short_eff["total_roofline_ideal_ms"]
    _conserve(short_eff, short_eff["graph_duration_ms"])
    mla = curves["deepseek-v4-pro"]
    decode = _step(mla, "decode", 1, 1024, 1)
    eff = analyze_network_efficiency(mla, scope="body", phase="decode", tokens=1, context=1024, sequences=1)
    assert _rel(eff["graph_duration_ms"], decode["duration_ms"]) <= REL_TOL
    _conserve(eff, decode["duration_ms"])
    rows = dict(cache_row_bytes_by_layer(parse_case(mla).model))
    stores = [op for op in eff["operators"] if op["recipe"] == "kv_store"]
    ropes = [op for op in eff["operators"] if op["recipe"] == "rotary"]
    attn = [op for op in eff["operators"] if op["recipe"] == "latent_attention0"]
    assert stores and ropes and attn
    assert {op["hbm_bytes"] for op in stores} == {2 * rows[stores[0]["layer"]]}
    assert len({op["hbm_bytes"] for op in ropes}) == 1 and ropes[0]["hbm_bytes"] > 0
    assert min(op["hbm_bytes"] for op in attn) > max(op["hbm_bytes"] for op in stores)
    moe_link = sum(op["link_bytes"] * op["occurrences"] for op in eff["operators"]
        if op["recipe"] in {"expert_dispatch", "expert_combine"})
    assert moe_link > 0
    assert moe_link <= eff["total_link_bytes"]
