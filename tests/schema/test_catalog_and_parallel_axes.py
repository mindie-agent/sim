"""Tier-0 tests for 39-model catalog, integer cache pools, parallel axes & bench tools.

[Role]
  Verify the expanded 39-model catalog and submodule re-exports, exact integer
  ceiling arithmetic and breakdown-to-pitch consistency in ``cache`` /
  ``cache_pools`` / ``cache_format``, all five multi-axis parallel combinations
  (DPxEP, DCPxEP, PCPxEP, PCPxDCP, PCPxTP), ``expert_tp_groups`` without empty
  tuples, ``world_size`` and non-contiguous ``rank_table`` validation, idle-DP
  cohort rendezvous, and zero-fallback ``tools/bench_*`` contracts.

[Invariants]
  * Pure in-memory Tier-0 execution with zero subprocesses, zero device
    context, and zero host environment mutations.
  * Locks every integer arithmetic, DCP draft-sharding, non-summed pool pitch,
    topology error propagation, physical fabric rank validation, display-vs-ABI
    separation, redundant expert-slot conservation, and packet validation rule.

[Timeouts & Locks]
  Deterministic in-memory checks (< 5s); no locks or network I/O.

[Side-Effect Policy]
  Pure read-only in-memory checks; never mutates repository files or env vars.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog import cache, hf_config, weights
from mindie_sim.catalog.attention_common import active_dcp_degree
from mindie_sim.catalog.cache_format import CacheFormat
from mindie_sim.catalog.cache_index import index_k_capacity_rows
from mindie_sim.catalog.cache_pools import (
    exact_element_bytes,
    request_pool_page_pitches,
    separate_pool_page_pitches,
    target_mla_page_bytes,
)
from mindie_sim.catalog.hf_value_formats import model_value_formats
from mindie_sim.catalog.speculative_state import speculative_state_plan
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec
from mindie_sim.catalog.weight_layer_terms import layer_weight_terms
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES, load_model
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import physical_slot_work
from mindie_sim.prediction.local_tp_groups import collective_requires_cohort, expert_tp_groups, local_tp_groups
from mindie_sim.prediction.operator_analysis import synchronize_moe_allgather_cohort
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_structure
from scripts.verify_m1 import CATALOG_MODELS, _manifest_reference_label
from tools.bench_packets import PacketError, assert_public_provenance, build_packets
from tools.bench_plan_builders import (
    BOUNDARY_BUILDER,
    BenchGeometryError,
    BenchTopologyError,
    UnmodeledCapabilityError,
    fabric_route_ranks,
    resolve_builder_strict,
    resolve_collective_cosets,
    resolve_collective_group_ranks,
    resolve_fabric_collective_abi,
)
from tools.bench_plan_parallel import PlanError, slot_plan_for
from tools.export_subgraph_bench_plan import (
    _declared_implementation,
    _display_name,
    _expert_bank_bytes,
    _explicit_ledger,
    _fabric_cases,
    _hardware_view,
    _node_fields,
)

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_inventory_reexports_and_relative_manifest_label() -> None:
    assert len(CATALOG) == 39
    assert len(CATALOG_MODELS) == 39
    new_mla = load_model("kimi-k2.6")
    new_hybrid = load_model("qwen3.5-122b-a10b")
    assert new_mla.attention == "mla" and new_mla.num_experts == 384 and new_mla.layers == 61
    assert new_hybrid.num_experts == 256 and new_hybrid.layers == 48
    assert load_model(MODEL_ALIASES["Kimi-K2.6"]) == new_mla
    assert load_model(MODEL_ALIASES["Qwen3.5-122B-A10B"]) == new_hybrid

    assert hf_config.model_value_formats is model_value_formats
    assert weights.layer_weight_terms is layer_weight_terms
    assert cache.index_k_capacity_rows is index_k_capacity_rows
    assert cache.separate_pool_page_pitches is separate_pool_page_pitches

    manifest = REPO_ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
    label = _manifest_reference_label(manifest)
    assert label == "tests/fixtures/catalog_reference_manifest.json"
    assert not label.startswith("/")


def test_cache_pools_integer_ceiling_arithmetic_and_breakdown_consistency() -> None:
    # Exact sub-byte, MX group-32, and >2**53 integer ceiling arithmetic with zero float loss.
    assert exact_element_bytes(3, "int4") == 2
    assert exact_element_bytes(5, "int4") == 3
    assert exact_element_bytes(33, "mxfp4") == 17 + 2
    assert exact_element_bytes(33, "mxfp8") == 33 + 2
    assert CacheFormat("mxfp4").row_bytes(33) == 19
    huge_width = (1 << 53) + 1
    assert CacheFormat("int8").row_bytes(huge_width) == huge_width

    mxfp4_layer = LayerPlan(
        index=0, attention="mla", ffn="dense", num_heads=16, num_kv_heads=1,
        head_dim=128, kv_lora_rank=33, qk_rope_head_dim=0,
    )
    mxfp4_model = ModelSpec(
        name="mxfp4-mla", layers=1, hidden_size=512, intermediate_size=1024,
        num_heads=16, num_kv_heads=1, head_dim=128, vocab_size=1024,
        attention="mla", ffn="dense", quant=QuantSpec("bf16", kv_dtype="mxfp4"),
        layer_plans=(mxfp4_layer,),
    )
    assert target_mla_page_bytes(mxfp4_model, 1) == 19
    huge_model = replace(
        mxfp4_model,
        quant=QuantSpec("bf16", kv_dtype="int8"),
        layer_plans=(replace(mxfp4_layer, kv_lora_rank=huge_width),),
        metadata=(("cache_representations", {"mla_kv": {"dtype": "int8"}}),),
    )
    assert target_mla_page_bytes(huge_model, 1) == huge_width

    # Mixed MLA/KDA/replicated-draft fixture (TP=4, DCP=2, block=16, 32 tokens):
    # request_cache_breakdown must match independent pool pitches exactly (8788 B).
    mla_layer = LayerPlan(
        index=0, attention="mla", ffn="dense", num_heads=16, num_kv_heads=1,
        head_dim=128, kv_lora_rank=512, qk_rope_head_dim=0,
    )
    linear_layer = LayerPlan(
        index=1, attention="kda", ffn="dense", num_heads=16, num_kv_heads=4,
        head_dim=128, linear_heads=16, linear_head_dim=128, conv_kernel=4,
        recurrent_state_elements=13, conv_state_elements=7,
        recurrent_state_dtype="fp32", conv_state_dtype="bf16",
    )
    replicated_draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=512, intermediate_size=1024,
        num_heads=8, num_kv_heads=2, head_dim=5, q_width=40, kv_width=10,
        vocab_size=1024, attention_kind="gqa", dcp_sharded=False,
    )
    sharded_mla_draft = replace(
        replicated_draft, attention_kind="mla", kv_lora_rank=512,
        qk_rope_head_dim=64, kv_cache_elements=576, dcp_sharded=True,
    )
    bad_gqa_draft = replace(replicated_draft, num_kv_heads=4, head_dim=128, dcp_sharded=True)
    ok_gqa_draft = replace(replicated_draft, num_kv_heads=2, head_dim=128, dcp_sharded=True)

    base_model = ModelSpec(
        name="synthetic-hybrid", layers=2, hidden_size=512, intermediate_size=1024,
        num_heads=16, num_kv_heads=4, head_dim=128, vocab_size=1024,
        attention="mla", ffn="dense", quant=QuantSpec("bf16", kv_dtype="int8"),
        layer_plans=(mla_layer, linear_layer),
        auxiliary=AuxiliarySpec(mtp=replicated_draft),
    )
    par_dcp2 = ParallelSpec(tp=4, cp=2, cp_plan="dcp")
    pitches = separate_pool_page_pitches(
        base_model, par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )
    assert pitches["state_page_bytes"] == 20
    assert pitches["target_mla_page_bytes"] == 16 * 528 == target_mla_page_bytes(base_model, 16)
    assert pitches["draft_page_bytes"] == 160 and pitches["draft_sharded"] is False
    assert "combined_page_bytes" not in pitches
    assert pitches["independent_pool_pitches"] == {
        "target_mla": 8448, "recurrent_state": 20, "draft": 160,
    }
    assert pitches["target_page_unchanged_by_draft"] is True

    breakdown = cache.request_cache_breakdown(base_model, par_dcp2, 32, draft_kind="mtp", interleave=16)
    assert breakdown["by_component"]["mla_kv"] == 8448.0
    assert breakdown["by_component"]["recurrent_state"] + breakdown["by_component"]["conv_state"] == 20.0
    assert breakdown["by_component"]["draft_kv"] == 2 * 160.0
    assert breakdown["total_bytes"] == 8788.0
    # Per-layer catalog row report matches packed INT8 MLA capacity row width (528 B).
    assert weights.cache_row_bytes_by_layer(base_model)[0] == (0, 528)
    # Explicit interleave=128 overrides runtime.kv_block_size=16; omitted interleave uses runtime block size.
    rt_b16 = SimpleNamespace(kv_block_size=16, metadata=())
    bd_explicit = cache.request_cache_breakdown(base_model, par_dcp2, 32, interleave=128, runtime=rt_b16)
    bd_default = cache.request_cache_breakdown(base_model, par_dcp2, 32, runtime=rt_b16)
    assert bd_explicit["by_component"]["mla_kv"] == 67584.0
    assert bd_default["by_component"]["mla_kv"] == 8448.0
    # KDA rollback q=1 -> q=2 preserves integer-sized base conv history (4 B -> 6 B, never 0 B).
    conv_q1 = speculative_state_plan(None, linear_layer, par_dcp2, query_tokens=1, recover_ssm=False, capacity_padded=True)
    conv_q2 = speculative_state_plan(None, linear_layer, par_dcp2, query_tokens=2, recover_ssm=False, capacity_padded=True)
    assert conv_q1.conv_bytes == 4.0 and conv_q2.conv_bytes == 6.0

    # draft_sharded checks actual draft DCP sharding rather than dcp > 1 alone.
    model_mla_draft = replace(base_model, auxiliary=AuxiliarySpec(mtp=sharded_mla_draft))
    assert separate_pool_page_pitches(model_mla_draft, par_dcp2, block_size=16, draft_kind="mtp", dcp=2)["draft_sharded"] is True
    assert separate_pool_page_pitches(
        model_mla_draft, ParallelSpec(tp=4, cp_plan="dsa_cp"), block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is False
    assert separate_pool_page_pitches(
        replace(base_model, auxiliary=AuxiliarySpec(mtp=bad_gqa_draft)), par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is False
    assert separate_pool_page_pitches(
        replace(base_model, auxiliary=AuxiliarySpec(mtp=ok_gqa_draft)), par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is True

    odd_model = replace(
        base_model,
        quant=QuantSpec("bf16", kv_dtype="int4"),
        auxiliary=AuxiliarySpec(mtp=replace(sharded_mla_draft, kv_cache_elements=5)),
    )
    odd_report = request_pool_page_pitches(odd_model, par_dcp2, block_size=4, draft_kind="mtp", dcp=2)
    assert odd_report is not None and odd_report["draft_page_bytes"] == 4 * 3


def test_parallel_world_size_five_axes_expert_tp_and_rank_table_contracts() -> None:
    spec = ParallelSpec.from_mapping({
        "external_data_parallel_size": 2,
        "dp": 2, "pp": 2, "pcp": 2, "tp": 2, "dcp": 4, "ep": 8,
        "world_size": 32,
    })
    assert spec.world_size == 32 and spec.to_dict()["external_dp"] == 2
    assert "external_dp" not in ParallelSpec(dp=2, tp=2).to_dict()
    with pytest.raises(ModelingError, match="world_size_conflict"):
        ParallelSpec.from_mapping({"dp": 2, "tp": 2, "dcp": 2, "ep": 4, "world_size": 16})

    auto_ep = ParallelSpec.from_mapping({
        "dp": 2, "tp": 2, "pcp": 2, "enable_expert_parallel": True,
        "pcp_lane_policy": "shard_decode_requests",
    })
    assert auto_ep.ep == 8 and auto_ep.ep_spans_pcp is True and auto_ep.pcp_shard_decode_requests is True

    mla_moe = load_model("deepseek-v2-lite")
    dense_gqa = load_model("qwen3-32b")
    for model, kw in (
        (mla_moe, {"dp": 4, "tp": 1, "ep": 4}),
        (mla_moe, {"dp": 1, "tp": 4, "dcp": 2, "ep": 4}),
        (mla_moe, {"dp": 1, "tp": 2, "pcp": 2, "ep": 4}),
        (mla_moe, {"dp": 1, "tp": 2, "pcp": 2, "dcp": 4, "ep": 4}),
        (dense_gqa, {"dp": 1, "tp": 4, "pcp": 2, "ep": 1}),
        (dense_gqa, {"dp": 1, "tp": 8, "pcp": 2, "dcp": 2, "ep": 1}),
    ):
        par = ParallelSpec(**kw)
        par.validate(model)
        assert active_dcp_degree(par, "prefill") == 1
        assert active_dcp_degree(par, "decode") == kw.get("dcp", 1)

    # Intermediate PCP+TP DCP (pcp=2, tp=4, dcp=4 -> pcp_factor=2, tp_gather=2) is algebraically valid on MLA.
    ParallelSpec(tp=4, pcp=2, dcp=4, ep=1).validate(mla_moe)
    with pytest.raises(ModelingError, match="dcp_factorization_invalid"):
        ParallelSpec(tp=4, pcp=2, dcp=3, ep=1).validate(mla_moe)
    with pytest.raises(ModelingError, match="cp_exceeds_gqa_decode_domain"):
        ParallelSpec(tp=2, pcp=2, dcp=4, ep=1).validate(dense_gqa)

    # Expert-TP groups under TP=2, PCP=2, EP=4 emit four singletons with zero empty tuples.
    pcp_ep_case = parse_case(compose_case("deepseek-v2-lite", "Ascend910_9362", tp=2, pcp=2, ep=4, layers=2))
    pcp_ep_topo = Topology(pcp_ep_case.machine, pcp_ep_case.runtime.parallel)
    assert pcp_ep_topo.groups("etp", 0, 0) == expert_tp_groups(pcp_ep_topo, replica=0, stage=0) == ((0,), (1,), (2,), (3,))
    assert resolve_collective_cosets(pcp_ep_topo, {"group_axis": "etp"}, stage=0) == [[0], [1], [2], [3]]
    assert collective_requires_cohort(SimpleNamespace(stage=0, metadata=(("group_axis", "etp"),)), pcp_ep_topo) is False

    # Non-trivial etp=2 with TP=4, PCP=2, EP=2 yields strided pairs ((0,2),(1,3),(4,6),(5,7)).
    etp2_case = parse_case(compose_case("deepseek-v2-lite", "Ascend910_9362", tp=4, pcp=2, ep=2, layers=2))
    etp2_topo = Topology(etp2_case.machine, etp2_case.runtime.parallel)
    assert etp2_topo.groups("etp", 0, 0) == expert_tp_groups(etp2_topo, replica=0, stage=0) == (
        (0, 2), (1, 3), (4, 6), (5, 7),
    )
    # Explicit EP rank_table [[0,2],[1,3]] under TP=4 groups equal EP positions -> ((0,1),(2,3)).
    etp_tbl_case = parse_case(compose_case("deepseek-v2-lite", "Ascend910_9362", tp=4, dp=1, ep=2, layers=2))
    etp_tbl_par = ParallelSpec(tp=4, dp=1, ep=2, rank_table=((0, 2), (1, 3)))
    etp_tbl_topo = Topology(etp_tbl_case.machine, etp_tbl_par)
    assert etp_tbl_topo.groups("etp", 0, 0) == expert_tp_groups(etp_tbl_topo, replica=0, stage=0) == ((0, 1), (2, 3))
    # External-DP local-TP builds groups independently inside each external replica -> ((0,1),(2,3)).
    ext_ltp_case = parse_case(compose_case("qwen3-32b", "Ascend910_9362", tp=1, dp=2, ep=1, num_devices=4, layers=1))
    ext_ltp_par = ParallelSpec(external_dp=2, dp=2, tp=1, ep=1, o_proj_tp=2)
    assert local_tp_groups(ext_ltp_case.machine, ext_ltp_par, stage=0, width=2) == ((0, 1), (2, 3))
    assert resolve_collective_cosets(
        Topology(ext_ltp_case.machine, ext_ltp_par), {"group_axis": "local_tp", "local_tp_width": 2}, stage=0,
    ) == [[0, 1], [2, 3]]

    non_contig = ParallelSpec(dp=2, tp=2, pcp=2, ep=2, rank_table=((0, 4), (2, 6)))
    non_contig.validate(mla_moe)
    with pytest.raises(ModelingError, match="rank_table_ranks_exceed_world"):
        ParallelSpec(dp=2, tp=1, ep=2, rank_table=((0, 1), (2, 3))).validate(mla_moe)
    with pytest.raises(ModelingError, match="rank_table_duplicate_rank"):
        ParallelSpec(dp=2, tp=2, ep=2, rank_table=((0, 2), (2, 3)))


def test_idle_dp_replica_collective_rendezvous_modes() -> None:
    base_raw = compose_case(
        "deepseek-v2-lite", hardware="Ascend910_9362",
        dp=4, tp=1, ep=4, layers=3, concurrency=4, graph_capture_mode="full",
    )
    base_raw["runtime"]["graph_mode"] = "full"
    base_raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    parsed_full = parse_case(base_raw)
    logical = {0: (RequestView("r0", 2, 64, False, "decode"),), 1: (RequestView("r1", 1, 64, False, "decode"),), 2: (), 3: ()}

    synced_full, padded_full = synchronize_moe_allgather_cohort(parsed_full, "decode", logical, pad_idle=True, default_context=64)
    assert padded_full is None
    assert synced_full[2][0].q_len == 2 and synced_full[2][0].padding is True
    assert synced_full[3][0].q_len == 2 and synced_full[3][0].padding is True

    ag_case = replace(parsed_full, runtime=replace(parsed_full.runtime, graph_mode="eager", metadata=(*parsed_full.runtime.metadata, ("moe_comm_mode", "allgather"))))
    synced_ag, padded_ag = synchronize_moe_allgather_cohort(ag_case, "decode", logical, pad_idle=False, default_context=64)
    assert padded_ag == 4 * 2 and all(sum(v.q_len for v in synced_ag[r]) == 2 for r in range(4))

    dummy_case = replace(parsed_full, runtime=replace(parsed_full.runtime, graph_mode="eager", metadata=(*parsed_full.runtime.metadata, ("eager_dp_dummy_run", True))))
    synced_dummy, _ = synchronize_moe_allgather_cohort(dummy_case, "decode", logical, pad_idle=False, default_context=64)
    assert synced_dummy[2][0].request_id.endswith(":dummy_run") and synced_dummy[2][0].q_len == 1 and synced_dummy[2][0].padding is True

    plain_case = replace(parsed_full, runtime=replace(parsed_full.runtime, graph_mode="eager", moe_recipe="alltoall"))
    synced_plain, _ = synchronize_moe_allgather_cohort(plain_case, "decode", logical, pad_idle=False, default_context=64)
    assert synced_plain[2] == () and synced_plain[3] == ()

    graphs = target_graphs(parsed_full, phase="decode", tokens=1, context=64, sequences=(2, 2, 0, 0), scope="body")
    assert sorted(graphs) == [0, 1, 2, 3]
    idle_attn = next(op for op in graphs[2] if op.family == "attention")
    assert dict(idle_attn.metadata)["graph_active_query_rows"] == 0
    assert dict(idle_attn.metadata)["graph_physical_query_rows"] == 2


def test_bench_plan_builders_zero_fallback_cosets_and_fabric_routes() -> None:
    assert resolve_builder_strict("barrier", "sync", "", ()) == (BOUNDARY_BUILDER, "barrier")
    assert resolve_builder_strict("rmsnorm", "rms_norm", "", ("bf16",)) == ("rms_norm", "aclnnRmsNorm")
    assert resolve_fabric_collective_abi("all_reduce") == ("all_reduce", "hcclAllReduce")
    with pytest.raises(UnmodeledCapabilityError) as exc_info:
        resolve_builder_strict("unknown_family", "unknown_recipe", "", ("bf16",), model_id="m", layer_idx=0)
    assert exc_info.value.payload["schema"] == "mindie-unmodeled-structure-hint/1"

    parsed = parse_case(compose_case("deepseek-v2-lite", hardware="Ascend910_9362", dp=2, tp=2, ep=4, layers=2))
    topo = Topology(parsed.machine, parsed.runtime.parallel)
    assert resolve_collective_cosets(topo, {"group_axis": "tp"}, stage=0) == [[0, 1], [2, 3]]
    assert resolve_collective_group_ranks(topo, {"group_axis": "tp"}, stage=0, rank=2) == [2, 3]
    assert resolve_collective_group_ranks(topo, {"group_axis": "tp", "group_ranks": [2, 3]}, stage=0) == [2, 3]
    assert resolve_collective_cosets(topo, {"group_axis": "dp"}, stage=0) == [[0, 2], [1, 3]]
    assert resolve_collective_cosets(topo, {"group_axis": "ep"}, stage=0) == [[0, 1, 2, 3]]
    with pytest.raises(BenchTopologyError, match="collective_rank_not_in_cosets"):
        resolve_collective_group_ranks(topo, {"group_axis": "tp"}, stage=0, rank=99)
    with pytest.raises(BenchTopologyError, match="collective_declared_group_unmatched"):
        resolve_collective_group_ranks(topo, {"group_axis": "tp", "group_ranks": [0, 3]}, stage=0)
    with pytest.raises(BenchTopologyError, match="collective_metadata_rank_not_in_declared_group"):
        resolve_collective_group_ranks(topo, {"group_axis": "tp", "group_ranks": [0, 1], "physical_rank": 99}, stage=0)
    with pytest.raises(BenchTopologyError):
        resolve_collective_cosets(topo, {"group_axis": "bogus_axis"}, stage=0)

    # external_dp=2, dp=1, tp=2 enumerates every external replica coset.
    ext_case = parse_case(compose_case("qwen3-32b", "Ascend910_9362", dp=1, tp=2, ep=1, num_devices=4, layers=1))
    ext_topo = Topology(ext_case.machine, ParallelSpec(external_dp=2, dp=1, tp=2, ep=1))
    assert resolve_collective_cosets(ext_topo, {"group_axis": "tp"}, stage=0) == [[0, 1], [2, 3]]

    # Node exporter honors declared group_ranks / physical_rank and rejects absent replica_idx.
    built_coll = _node_fields(
        node_id="c0", family="allreduce", recipe="tp_allreduce",
        operator={"dtype": "bf16", "local_shape": [4, 64], "metadata": {"group_axis": "tp", "group_ranks": [2, 3]}},
        metadata={"group_axis": "tp", "group_ranks": [2, 3]}, deps=(), outputs=(),
        parallel={"tp": 2, "dp": 2, "ep": 4}, topology=topo,
    )
    assert built_coll["group_ranks"] == [2, 3] and built_coll["group_cosets"] == [[0, 1], [2, 3]]
    with pytest.raises(BenchTopologyError, match="collective_replica_has_no_ranks"):
        _node_fields(
            node_id="c1", family="allreduce", recipe="tp_allreduce",
            operator={"dtype": "bf16", "local_shape": [4, 64], "metadata": {"group_axis": "tp"}},
            metadata={"group_axis": "tp"}, deps=(), outputs=(),
            parallel={"tp": 2, "dp": 2, "ep": 4}, topology=topo, replica_idx=99,
        )

    # Fabric routes: single-node 16-die machine never emits inter_node_roce or ranks >= 16;
    # 3-node topology with RoCE only on nodes 1-2 picks nodes 1-2 endpoints (not node 0 SuperPod).
    single_node_view = _hardware_view("Ascend910_9362", "Ascend910_9362")
    assert single_node_view["nodes"] == 1 and single_node_view["has_inter_node_link"] is False
    fab_cases = _fabric_cases(single_node_view)
    assert fab_cases and all(c["route"] != "inter_node_roce" for c in fab_cases)
    assert all(r < 16 for c in fab_cases for r in c["group_ranks"])
    ar_case = next(c for c in fab_cases if c["recipe"] == "all_reduce")
    assert ar_case["nodes"][0]["upstream_abi"] == "hcclAllReduce"
    with pytest.raises(BenchTopologyError, match="fabric_inter_node_requires_multi_node"):
        fabric_route_ranks("inter_node_roce", 8, single_node_view)
    hw_3node = {
        "dies_per_package": 2, "dies_per_node": 4, "nodes": 3,
        "ranks_by_node": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        "declared_links": [
            {"name": "superpod_0_1", "channel": "superpod", "members": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"name": "inter_node_roce_1_2", "channel": "roce", "members": [4, 5, 6, 7, 8, 9, 10, 11]},
        ],
    }
    assert fabric_route_ranks("inter_node_roce", 4, hw_3node) == [4, 5, 8, 9]

    dual_node_hw = {"dies_per_package": 2, "dies_per_node": 16, "nodes": 2}
    single_die_hw = {"dies_per_package": 1, "dies_per_node": 8, "nodes": 1}
    assert fabric_route_ranks("intra_package_die", 2, dual_node_hw) == [0, 1]
    assert fabric_route_ranks("intra_node_hccs", 4, dual_node_hw) == [0, 2, 4, 6]
    assert fabric_route_ranks("inter_node_roce", 8, dual_node_hw) == [0, 1, 2, 3, 16, 17, 18, 19]
    with pytest.raises(BenchGeometryError, match="fabric_intra_package_unrealizable"):
        fabric_route_ranks("intra_package_die", 2, single_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_intra_node_exceeds_node"):
        fabric_route_ranks("intra_node_hccs", 16, single_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_route_unknown"):
        fabric_route_ranks("mystery_link", 2, dual_node_hw)
    with pytest.raises(BenchGeometryError, match="fabric_route_ranks_unavailable"):
        fabric_route_ranks("intra_node_hccs", 4, dual_node_hw, available_ranks=[0, 2, 4])


def test_export_plan_abi_separation_redundant_expert_slots_and_packet_validation() -> None:
    assert _declared_implementation({}) == ""
    assert _display_name("rmsnorm", "rms_norm") == "rmsnorm:rms_norm"
    built_default = _node_fields(
        node_id="n0", family="rmsnorm", recipe="rms_norm",
        operator={"dtype": "bf16", "local_shape": [8, 512], "metadata": {}},
        metadata={}, deps=(), outputs=(), parallel={"tp": 1, "ep": 1},
    )
    assert built_default["display_name"] == "rmsnorm:rms_norm"
    assert built_default["upstream_abi"] == "aclnnRmsNorm"
    assert built_default["implementation"] == "aclnnRmsNorm"
    assert built_default["implementation_source"] == "builder_upstream_abi"

    built_explicit = _node_fields(
        node_id="n1", family="rmsnorm", recipe="rms_norm",
        operator={"dtype": "bf16", "local_shape": [8, 512], "metadata": {"native_kind": "CustomRmsNorm"}},
        metadata={"native_kind": "CustomRmsNorm"}, deps=(), outputs=(), parallel={"tp": 1, "ep": 1},
    )
    assert built_explicit["implementation"] == "CustomRmsNorm"
    assert built_explicit["implementation_source"] == "metadata"
    assert built_explicit["upstream_abi"] == "aclnnRmsNorm"

    # Shared expert_slot_plan for E=10, EP=4, redundant=1 keeps bank and ledger aligned (4 slots/rank).
    padded_model = SimpleNamespace(
        num_experts=10, moe_intermediate_size=256, hidden_size=512,
        layer_plans=(), quant=QuantSpec("bf16"),
    )
    slots = slot_plan_for(10, 4, redundant=1)
    bank = _expert_bank_bytes(padded_model, {"ep": 4, "redundant_experts_per_rank": 1}, slots=slots)
    assert bank is not None and bank["base_local_experts"] == 3 and bank["local_experts"] == 4
    expert_rows = [2] * 10
    topk_ids = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]] * 2
    ledger = _explicit_ledger(expert_rows, topk_ids, ep=4, slots=slots)
    assert ledger["padded_expert_slots"] == 2 and len(ledger["group_list"]) == bank["local_experts"] == 4
    assert ledger["local_weight_expert_ids"] == bank["local_weight_expert_ids"]
    assert sum(ledger["rank_assignment_rows"]) == sum(expert_rows) == 20
    # Redundant-slot round-robin matches runtime physical_slot_work ([1, 1, 0, 0] for topk=[[3],[3]]).
    rr_rows = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    rr_ledger = _explicit_ledger(rr_rows, [[3], [3]], ep=4, slots=slots)
    rr_runtime = physical_slot_work(slot_plan=slots, logical_inclusions={}, global_tokens=2, routing_realization={3: 2})
    assert rr_ledger["rank_assignment_rows"] == [1, 1, 0, 0] == [int(v) for v in rr_runtime.expected_rows_by_rank]
    with pytest.raises(PlanError, match="explicit_ledger_topk_histogram_mismatch"):
        _explicit_ledger([1] + [0] * 9, [[9]], ep=4, slots=slots)

    # Bench packets provenance scrubber, group_size rank-count check, and explicit split/URI.
    with pytest.raises(PacketError, match="private_provenance_home_path"):
        assert_public_provenance({"note": "/mnt/private/run"})
    with pytest.raises(PacketError, match="private_provenance_ipv4"):
        assert_public_provenance({"src": "10.0.0.1"})
    with pytest.raises(PacketError, match="observation_requires_explicit_split"):
        build_packets([{"status": "ok", "source": "capture://run1/s0", "duration_ms": 1.0}])

    def _fabric_row(rank: int, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "schema": "mindie-fabric-probe-sample/3", "validated": True, "timing_mode": "device_event",
            "device_completion_us": 50.0, "run_id": "r0", "tier": "hccs", "op": "all_reduce",
            "group_size": 4, "bytes": 4096, "protocol": "ring", "regime": "mid",
            "split_style": "even", "trial_index": 0, "repeat_index": 0, "pair_index": 0, "round_index": 0,
            "group_ranks": [0, 1], "rank": rank, "hardware": "Ascend910_9362",
            "split": "train", "source": "capture://run0/fab",
        }
        row.update(overrides)
        return row

    with pytest.raises(PacketError, match="fabric_group_size_rank_count_mismatch"):
        build_packets([_fabric_row(0), _fabric_row(1)])
    with pytest.raises(PacketError, match="fabric_sample_requires_explicit_split"):
        build_packets([_fabric_row(0, group_size=2, split=None), _fabric_row(1, group_size=2, split=None)])
    with pytest.raises(PacketError, match="observation_requires_explicit_source"):
        build_packets([_fabric_row(0, group_size=2, source=None), _fabric_row(1, group_size=2, source=None)])

    # 4-rank send_recv repeat across pairs [0,1] and [2,3] validates communicator rank count cleanly;
    # heldout 32 KiB all_reduce preserves/resolves hcclAllReduce for accuracy pairing.
    sr_rows = [
        row
        for size in (4096, 16384)
        for row in (
            _fabric_row(0, op="send_recv", bytes=size, group_size=4, group_ranks=[0, 1, 2, 3], pair=[0, 1], pair_index=0, device_completion_us=50.0),
            _fabric_row(1, op="send_recv", bytes=size, group_size=4, group_ranks=[0, 1, 2, 3], pair=[0, 1], pair_index=0, device_completion_us=50.5),
            _fabric_row(2, op="send_recv", bytes=size, group_size=4, group_ranks=[0, 1, 2, 3], pair=[2, 3], pair_index=1, device_completion_us=52.0),
            _fabric_row(3, op="send_recv", bytes=size, group_size=4, group_ranks=[0, 1, 2, 3], pair=[2, 3], pair_index=1, device_completion_us=52.5),
        )
    ]
    sr_bundle = build_packets(sr_rows)
    assert sr_bundle["manifest"]["observations"] == 4 and len(sr_bundle["fit-rate"]) == 1
    assert sr_bundle["rate_conditions"][0]["identity"]["implementation"] == "hcclSendRecv"
    heldout_ar = [
        _fabric_row(r, op="all_reduce", group_size=4, group_ranks=[0, 1, 2, 3], bytes=32768, split="heldout",
                    identity={"implementation": "hcclAllReduce"})
        for r in range(4)
    ]
    ar_bundle = build_packets(heldout_ar)
    assert ar_bundle["accuracy"]["observations"][0]["identity"]["implementation"] == "hcclAllReduce"

    case_spec = parse_case(compose_case("qwen3-32b", hardware="Ascend910_9362", tp=4, pcp=2, ep=1, layers=2))
    rep = export_structure(case_spec, phase="prefill", tokens=64, context=0, sequences=1)
    assert any((n.get("operator") or {}).get("recipe") == "pcp_kv_gather" for n in rep["nodes"])


from mindie_sim.model_catalog import list_models as _list_catalog_models

_CATALOG_KEYS = tuple(entry["key"] for entry in _list_catalog_models())


@pytest.mark.parametrize("key", _CATALOG_KEYS, ids=list(_CATALOG_KEYS))
def test_physical_key_count_is_bounded_by_layer_archetypes_not_layer_count(key):
    from mindie_sim import target_graphs
    from mindie_sim.prediction.physical_key import physical_key

    cases = {n: compose_case(key, hardware="a5", layers=n) for n in (2, 6, 12)}
    counts = {
        n: len({
            physical_key(op)
            for ops in target_graphs(case, scope="body", phase="decode", tokens=1, context=128, sequences=1).values()
            for op in ops
        })
        for n, case in cases.items()
    }
    assert counts[2] > 0 and counts[6] > 0 and counts[12] > 0
    assert counts[6] <= 3 * counts[2], (key, counts)
    archetypes = {
        (p["attention"], p["ffn"], p["indexer"], p["window"], p["compress_ratio"])
        for p in cases[6]["model"]["layer_plans"]
    }
    if len(archetypes) == 1:
        assert counts[2] == counts[6] == counts[12], (key, counts)
    else:
        assert counts[12] <= counts[6] + counts[2] + 4, (key, counts)
