"""Tier-0 tests for 39-model catalog, integer cache pools, parallel axes & bench tools.

[Role]
  Verify the expanded 39-model catalog and submodule re-exports, exact integer
  ceiling arithmetic and independent pool pitches in ``cache_pools``, all five
  multi-axis parallel combinations (DPxEP, DCPxEP, PCPxEP, PCPxDCP, PCPxTP),
  ``world_size`` and non-contiguous ``rank_table`` validation, idle-DP cohort
  rendezvous, and zero-fallback ``tools/bench_*`` contracts.

[Invariants]
  * Pure in-memory Tier-0 execution with zero subprocesses, zero device
    context, and zero host environment mutations.
  * Locks every integer arithmetic, DCP draft-sharding, non-summed pool pitch,
    topology error propagation, physical fabric rank validation, display-vs-ABI
    separation, and padded expert-slot conservation invariant.

[Timeouts & Locks]
  Deterministic in-memory checks (< 5s); no locks or network I/O.

[Side-Effect Policy]
  Autouse fixture isolates home/config/cache env vars under ``tmp_path``; never
  mutates repository files.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog import cache, hf_config, weights
from mindie_sim.catalog.attention_common import active_dcp_degree
from mindie_sim.catalog.cache_index import index_k_capacity_rows
from mindie_sim.catalog.cache_pools import (
    exact_element_bytes,
    request_pool_page_pitches,
    separate_pool_page_pitches,
    target_mla_page_bytes,
)
from mindie_sim.catalog.hf_value_formats import model_value_formats
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec
from mindie_sim.catalog.weight_layer_terms import layer_weight_terms
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES, load_model
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
)
from tools.export_subgraph_bench_plan import (
    _declared_implementation,
    _display_name,
    _expert_bank_bytes,
    _explicit_ledger,
    _node_fields,
)

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
        "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT",
        "REMOTE_DEV_AUTH_DIR", "UV_CACHE_DIR", "TMPDIR",
    ):
        d = tmp_path / var.lower()
        d.mkdir(exist_ok=True)
        monkeypatch.setenv(var, str(d))


def test_catalog_inventory_reexports_and_relative_manifest_label() -> None:
    assert len(CATALOG) == 39
    assert len(CATALOG_MODELS) == 39
    new_mla = load_model("kimi-k2.6")
    new_hybrid = load_model("qwen3.5-122b-a10b")
    assert new_mla.attention == "mla" and new_mla.num_experts == 384 and new_mla.layers == 61
    assert new_hybrid.num_experts == 256 and new_hybrid.layers == 48
    assert load_model(MODEL_ALIASES["Kimi-K2.6"]) == new_mla
    assert load_model(MODEL_ALIASES["Qwen3.5-122B-A10B"]) == new_hybrid

    # Submodule split re-exports stay identical to top-level catalog modules.
    assert hf_config.model_value_formats is model_value_formats
    assert weights.layer_weight_terms is layer_weight_terms
    assert cache.index_k_capacity_rows is index_k_capacity_rows
    assert cache.separate_pool_page_pitches is separate_pool_page_pitches

    manifest = REPO_ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
    label = _manifest_reference_label(manifest)
    assert label == "tests/fixtures/catalog_reference_manifest.json"
    assert not label.startswith("/")


def test_cache_pools_integer_ceiling_arithmetic_and_independent_pitches() -> None:
    # Exact sub-byte and MX ceiling arithmetic without float truncation or banker's round.
    assert exact_element_bytes(3, "int4") == 2
    assert exact_element_bytes(5, "int4") == 3
    assert exact_element_bytes(33, "mxfp4") == 17 + 2
    assert exact_element_bytes(33, "mxfp8") == 33 + 2

    # Indivisible recurrent/conv state elements must ceil per shard, never floor.
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
    # 13 / 4 -> ceil = 4 FP32 = 16 B; 7 / 4 -> ceil = 2 BF16 = 4 B => 20 B total.
    assert pitches["state_page_bytes"] == 20
    # Target MLA C8 packed row: 512 + ceil(512/128)*4 = 528 B; block=16 => 8448 B.
    assert pitches["target_mla_page_bytes"] == 16 * 528
    assert target_mla_page_bytes(base_model, 16) == 16 * 528
    # Independent pitches are never summed into a single combined_page_bytes pitch.
    assert "combined_page_bytes" not in pitches
    assert pitches["independent_pool_pitches"] == {
        "target_mla": pitches["target_mla_page_bytes"],
        "recurrent_state": pitches["state_page_bytes"],
        "draft": pitches["draft_page_bytes"],
    }
    assert pitches["target_page_unchanged_by_draft"] is True

    # draft_sharded checks actual draft DCP sharding rather than dcp > 1 alone.
    assert pitches["draft_sharded"] is False
    model_mla_draft = replace(base_model, auxiliary=AuxiliarySpec(mtp=sharded_mla_draft))
    assert separate_pool_page_pitches(
        model_mla_draft, par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is True
    # Non-DCP plan (dsa_cp) never claims draft_sharded.
    assert separate_pool_page_pitches(
        model_mla_draft, ParallelSpec(tp=4, cp_plan="dsa_cp"),
        block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is False
    # GQA draft with num_kv_heads=4 on tp=4 (domain = 1 < dcp=2) is not sharded.
    model_bad_gqa = replace(base_model, auxiliary=AuxiliarySpec(mtp=bad_gqa_draft))
    assert separate_pool_page_pitches(
        model_bad_gqa, par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is False
    # GQA draft with num_kv_heads=2 on tp=4 (domain = 2 >= dcp=2) IS sharded.
    model_ok_gqa = replace(base_model, auxiliary=AuxiliarySpec(mtp=ok_gqa_draft))
    assert separate_pool_page_pitches(
        model_ok_gqa, par_dcp2, block_size=16, draft_kind="mtp", dcp=2,
    )["draft_sharded"] is True

    # Odd int4 draft width uses exact ceiling, not round(): 2 * 1 * 5 = 10 -> 5 B;
    # width=5 with int4 (num_kv_heads=1, head_dim=5 on mla kv_cache_elements=5 -> ceil(5/2)=3 B).
    odd_int4_draft = replace(sharded_mla_draft, kv_cache_elements=5)
    model_odd_int4 = replace(
        base_model,
        quant=QuantSpec("bf16", kv_dtype="int4"),
        auxiliary=AuxiliarySpec(mtp=odd_int4_draft),
    )
    odd_report = request_pool_page_pitches(model_odd_int4, par_dcp2, block_size=4, draft_kind="mtp", dcp=2)
    assert odd_report is not None and odd_report["draft_page_bytes"] == 4 * 3


def test_parallel_world_size_five_axes_and_rank_table_contracts() -> None:
    # world_size = external_dp * dp * pp * pcp * tp; DCP and EP never multiply into world_size.
    spec = ParallelSpec.from_mapping({
        "external_data_parallel_size": 2,
        "dp": 2, "pp": 2, "pcp": 2, "tp": 2, "dcp": 4, "ep": 8,
        "world_size": 32,
    })
    assert spec.world_size == 32
    assert spec.to_dict()["external_dp"] == 2
    assert "external_dp" not in ParallelSpec(dp=2, tp=2).to_dict()

    with pytest.raises(ModelingError, match="world_size_conflict"):
        ParallelSpec.from_mapping({"dp": 2, "tp": 2, "dcp": 2, "ep": 4, "world_size": 16})

    # enable_expert_parallel defaults ep = tp * pcp * dp; shard_decode_requests policy accepted.
    auto_ep = ParallelSpec.from_mapping({
        "dp": 2, "tp": 2, "pcp": 2, "enable_expert_parallel": True,
        "pcp_lane_policy": "shard_decode_requests",
    })
    assert auto_ep.ep == 8 and auto_ep.ep_spans_pcp is True
    assert auto_ep.pcp_shard_decode_requests is True

    # Five multi-axis parallel combinations validate cleanly against real catalog models.
    mla_moe = load_model("deepseek-v2-lite")
    dense_gqa = load_model("qwen3-32b")
    for model, kw in (
        (mla_moe, {"dp": 4, "tp": 1, "ep": 4}),                         # DP x EP
        (mla_moe, {"dp": 1, "tp": 4, "dcp": 2, "ep": 4}),               # DCP x EP
        (mla_moe, {"dp": 1, "tp": 2, "pcp": 2, "ep": 4}),               # PCP x EP
        (mla_moe, {"dp": 1, "tp": 2, "pcp": 2, "dcp": 4, "ep": 4}),     # PCP x DCP (full TP*PCP)
        (dense_gqa, {"dp": 1, "tp": 4, "pcp": 2, "ep": 1}),             # PCP x TP
        (dense_gqa, {"dp": 1, "tp": 8, "pcp": 2, "dcp": 2, "ep": 1}),   # GQA PCP x DCP (equal)
    ):
        par = ParallelSpec(**kw)
        par.validate(model)
        assert active_dcp_degree(par, "prefill") == 1
        assert active_dcp_degree(par, "decode") == kw.get("dcp", 1)

    # Invalid PCPxDCP degrees rejected cleanly.
    with pytest.raises(ModelingError, match="dcp_pcp_domain_invalid"):
        ParallelSpec(tp=4, pcp=2, dcp=4, ep=1).validate(mla_moe)
    with pytest.raises(ModelingError, match="gqa_pcp_dcp_requires_equal_degrees"):
        ParallelSpec(tp=2, pcp=2, dcp=4, ep=1).validate(dense_gqa)

    # Explicit non-contiguous rank_table passes when bounded by world_size; invalid tables fail.
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
    logical = {
        0: (RequestView("r0", 2, 64, False, "decode"),),
        1: (RequestView("r1", 1, 64, False, "decode"),),
        2: (),
        3: (),
    }
    synced_full, padded_full = synchronize_moe_allgather_cohort(
        parsed_full, "decode", logical, pad_idle=True, default_context=64,
    )
    assert padded_full is None
    assert synced_full[2][0].q_len == 2 and synced_full[2][0].padding is True
    assert synced_full[3][0].q_len == 2 and synced_full[3][0].padding is True

    # Non-SP MoE AllGather pads all DP replicas to max_dp_rows.
    ag_case = replace(
        parsed_full,
        runtime=replace(
            parsed_full.runtime,
            graph_mode="eager",
            metadata=(*parsed_full.runtime.metadata, ("moe_comm_mode", "allgather")),
        ),
    )
    synced_ag, padded_ag = synchronize_moe_allgather_cohort(
        ag_case, "decode", logical, pad_idle=False, default_context=64,
    )
    assert padded_ag == 4 * 2
    assert all(sum(v.q_len for v in synced_ag[r]) == 2 for r in range(4))

    # Eager decode with nonzero-local-token contract emits 1-token dummy_run on idle DP lanes.
    dummy_case = replace(
        parsed_full,
        runtime=replace(
            parsed_full.runtime,
            graph_mode="eager",
            metadata=(*parsed_full.runtime.metadata, ("eager_dp_dummy_run", True)),
        ),
    )
    synced_dummy, _ = synchronize_moe_allgather_cohort(
        dummy_case, "decode", logical, pad_idle=False, default_context=64,
    )
    assert synced_dummy[2][0].request_id.endswith(":dummy_run")
    assert synced_dummy[2][0].q_len == 1 and synced_dummy[2][0].padding is True

    # Plain receive-only eager alltoall keeps idle DP lanes empty (zero synthetic attention rows).
    plain_case = replace(
        parsed_full,
        runtime=replace(parsed_full.runtime, graph_mode="eager", moe_recipe="alltoall"),
    )
    synced_plain, _ = synchronize_moe_allgather_cohort(
        plain_case, "decode", logical, pad_idle=False, default_context=64,
    )
    assert synced_plain[2] == () and synced_plain[3] == ()

    # End-to-end target_graphs with captured idle DP replicas keeps logical rows at 0.
    graphs = target_graphs(
        parsed_full, phase="decode", tokens=1, context=64,
        sequences=(2, 2, 0, 0), scope="body",
    )
    assert sorted(graphs) == [0, 1, 2, 3]
    idle_attn = next(op for op in graphs[2] if op.family == "attention")
    assert dict(idle_attn.metadata)["graph_active_query_rows"] == 0
    assert dict(idle_attn.metadata)["graph_physical_query_rows"] == 2


def test_bench_plan_builders_zero_fallback_cosets_and_fabric_routes() -> None:
    # Strict builder mapping and structured UnmodeledCapabilityError.
    assert resolve_builder_strict("barrier", "sync", "", ()) == (BOUNDARY_BUILDER, "barrier")
    assert resolve_builder_strict("rmsnorm", "rms_norm", "", ("bf16",)) == ("rms_norm", "aclnnRmsNorm")
    with pytest.raises(UnmodeledCapabilityError) as exc_info:
        resolve_builder_strict("unknown_family", "unknown_recipe", "", ("bf16",), model_id="m", layer_idx=0)
    assert exc_info.value.payload["schema"] == "mindie-unmodeled-structure-hint/1"

    # Collective cosets preserve ALL cosets and never swallow topology exceptions.
    parsed = parse_case(compose_case(
        "deepseek-v2-lite", hardware="Ascend910_9362",
        dp=2, tp=2, ep=4, layers=2,
    ))
    topo = Topology(parsed.machine, parsed.runtime.parallel)
    tp_cosets = resolve_collective_cosets(topo, {"group_axis": "tp"}, stage=0)
    assert tp_cosets == [[0, 1], [2, 3]]
    assert resolve_collective_group_ranks(topo, {"group_axis": "tp"}, stage=0, rank=2) == [2, 3]
    dp_cosets = resolve_collective_cosets(topo, {"group_axis": "dp"}, stage=0)
    assert dp_cosets == [[0, 2], [1, 3]]
    ep_cosets = resolve_collective_cosets(topo, {"group_axis": "ep"}, stage=0)
    assert ep_cosets == [[0, 1, 2, 3]]
    with pytest.raises(BenchTopologyError):
        resolve_collective_cosets(topo, {"group_axis": "bogus_axis"}, stage=0)
    with pytest.raises(BenchTopologyError):
        resolve_collective_group_ranks(topo, {"group_axis": "bogus_axis"}, stage=0)

    # Fabric route ranks reject truncated, oversubscribed, unknown, or unavailable ranks.
    dual_die_hw = {"dies_per_package": 2, "dies_per_node": 16}
    single_die_hw = {"dies_per_package": 1, "dies_per_node": 8}
    assert fabric_route_ranks("intra_package_die", 2, dual_die_hw) == [0, 1]
    assert fabric_route_ranks("intra_node_hccs", 4, dual_die_hw) == [0, 2, 4, 6]
    assert fabric_route_ranks("inter_node_roce", 8, dual_die_hw) == [0, 1, 2, 3, 16, 17, 18, 19]
    with pytest.raises(BenchGeometryError, match="fabric_intra_package_unrealizable"):
        fabric_route_ranks("intra_package_die", 2, single_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_intra_package_unrealizable"):
        fabric_route_ranks("intra_package_die", 4, dual_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_intra_node_exceeds_node"):
        fabric_route_ranks("intra_node_hccs", 16, single_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_route_unknown"):
        fabric_route_ranks("mystery_link", 2, dual_die_hw)
    with pytest.raises(BenchGeometryError, match="fabric_route_ranks_unavailable"):
        fabric_route_ranks("intra_node_hccs", 4, dual_die_hw, available_ranks=[0, 2, 4])


def test_export_plan_abi_separation_padded_expert_slots_and_packet_scrubber() -> None:
    # Display name is strictly separated from declared implementation and certified upstream ABI.
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

    # Non-divisible E % EP != 0 expert banks and ledgers use expert_slot_plan padded slots.
    padded_model = SimpleNamespace(
        num_experts=10, moe_intermediate_size=256, hidden_size=512,
        layer_plans=(), quant=QuantSpec("bf16"),
    )
    bank = _expert_bank_bytes(padded_model, {"ep": 4})
    assert bank is not None
    assert bank["base_local_experts"] == 3 and bank["padded_expert_slots"] == 2
    expert_rows = [2] * 10
    topk_ids = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]] * 2
    ledger = _explicit_ledger(expert_rows, topk_ids, ep=4)
    assert ledger["padded_expert_slots"] == 2
    assert sum(ledger["rank_assignment_rows"]) == sum(expert_rows) == 20
    assert ledger["local_weight_expert_ids"] == [0, 1, 2]

    # Bench packets provenance scrubber & explicit split enforcement.
    with pytest.raises(PacketError, match="private_provenance_home_path"):
        assert_public_provenance({"note": "/mnt/private/run"})
    with pytest.raises(PacketError, match="private_provenance_ipv4"):
        assert_public_provenance({"src": "10.0.0.1"})
    with pytest.raises(PacketError, match="observation_requires_explicit_split"):
        build_packets([{"status": "ok", "source": "capture://run1/s0", "duration_ms": 1.0}])

    # End-to-end structure export attaches group_cosets alongside group_ranks for collectives.
    case_spec = parse_case(compose_case(
        "qwen3-32b", hardware="Ascend910_9362", tp=4, pcp=2, ep=1, layers=2,
    ))
    rep = export_structure(case_spec, phase="prefill", tokens=64, context=0, sequences=1)
    assert any((n.get("operator") or {}).get("recipe") == "pcp_kv_gather" for n in rep["nodes"])
