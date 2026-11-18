"""Regression invariants for the v0.9.19 Track A audit fixes.

[Role] Exercise hardware derivation, capacity/catalog parity, composition and
  observed-contract closure through the public case/graph paths.
[Invariants] Every expectation is an independent byte, row, stage or identity
  conservation statement; no timing tolerance is widened.
[Timeouts & Locks] Bounded CPU-only construction; no device contexts.
[Side-Effect Policy] Isolated pytest state; no filesystem or NPU mutation.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case, target_graphs
from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.cache import layer_cache_components, request_cache_breakdown
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.catalog.cache_ownership import batch_cache_footprint
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import (
    cache_row_bytes_by_layer, engram_table_terms, stage_weight_terms,
)
from mindie_sim.cohort_comparison import compare_cohorts
from mindie_sim.compose import assemble
from mindie_sim.hardware_derivation import derive_hardware_sku
from mindie_sim.observed_contracts.attention_contracts import (
    observed_attention_contract,
)
from mindie_sim.observed_contracts.tensor_contracts import (
    observed_contract, observed_contract_diagnostic,
)
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity_features import batch_cache_bytes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_structure
from mindie_sim.structure_implementations import (
    native_candidates, native_implementation,
)
from mindie_sim.structure_physical import node_physical
from mindie_sim.structure_selection import select_structure_targets
from mindie_sim.structure_subgraphs import subgraph_contract


def _case(model, **overrides):
    options = {
        "hardware": "Ascend950DT_9582", "graph_capture_mode": "none",
        "prompt_tokens": 128, "output_tokens": 2,
    }
    options.update(overrides)
    return parse_case(compose_case(model, **options))


def _observed_row(kind, inputs, outputs):
    return SimpleNamespace(
        kind=kind,
        input_shapes=tuple(slot[0] for slot in inputs),
        input_dtypes=tuple(slot[1] for slot in inputs),
        input_formats=tuple(slot[2] for slot in inputs),
        output_shapes=tuple(slot[0] for slot in outputs),
        output_dtypes=tuple(slot[1] for slot in outputs),
        output_formats=tuple(slot[2] for slot in outputs),
    )


def test_ascend950pr_958b_cacheline_service_matches_hbm_supply():
    device = profiles()["Ascend950PR_958b"]["device"]
    service = device["compute_arch"]["instruction_services"]["gm_hbm_cacheline_128b"]
    owners = device["resource_units"]["vector"]
    hbm = device["rates"]["hbm"]
    assert service["per_owner_rate"] == pytest.approx(hbm / (128 * owners))
    # A single AIV servicing 1,025 lines takes the declared HBM share.
    assert 1025 / service["per_owner_rate"] * 1e9 == pytest.approx(5997.7, abs=0.2)


def test_derived_aic_count_scales_cacheline_and_service_modes():
    derived = derive_hardware_sku(
        "Ascend950PR_958b", sku="derived-958b-28", aic_count=28)
    device = derived.dies[0]
    rates, units, modes = dict(device.rates), dict(device.resource_units), dict(
        device.service_modes)
    assert units["vector"] == 56
    assert units["gm_hbm_cacheline_128b"] == 56
    assert rates["gm_hbm_cacheline_128b"] == pytest.approx(rates["hbm"] / 128)
    assert modes["mte1_l0a"][1] == pytest.approx(13.5168e12 * 28 / 32)
    assert modes["ub_copy"][1] == pytest.approx(27.0336e12 * 56 / 64)
    cacheline = next(
        service for service in device.compute_arch.instruction_services
        if service.name == "gm_hbm_cacheline_128b")
    assert cacheline.per_owner_rate * units["vector"] == pytest.approx(
        rates["hbm"] / 128)
    scaled = derive_hardware_sku(
        "Ascend950DT_9582", sku="derived-9582-28", aic_count=28,
        hbm_peak_bytes_per_s=2e12, hbm_efficiency=.8)
    scaled_device = scaled.dies[0]
    scaled_units = dict(scaled_device.resource_units)
    scaled_rates = dict(scaled_device.rates)
    scaled_cacheline = next(
        service for service in scaled_device.compute_arch.instruction_services
        if service.name == "gm_hbm_cacheline_128b")
    assert scaled_rates["hbm"] == pytest.approx(1.6e12)
    assert scaled_units["gm_hbm_cacheline_128b"] == 56
    assert scaled_rates["gm_hbm_cacheline_128b"] == pytest.approx(
        scaled_rates["hbm"] / 128)
    assert scaled_cacheline.per_owner_rate * scaled_units["vector"] == pytest.approx(
        scaled_rates["hbm"] / 128)
    assert dict(scaled_device.service_modes)["mte1_l0a"][1] == pytest.approx(
        11.8272e12)


def test_wide_node_uses_one_port_unless_semantics_is_explicit():
    options = dict(
        hardware="Ascend950DT_9581", tp=4, dp=8, ep=4, num_devices=32,
        ranks_per_node=16, superpod_size=32, intra_node_bw_gbps=800,
        intra_node_bw_util=.85, intra_node_latency_us=12,
        prompt_tokens=8, output_tokens=2,
    )
    manual = _case("deepseek-v3", **options)
    preset = _case("deepseek-v3", topology_preset="a5_pod", **options)
    for case in (manual, preset):
        links = [link for link in case.machine.links
                 if link.channel == "d2d_intra_node"]
        assert links
        assert all(link.semantics == "per_port" for link in links)
        assert all(link.capacity_bytes_per_ms == pytest.approx(680e6)
                   for link in links)


@pytest.mark.parametrize("model", ["deepseek-v4.1-flash", "step-5-preview"])
def test_windowed_mtp_allocation_report_conserves_cache_bytes(model):
    case = _case(model, tp=4, ep=4, layers=4, mtp_proposal_tokens=3)
    memory = Memory(case)
    report = memory.allocation_report(8192)
    assert report["cache"]["allocated"]["total_bytes_per_stage_rank"] == list(
        memory.cache_bytes(8192))


def test_engram_quant_override_moves_table_storage_mode():
    case = _case(
        "deepseek-v4.1-flash", tp=8, ep=8, num_devices=8,
        ranks_per_node=8, engram_cpu_offload=False,
        quant={
            "name": "w8a8", "weight_format": "int8",
            "activation_format": "int8", "expert_weight_format": "int8",
            "kv_dtype": "bf16", "engram_table_mode": "device_int8_fp32",
        },
    )
    assert case.model.engram.storage_mode == "device_int8_fp32"
    ledger = sum(
        term_bytes(term, case.model.quant, case.runtime.parallel,
                   case.machine.ranks_per_node)
        for term in engram_table_terms(case.model))
    assert ledger == pytest.approx(case.model.engram.table_storage_bytes(
        node_rows=case.machine.ranks_per_node))


def test_draft_layer_plan_indexer_and_prepared_router_are_resident():
    glm = _case("glm-5.3-flash", tp=8, ep=8, layers=4,
                mtp_proposal_tokens=1)
    draft = glm.model.auxiliary.mtp
    terms = draft_terms(draft, glm.model)
    indexer = [term for term in terms if term.name.startswith("indexer_")]
    assert indexer
    assert sum(term_bytes(term, glm.model.quant, glm.runtime.parallel,
                          glm.machine.ranks_per_node)
               for term in indexer) == pytest.approx(14_945_280)
    assert any(term.name == "indexer_wq" for term in indexer)

    pro = _case("deepseek-v4-pro", tp=8, ep=8, layers=4,
                mtp_proposal_tokens=2, raw_eager=True)
    pro_draft = pro.model.auxiliary.mtp or pro.model.auxiliary.dspark
    prepared = [term for term in draft_terms(pro_draft, pro.model)
                if term.name == "draft_router_precast"]
    assert len(prepared) == 3
    assert sum(term_bytes(term, pro.model.quant, pro.runtime.parallel,
                          pro.machine.ranks_per_node)
               for term in prepared) == pytest.approx(33_030_144)
    staged = [term for group in stage_weight_terms(
        pro.model, pro.runtime.parallel, include_draft=True, draft_kind="dspark")
        for term in group]
    assert any(term.name == "draft_router_precast" for term in staged)
    assert not any(term.category.startswith("target_alias") for term in staged)


def test_int8_packed_mla_cache_is_phase_independent_and_uses_row_format():
    case = _case(
        "glm-5.3-flash", tp=8, ep=8, layers=4,
        quant={
            "name": "w8a8", "weight_format": "int8",
            "activation_format": "int8", "kv_dtype": "int8",
        },
    )
    assert cache_format(case.model, "mla_kv").row_bytes(512) == 528
    assert cache_row_bytes_by_layer(case.model)[-1] == (3, 528)
    for phase in ("prefill", "decode"):
        effective, _ = effective_body_declaration(case, phase)
        assert cache_format(effective.model, "mla_kv").row_bytes(512) == 528
        assert cache_row_bytes_by_layer(effective.model)[-1] == (3, 528)


def test_gemma_shared_projection_still_stores_two_cache_planes():
    case = _case("gemma-4-31b", tp=1, ep=1, layers=6)
    plan = next(plan for plan in case.model.layer_plans if plan.index == 5)
    expected = CacheFormat(case.model.kv_dtype).row_bytes(plan.cache_row_elements)
    assert cache_row_bytes_by_layer(case.model)[5][1] == expected == 8192


def test_pp_partition_is_shared_by_capacity_and_catalog_paths():
    case = _case("deepseek-v3", tp=8, ep=8, layers=5, pp=2,
                 pp_layer_partition=(1, 4), quant="w8a8")
    memory = Memory(case)
    expected = list(memory.cache_bytes(128))
    breakdown = request_cache_breakdown(
        case.model, case.runtime.parallel, 128, runtime=case.runtime)
    assert breakdown["stage_bytes"] == pytest.approx(expected)
    footprint = batch_cache_bytes(
        case, sequences=1, context=127, tokens_per_seq=1, phase="decode",
        padded=True)
    assert list(footprint.total_bytes) == pytest.approx(expected)


def test_shared_prefix_footprint_charges_longest_pages_once():
    case = _case("kimi-k3.1", hardware="a5", tp=8, ep=8, layers=4,
                 block_size=1536, max_model_len=20000,
                 prefix_cache_mode="shared")
    def view(index, prefix):
        return RequestView(
            f"r{index}", 1, 8000, True, "decode", prefix_tokens=prefix,
            prefix_cache_key="doc")
    short, long = view(0, 1536), view(1, 3072)
    short_first = batch_cache_footprint(
        case, requests=(short, long), replica=0, phase="decode", padded=True)
    long_first = batch_cache_footprint(
        case, requests=(long, short), replica=0, phase="decode", padded=True)
    assert short_first.total == long_first.total
    assert short_first.shared_once_bytes == long_first.shared_once_bytes
    assert short_first.shared_once_total > 0


def test_fp16_preset_is_built_from_named_dtype_fields():
    case = parse_case(compose_case(
        "gemma-4-31b", hardware="Ascend950DT_9582", layers=4, tp=4, ep=1,
        quant="fp16", prompt_tokens=8, output_tokens=1,
        graph_capture_mode="none"))
    assert case.model.quant.weight_format == "fp16"
    assert case.model.quant.kv_dtype == "fp16"
    assert case.model.quant.index_dtype == "fp16"
    graph = target_graphs(case, phase="decode", tokens=1, context=8, scope="model")
    assert graph[0]


def test_v41_sliced_layers_keep_packed_rows_and_mixed_quant_decode():
    for layers in (2, 4):
        case = _case("deepseek-v4.1-flash", tp=1, dp=1, ep=1, layers=layers)
        first = case.model.layer_plans[0]
        assert cache_format(case.model, "window").row_bytes(
            first.cache_row_elements) == 544
        graph = target_graphs(
            case, phase="decode", tokens=1, context=2048, scope="model")
        layer0 = [op.recipe for op in graph[0]
                  if op.layer == 0 and op.family == "attention"]
        assert any("mixed_quant_sparse" in recipe for recipe in layer0)


def test_exclusions_remove_profile_bindings_without_activating_a_track():
    kimi = _case("kimi-k3.1", tp=8, ep=8, layers=4,
                 exclude_optimizations=("attn_res_prefix_fusion",))
    lowerings = dict(kimi.runtime.kernels.get("lowerings", {}))
    assert "attention_residual" not in lowerings
    graph = target_graphs(kimi, phase="decode", tokens=1, context=8, scope="model")
    assert not any("attn_res" in op.op_id for op in graph[0])

    bare = _case("deepseek-v3", tp=8, ep=8, layers=4, va_recommended=False,
                 exclude_optimizations=("mla_prolog_fusion",))
    assert bare.runtime.graph_capture_mode == "none"
    assert not dict(bare.runtime.kernels)


def test_model_scope_uses_the_service_speculative_iteration_plan():
    case = _case("glm-5.3-flash", tp=1, dp=1, ep=1, layers=4,
                 mtp_proposal_tokens=5, mtp_acceptance_length=6,
                 prompt_tokens=4, output_tokens=8, raw_eager=True)
    model = assemble(
        case, scope="model", phase="decode", iterations=1, tokens=6, context=4)
    service = assemble(
        case, scope="service", phase="decode", iterations=1, tokens=6, context=4)
    assert len(model["steps"]) == 1
    assert len(service["steps"]) == 1
    assert model["steps"][0]["valid_tokens"] == 6
    assert service["steps"][0]["valid_tokens"] == 6
    assert model["steps"][0]["program_assembly"] == "model"

    partial = _case("glm-5.3-flash", tp=1, dp=1, ep=1, layers=4,
                    mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
                    prompt_tokens=4, output_tokens=8, raw_eager=True)
    partial_model = assemble(
        partial, scope="model", phase="decode", iterations=2, tokens=4, context=4)
    partial_service = assemble(
        partial, scope="service", phase="decode", iterations=2, tokens=4, context=4)
    assert ([step["valid_tokens"] for step in partial_model["steps"]]
            == [step["valid_tokens"] for step in partial_service["steps"]]
            == [6, 5, 2])
    assert [step["context_tokens"]["req-0"]
            for step in partial_model["steps"]] == [4, 7, 10]
    assert all(step["d2h_bytes"] == 0.0 and not step["emitted_tokens"]
               for step in partial_model["steps"])


def test_observed_mqsmla_spellings_share_the_active_contract():
    query = ((1, 64, 512), "bf16", "ND")
    empty = ((), None, None)
    inputs = [
        query, empty, ((16, 1, 1, 544), "uint8", "ND"),
        ((1, 16), "int32", "ND"), empty, empty, empty,
    ]
    outputs = [query, empty]
    v41 = _case("deepseek-v4.1-flash", tp=1, dp=1, ep=1, layers=2)
    exported = export_structure(
        v41, phase="decode", tokens=1, context=2048, scope="model")
    declared = [tuple(node["contract"]) for node in exported["nodes"]
                if node.get("recipe") == "aclnn_mixed_quant_sparse_flash_mla"]
    assert len(declared) == 2
    for kind, expected in (
        ("MqsmlaKernel___call__", "MqsmlaKernel___call__"),
        ("MqsmlaKernel", "MqsmlaKernel___call__"),
        ("aclnnMixedQuantSparseFlashMla", "aclnnMixedQuantSparseFlashMla"),
        ("MixedQuantSparseFlashMla", "aclnnMixedQuantSparseFlashMla"),
    ):
        contract = observed_attention_contract(_observed_row(kind, inputs, outputs))
        assert contract is not None
        assert contract[0] == "sparse_attention_active"
        assert contract[1] == expected
        assert contract[2][0][0] == query[0]
        if expected == "aclnnMixedQuantSparseFlashMla":
            assert list(contract[:2]) == list(declared[0][:2])


def test_compressor_contract_omits_gate_and_ape_for_c1():
    view = {
        "tokens": 8, "hidden": 5120, "width": 512, "coff": 2,
        "dtype": "bf16", "weight_dtype": "bf16", "ape_dtype": "fp32",
        "norm_dtype": "bf16", "state_dtype": "fp32", "rope_width": 64,
        "rope_dtype": "bf16", "ratio": 1,
    }
    contract = subgraph_contract({
        "implementation": {}, "coverage": {}, "tensor_view": view,
        "role": "compressor",
    })
    assert len(contract[3]) == 2
    gated = subgraph_contract({
        "implementation": {}, "coverage": {}, "tensor_view": dict(view, ratio=4),
        "role": "compressor",
    })
    assert len(gated[3]) == 4


def test_v4_prolog_exports_and_decodes_the_v4_native_identity():
    view = {
        "tokens": 2, "hidden": 7168, "heads": 16, "query_rank": 1536,
        "latent_rank": 512, "nope_width": 128, "rope_width": 64,
        "projection_weight_dtypes": ("fp8_e4m3", "fp8_e4m3", "fp8_e4m3"),
        "norm_dtype": "bf16", "cache_dtype": "int8",
        "query_norm_output": True, "query_norm_dtype": "fp8_e4m3",
        "token_dtype": "fp8_e4m3", "absorbed_weight_dtype": "bf16",
        "packed_cache": {"row_bytes": 656},
    }
    for declared_kind in ("aclnnMlaPrologV4WeightNz", "MlaPrologV4WeightNz",
                          "MlaPrologV4"):
        declared = subgraph_contract({
            "implementation": {"native_kind": declared_kind},
            "coverage": {}, "tensor_view": view, "role": "mla_prolog",
        })
        assert declared[0] == "mla_prolog_active"
        assert declared[1] == "aclnnMlaPrologV4WeightNz"

    empty_bf16 = ((), "bf16", "ND")
    empty_none = ((), None, None)
    v4_inputs = [
        ((2, 7168), "fp8_e4m3", "ND"),
        ((7168, 1536), "fp8_e4m3", "FRACTAL_NZ"),
        ((1536, 3072), "fp8_e4m3", "FRACTAL_NZ"),
        ((16, 128, 512), "bf16", "ND"),
        ((7168, 576), "fp8_e4m3", "FRACTAL_NZ"),
        ((1536,), "bf16", "ND"),
        ((512,), "bf16", "ND"),
        empty_none, empty_none,
        ((16, 128, 1, 656), "int8", "ND"),
        empty_bf16,
    ] + [empty_none] * 10
    v4_outputs = [
        ((2, 16, 512), "bf16", "ND"),
        ((2, 16, 64), "bf16", "ND"),
        ((16, 128, 1, 656), "int8", "ND"),
        empty_bf16, empty_none,
        ((2, 1536), "fp8_e4m3", "ND"),
        ((2, 1), "fp32", "ND"),
    ]
    for observed_kind in ("aclnnMlaPrologV4WeightNz", "MlaPrologV4WeightNz"):
        row = _observed_row(observed_kind, v4_inputs, v4_outputs)
        assert observed_contract(row) == declared
        assert observed_contract_diagnostic(row)["reason"] == (
            "observed_tensor_contract_available")
    assert observed_contract(_observed_row("MlaPrologV3", v4_inputs, v4_outputs)) is None

    v3_inputs = [
        ((2, 7168), "int8", "ND"),
        ((7168, 1536), "int8", "FRACTAL_NZ"),
        ((1536, 3072), "int8", "FRACTAL_NZ"),
        *v4_inputs[3:4],
        ((7168, 576), "int8", "FRACTAL_NZ"),
        *v4_inputs[5:],
    ]
    v3_outputs = [*v4_outputs[:5], ((2, 1536), "int8", "ND"), v4_outputs[6]]
    v3_row = _observed_row("MlaPrologV3", v3_inputs, v3_outputs)
    v3_contract = observed_contract(v3_row)
    assert v3_contract is not None and v3_contract[:2] == (
        "mla_prolog_active", "MlaPrologV3")
    assert observed_contract(
        _observed_row("aclnnMlaPrologV4WeightNz", v3_inputs, v3_outputs)) is None


def test_explicit_cache_epilog_native_kind_is_a_candidate():
    op = {
        "op_id": "cache", "family": "cache_store", "recipe": "indexer_cache",
        "source_boundary": None, "dtype": "bf16", "local_shape": (1, 128),
        "metadata": {"native_kind": "aclnnIndexerQuantCache"},
    }
    implementation = native_implementation(op, None)
    assert implementation["native_kinds"] == ("aclnnIndexerQuantCache",)
    candidates, execution, gaps = native_candidates(op, None)
    assert [row["native"] for row in candidates] == ["aclnnIndexerQuantCache"]
    assert candidates[0]["selected"] is False
    assert execution["selection_state"] == "single_compatible_unselected"
    assert not gaps


def test_selective_structure_hash_includes_onboarding_hints():
    case = _case("qwen3.5-35b-a3b", hardware="Ascend910_9392", layers=2,
                 prompt_tokens=8, output_tokens=2)
    plans = tuple(replace(plan, attention="unmodeled",
                          architecture="custom_attention")
                  for plan in case.model.layer_plans)
    case = replace(case, model=replace(
        case.model, attention="unmodeled", layer_plans=plans))
    full = export_structure(
        case, phase="decode", tokens=1, context=8, scope="model")
    hints = full.get("unmodeled_structure_onboarding_hints")
    assert hints
    targets = [row["node_id"] for row in full["nodes"][:4]]
    selected = select_structure_targets(
        case, targets, phase="decode", tokens=1, context=8, scope="model")
    assert selected["graph_id"] == full["graph_id"]


def test_cohort_comparison_regenerates_a_projected_manifest():
    case = _case("qwen3.5-35b-a3b", hardware="Ascend910_9392", layers=2,
                 dp=2, ep=2, tp=1, prompt_tokens=8, output_tokens=2)
    full = export_structure(
        case, phase="decode", tokens=1, context=8, sequences=2, scope="model",
        replicas=[0])
    report = {
        "schema_version": "mindie-profile-structure-mapping/1",
        "declared_execution": {
            "phase": "decode", "tokens": 1, "context": 8, "sequences": 2,
            "scope": "model", "replica_projection": {"selected": [0]},
        },
        "graph_id": full["graph_id"],
        "windows": [],
    }
    result = compare_cohorts(case, report)
    assert result["graph_id"] == full["graph_id"]


def test_draft_prefixed_occurrences_share_a_physical_signature():
    common = dict(
        family="grouped_gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=(), semantic_shape=(1, 2), local_shape=(1, 2),
        padded_shape=(1, 2), launch_shape=(1, 2), flops=1, hbm_bytes=1,
        mte2_bytes=0, mte3_bytes=0, vector_ops=0, link_bytes=0,
        link_scope="", recipe="grouped_gate_up", coverage=None,
        dtype="bf16", layout="ND",
    )
    first = OpSpec(
        op_id="first", metadata=(("producer", "draft0.s0.l0.expert_act"),),
        **common)
    second = OpSpec(
        op_id="second", metadata=(("producer", "draft0.s0.l1.expert_act"),),
        **common)
    assert (node_physical(first)["physical_signature"]
            == node_physical(second)["physical_signature"])


def test_moe_init_routing_v3_projects_the_mxfp8_routing_contract():
    inputs = [((4, 4096), "bf16", "ND"), ((4, 8), "int32", "ND")]
    outputs = [
        ((32, 4096), "fp8_e4m3", "ND"), ((32,), "int32", "ND"),
        ((288,), "int64", "ND"), ((32, 128), "fp8_e8m0", "ND"),
    ]
    row = _observed_row("MoeInitRoutingV3", inputs, outputs)
    contract = observed_contract(row)
    assert contract == (
        "moe_init_routing_mxfp8_active", "MoeInitRoutingV3",
        (inputs[0], inputs[1]), tuple(outputs))
    assert observed_contract_diagnostic(row)["reason"] == (
        "observed_tensor_contract_available")


def test_shard1_windowed_mtp_request_cache_breakdown_conserves_bytes():
    case = _case("deepseek-v4.1-flash", tp=4, ep=4, layers=4,
                 mtp_proposal_tokens=3, mtp_acceptance_length=4)
    memory = Memory(case)
    breakdown = request_cache_breakdown(
        case.model, case.runtime.parallel, 8192, draft_kind="mtp",
        runtime=case.runtime)
    batch = batch_cache_bytes(
        case, sequences=1, context=8191, tokens_per_seq=1, phase="decode",
        padded=True)
    assert breakdown["by_component"]["draft_kv"] == pytest.approx(208_896.0)
    assert breakdown["total_bytes"] == pytest.approx(2_207_744.0)
    assert memory.cache_bytes(8192) == pytest.approx((2_207_744.0,))
    assert tuple(batch.total_bytes) == pytest.approx((2_207_744.0,))


def test_shard1_dense_mtp_index_cache_aligns_memory_and_batch_ledgers():
    case = _case("glm-5.3-flash", tp=4, ep=4, layers=4,
                 mtp_proposal_tokens=3, mtp_acceptance_length=4)
    memory = Memory(case)
    report = memory.allocation_report(8192)
    breakdown = request_cache_breakdown(
        case.model, case.runtime.parallel, 8192, draft_kind="mtp",
        runtime=case.runtime)
    batch = batch_cache_bytes(
        case, sequences=1, context=8188, tokens_per_seq=4, phase="decode",
        padded=True)
    assert memory.cache_bytes(8192) == pytest.approx((30_610_432.0,))
    assert report["cache"]["allocated"]["total_bytes_per_stage_rank"] == pytest.approx(
        [30_610_432.0])
    assert breakdown["total_bytes"] == pytest.approx(30_610_432.0)
    assert tuple(batch.total_bytes) == pytest.approx((30_610_432.0,))


def test_shard1_v41_window_layers_report_packed_window_row_bytes():
    case = _case("deepseek-v4.1-flash", tp=4, ep=4, layers=4)
    rows = dict(cache_row_bytes_by_layer(case.model))
    assert rows[0] == 544
    assert rows[1] == 544


@pytest.mark.parametrize(("scale_dtype", "expected_row_bytes"), [
    ("bf16", 608),
    ("uint8", 584),
])
def test_shard1_mqsmla_mode1_and_mode2_align_capacity_structure_and_observed(
    scale_dtype, expected_row_bytes,
):
    raw = compose_case(
        "deepseek-v4.1-flash", hardware="Ascend950PR_958b", tp=4, dp=1, ep=4,
        layers=4, prompt_tokens=128, output_tokens=2, graph_capture_mode="none",
    )
    region = {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": scale_dtype}
    meta = raw["model"].setdefault("metadata", {})
    reps = dict(meta.get("cache_representations", {}))
    reps["window"] = dict(region)
    reps["compressed_kv"] = dict(region)
    meta["cache_representations"] = reps
    case = parse_case(raw)

    assert dict(cache_row_bytes_by_layer(case.model))[0] == expected_row_bytes
    layer0 = layer_cache_components(
        case.model, case.model.layer_plans[0], case.runtime.parallel, 128,
        runtime=case.runtime)
    assert layer0["window"] == pytest.approx(128 * expected_row_bytes)
    Memory(case).allocation_report(512)

    exported = export_structure(
        case, phase="decode", tokens=1, context=512, scope="model")
    mqsmla_nodes = [node for node in exported["nodes"]
                    if node.get("recipe") == "aclnn_mixed_quant_sparse_flash_mla"]
    assert len(mqsmla_nodes) == 4
    assert mqsmla_nodes[2]["contract"][3] == [
        [[1, expected_row_bytes], "uint8", "ND"],
        [[1, expected_row_bytes], "uint8", "ND"],
    ]

    query = ((1, 16, 512), "bf16", "ND")
    empty = ((), None, None)
    obs_inputs = [
        query, empty, ((16, 1, 1, expected_row_bytes), "uint8", "ND"),
        ((1, 16), "int32", "ND"), empty, empty, empty,
        ((16, 1, 1, expected_row_bytes), "uint8", "ND"),
        ((1, 16), "int32", "ND"), empty,
    ]
    obs = observed_attention_contract(
        _observed_row("aclnnMixedQuantSparseFlashMla", obs_inputs, [query, empty]))
    assert obs is not None
    assert obs[3] == (
        ((1, expected_row_bytes), "uint8", "ND"),
        ((1, expected_row_bytes), "uint8", "ND"),
    )
    bad_inputs = [*obs_inputs[:7], ((16, 1, 1, 320), "uint8", "ND"), *obs_inputs[8:]]
    assert observed_attention_contract(
        _observed_row("aclnnMixedQuantSparseFlashMla", bad_inputs, [query, empty])
    ) is None


def test_shard1_multi_request_speculative_assemble_matches_service_steps():
    case = _case(
        "glm-5.3-flash", tp=1, dp=1, ep=1, layers=4,
        mtp_proposal_tokens=2, mtp_acceptance_length=1.5,
        max_num_batched_tokens=2,
        prompt_tokens=4, output_tokens=3, raw_eager=True,
    )
    requests = [
        {"request_id": "r0", "prompt_tokens": 4, "output_tokens": 3},
        {"request_id": "r1", "prompt_tokens": 4, "output_tokens": 3},
    ]
    service = assemble(
        case, scope="service", phase="decode", iterations=3, tokens=1,
        context=4, requests=requests)
    model = assemble(
        case, scope="model", phase="decode", iterations=3, tokens=1,
        context=4, requests=requests)
    assert ([step["valid_tokens"] for step in model["steps"]]
            == [step["valid_tokens"] for step in service["steps"]]
            == [2, 2, 2, 2, 1])


def test_shard1_subgraph_boundary_algebra_respects_producer_and_consumer_caps():
    case = _case("deepseek-v3.2", tp=8, ep=8, layers=4, quant="w8a8")
    exported = export_structure(
        case, phase="prefill", tokens=128, context=0, scope="model")
    prolog = next(
        group for group in exported["subgraphs"]
        if group["subgraph_id"].endswith("l0.mla_prolog"))
    nodes = {node["node_id"]: node for node in exported["nodes"]}
    members = tuple(prolog["node_ids"])
    ext_out = set(prolog["external_outputs"])
    internal_producers = {
        m for m in members
        if m not in ext_out
        and any(m in nodes[c]["deps"] for c in members)
    }
    writes = sum(float(nodes[m]["physical_work"]["mte3_bytes"])
                 for m in internal_producers)
    max_reads = sum(
        min(
            float(nodes[c]["physical_work"]["mte2_bytes"]),
            sum(
                min(float(nodes[p]["physical_work"]["mte3_bytes"]),
                    float(nodes[c]["physical_work"]["mte2_bytes"]))
                for p in nodes[c]["deps"] if p in internal_producers
            ),
        )
        for c in members
    )
    breakdown = prolog["math_summary"]["boundary_breakdown_bytes"]
    assert max_reads == pytest.approx(1_983_488.0)
    assert breakdown["eliminated_internal_roundtrip_bytes"] == pytest.approx(
        writes + max_reads)
    assert breakdown["minimal_boundary_hbm_bytes"] == pytest.approx(
        breakdown["unfused_member_hbm_bytes"] - writes - max_reads)
