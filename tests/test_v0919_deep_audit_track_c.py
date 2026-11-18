"""v0.9.19 Track C deep-audit regressions: DES, fusion causality and attention.

[Role] Bind each fixed contract to an independent structural or arithmetic
  invariant: Python/C11 arbitration parity, tile-zero stream ordering, catalog
  cohort propagation, mHC/QKV/KDA producer dependencies, selected native-group
  isolation, packed-weight conservation, MTP selection geometry, PCP ownership
  and shared-prefix byte unions.
[Invariants] Expected values are derived from declared work, intervals and
  formats; no test reads a measured duration or a fitted multiplier.
[Timeouts & Locks] Finite in-memory engines; no device, network or file locks.
[Side-Effect Policy] Writes only pytest temporary cache directories.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mindie_sim.compose import compose_case  # noqa: E402
from mindie_sim.prediction.api import parse_case  # noqa: E402
from mindie_sim.prediction.implementations import (  # noqa: E402
    input_ports, produced_values,
)
from mindie_sim.prediction.operator_analysis import (  # noqa: E402
    target_graphs, target_operators,
)
from mindie_sim.prediction.program import (  # noqa: E402
    KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate, Stage,
    coalesce_execution_program,
)
from mindie_sim.prediction.program_catalog import (  # noqa: E402
    compile_pipeline_template, compile_program, submit_program,
    submit_program_catalog,
)
from mindie_sim.prediction.runtime import accel_build  # noqa: E402
from mindie_sim.prediction.runtime.contracts import CatalogGeneration  # noqa: E402
from mindie_sim.prediction.runtime.engine import Engine, Resource  # noqa: E402


GENERATION = CatalogGeneration(
    registry_generation="track-c", implementation_generation="deep-audit")
CACHEABLE = {}


def _cached(key, factory):
    if key not in CACHEABLE:
        CACHEABLE[key] = factory()
    return CACHEABLE[key]


def _case(model, **kwargs):
    defaults = dict(hardware="a3", tp=8, ep=8, num_devices=8, layers=2)
    defaults.update(kwargs)
    return _cached((model, repr(sorted(defaults.items()))),
                   lambda: parse_case(compose_case(model, **defaults)))


def _ops(case, **kwargs):
    return target_operators(case, scope="body", **kwargs)


def _with_passes(case, *passes, kernels=None):
    current = tuple(name for name in case.runtime.lowering_passes
                    if not name.startswith("mhc_boundary")
                    and name != "kda_in_kernel_l2norm")
    runtime = replace(case.runtime, lowering_passes=(*current, *passes))
    if kernels is not None:
        runtime = replace(runtime, kernels={
            **dict(runtime.kernels),
            "lowerings": {**dict(runtime.kernels.get("lowerings", {})),
                          **kernels}})
    return replace(case, runtime=runtime)


def _native_engine(tmp_path, resources):
    from mindie_sim.prediction.runtime import accel_bridge
    previous = os.environ.get("MINDIE_CACHE_HOME")
    os.environ["MINDIE_CACHE_HOME"] = str(tmp_path / "accel-cache")
    try:
        accel_build.reset_build_memo()
        status = accel_build.build_accel()
        if not status["ok"]:
            pytest.skip("native accelerator unavailable: "
                        + str(status.get("reason") or status.get("detail")))
        accel_bridge.load_library(status)
        return Engine(resources, backend="c")
    finally:
        if previous is None:
            os.environ.pop("MINDIE_CACHE_HOME", None)
        else:
            os.environ["MINDIE_CACHE_HOME"] = previous
        accel_build.reset_build_memo()


# ---------------------------------------------------------------------------
# Fix 1: core DES / C11 parity, tile-zero ordering and catalog cohort


def _placed_fixture(backend, tmp_path=None, peer_count=8):
    resources = [Resource("cube", 4.0, units=4), Resource("hbm", 4.0, units=4)]
    engine = (_native_engine(tmp_path, resources) if backend == "c"
              else Engine(resources, backend="python"))
    pinned = engine.submit(
        "pinned", {"cube": 4.0, "hbm": 4.0}, coupled=True,
        placement={"cube": (0,), "hbm": (0,)})
    peers = [engine.submit(f"peer{i}", {"cube": 4.0, "hbm": 4.0},
                           coupled=True) for i in range(peer_count)]
    timer = engine.submit("timer", {}, after=(peers[0],), latency=10.0)
    for _ in range(100):
        if all(task.done for task in (*peers, pinned)) and timer.done:
            break
        engine.advance(until=(timer,), max_steps=1000)
    return engine, pinned, peers, timer


def test_c11_and_python_normalize_mixed_placements_identically(tmp_path):
    for peer_count, expected_timer, expected_pinned in (
        (8, 19.0, 12.0),
        (3, 14.0, 7.0),
    ):
        python = _placed_fixture("python", peer_count=peer_count)
        native = _placed_fixture(
            "c", tmp_path / f"peers-{peer_count}", peer_count=peer_count)
        assert native[0].accel_status()["native_steps"] > 0
        for left, right in zip((python[1], *python[2]), (native[1], *native[2])):
            assert left.time == pytest.approx(right.time, abs=1e-9)
        assert python[3].time == pytest.approx(expected_timer)
        assert native[3].time == pytest.approx(expected_timer)
        # The pinned cohort advances on its own one-unit share, not the aggregate.
        assert python[1].time == pytest.approx(expected_pinned)


def _one_tile(minimum=0.0):
    engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)],
                    backend="python")
    program = PipelineProgram(1, 2, lambda _index: (
        Stage("load", {"hbm": 10.0}),
        Stage("compute", {"cube": 10.0}, previous=("load",)),
    ), minimum_ms=minimum)
    done = program.start(engine, "p", engine.submit("ready"))
    engine.advance(until=(done,), max_steps=100)
    return done.time


def test_tile_zero_ignores_previous_in_both_fast_paths():
    assert _one_tile() == pytest.approx(10.0)
    assert _one_tile(0.01) == pytest.approx(10.0)
    coalesced = coalesce_execution_program(PipelineProgram(
        1, 2, lambda _index: (
            Stage("load", {"hbm": 10.0}),
            Stage("compute", {"cube": 10.0}, previous=("load",)),
        )))
    engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)],
                    backend="python")
    done = coalesced.start(engine, "c", engine.submit("ready"))
    engine.advance(until=(done,), max_steps=100)
    assert done.time == pytest.approx(10.0)


def test_catalog_resolves_iteration_delta_against_the_prior_tile():
    engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)],
                    backend="python")
    template = PipelineTemplate(owner=0, slots=2, phases=(PipelinePhase(
        "tile", (Stage("load", {"hbm": 10.0}),
                 Stage("compute", {"cube": 10.0}, previous=("load",))), 2),))
    catalog = compile_pipeline_template(template, generation=GENERATION)
    submission = submit_program_catalog(
        engine, catalog, name="k", after=(engine.submit("ready"),))
    engine.advance(until=(submission.done,), max_steps=100)
    assert submission.done.time == pytest.approx(20.0)


def test_catalog_carries_stage_cohort_size_and_matches_direct_execution():
    kernel = KernelProgram((
        Stage("group", {"hbm": 4.0, "cube": 4.0}, coupled=True,
              cohort_size=4),
        Stage("peer", {"hbm": 1.0, "cube": 1.0}, coupled=True),
    ))
    catalog = compile_program(kernel, generation=GENERATION)
    group = next(spec for spec in catalog.stages if spec.label == "group")
    assert group.cohort_size == 4
    engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)],
                    backend="python")
    submission = submit_program(
        engine, kernel, name="k", ready=engine.submit("ready"))
    engine.advance(until=(submission.done,), max_steps=100)
    direct = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)],
                    backend="python")
    peer = direct.submit("peer", {"hbm": 1.0, "cube": 1.0}, coupled=True)
    direct.submit("group", {"hbm": 4.0, "cube": 4.0}, coupled=True,
                  cohort_size=4)
    for _ in range(100):
        if peer.done:
            break
        direct.advance(until=(peer,), max_steps=1000)
    assert peer.time == pytest.approx(5.0)
    assert submission.done.time == pytest.approx(peer.time)


# ---------------------------------------------------------------------------
# Fix 2: mHC causality, QKV ports, KDA consumers and native-group isolation


def _mhc_layer(pass_name):
    case = _case("deepseek-v4.1-flash", layers=2,
                 feature_track="expected_ideal")
    return _with_passes(case, pass_name)


def test_mhc_norm_waits_for_its_pre_gate_producer_and_preserves_sinkhorn_overlap():
    pr_ops = _ops(_mhc_layer("mhc_boundary_pr"), phase="decode", tokens=1,
                  context=2048, sequences=2)
    pr_by_id = {op.op_id: op for op in pr_ops}
    norm1 = pr_by_id["s0.l0.norm1"]
    assert pr_by_id["s0.l0.attention.hc_affine"].op_id in norm1.deps
    assert pr_by_id["s0.l0.attention.hc_sinkhorn"].op_id not in norm1.deps
    assert pr_by_id["s0.l0.ffn.hc_post"].recipe == "hc_post_mix"
    norm2 = pr_by_id["s0.l0.norm2"]
    assert "s0.l0.attention.hc_post" in norm2.deps
    assert pr_by_id["s0.l0.ffn.hc_affine"].op_id in norm2.deps
    assert pr_by_id["s0.l0.ffn.hc_sinkhorn"].op_id not in norm2.deps

    ideal_ops = _ops(_mhc_layer("mhc_boundary_ideal"), phase="decode", tokens=1,
                     context=2048, sequences=2)
    ideal_by_id = {op.op_id: op for op in ideal_ops}
    assert (ideal_by_id["s0.l0.attention.hc_sinkhorn"].op_id
            in ideal_by_id["s0.l0.norm1"].deps)
    assert (ideal_by_id["s0.l0.ffn.hc_sinkhorn"].op_id
            in ideal_by_id["s0.l0.norm2"].deps)


def test_lowrank_mhc_preserves_declared_injection_and_gate_slice_reads():
    case = _with_passes(_case("qwen3.8-flash-next", layers=1, quant="w8a8"),
                        "mhc_boundary_pr")
    ops = _ops(case, phase="decode", tokens=1, context=2048)
    by_id = {op.op_id: op for op in ops}
    gate = by_id["s0.l0.attention.hc_gate_mix"]
    combine = by_id["s0.l0.attention.hc_post"]
    up = by_id["s0.l0.attention.hc_up"]
    down = by_id["s0.l0.attention.hc_down_inject"]
    # Declared edges stay conserved: hc_up -> hc_gate_mix (40,960 B) and
    # hc_down_inject -> hc_post (25,608 B), totaling 66,568 B with zero leak.
    assert up.op_id in gate.deps and down.op_id not in gate.deps
    assert gate.mte2_bytes == pytest.approx(40960.0)
    assert down.op_id in combine.deps and up.op_id not in combine.deps
    assert combine.mte2_bytes == pytest.approx(25608.0)


def test_qkv_norm_rope_writes_every_declared_output_and_names_ports():
    case = _with_passes(_case("qwen3-next-80b-a3b", layers=4),
                        "qkv_norm_rope")
    ops = _ops(case, phase="decode", tokens=1, context=128)
    by_id = {op.op_id: op for op in ops}
    fused = by_id["s0.l3.rope"]
    metadata = dict(fused.metadata)
    shapes = dict(metadata["qkv_output_shapes"])
    assert set(shapes) == {"query", "key", "value", "gate"}
    assert fused.mte3_bytes == pytest.approx(
        sum(math.prod(shape) * 2 for shape in shapes.values()))
    assert fused.mte3_bytes == pytest.approx(3072.0)
    assert fused.local_shape == shapes["query"]
    outputs = {value["output"] for value in produced_values(fused)}
    assert outputs == {"query", "key", "value", "gate"}
    cache_ports = {entry["output"]
                   for entry in input_ports(by_id["s0.l3.cache"])}
    assert {"key", "value"} <= cache_ports
    gate_ports = [entry for entry in input_ports(by_id["s0.l3.output_gate"])
                  if entry.get("output") == "gate"]
    assert gate_ports
    # The fused query producer feeds attention directly; the cache edge alone
    # carries key/value and would leave the query output unbound.
    attention = by_id["s0.l3.attention"]
    assert fused.op_id in attention.deps
    assert any(entry.get("op") == fused.op_id
               and entry.get("output") == "query"
               for entry in input_ports(attention))
    from mindie_sim.prediction.implementations import consumer_index
    _, consumers = consumer_index(ops)
    assert any(consumer.op_id == attention.op_id
               and any(entry.get("op") == fused.op_id
                       and entry.get("output") == "query"
                       for entry in input_ports(consumer))
               for _, consumer in consumers.get(fused.op_id, ()))


def test_kda_l2norm_is_retained_for_a_second_sequence_or_phase():
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.recurrent import linear
    from mindie_sim.prediction.transforms import lower_implementations
    from mindie_sim.schema.workload import RequestView

    case = _case("kimi-k3", layers=1, tp=1, ep=1, num_devices=1)
    case = _with_passes(case, "kda_in_kernel_l2norm")
    single = _ops(case, phase="prefill", tokens=64, context=0, sequences=1)
    assert not [op for op in single if op.recipe == "qk_l2norm"]
    assert [op for op in single if dict(op.metadata).get("in_kernel_qk_l2norm")]
    two = _ops(case, phase="prefill", tokens=64, context=0, sequences=2)
    assert [op for op in two if op.recipe == "qk_l2norm"]
    assert not [op for op in two if dict(op.metadata).get("in_kernel_qk_l2norm")]

    # Mixed prefill+decode batches share op.phase == "prefill" in recurrent.py
    # and emit two scan consumers (chunkwise + recurrent_scan1); retain qk_l2norm.
    from mindie_sim.prediction.implementation_selection import resolve
    plan = case.model.layer_plans[0]
    views = (RequestView("p", 64, 0, True, "prefill"),
             RequestView("d", 1, 128, True, "decode"))
    resolved_rt, _ = resolve(
        case.runtime, views, padded_tokens=65, replay_scope="none")
    res = linear(
        prefix="s0.l0", plan=plan, views=views, stage=0, deps="in",
        model=case.model, parallel=resolved_rt.parallel,
        quant=case.model.quant, mode="prefill",
        policy=AttentionPolicy(recurrent_chunk=64), runtime=resolved_rt,
        padded_tokens=65)
    mixed = lower_implementations(
        res.ops, resolved_rt, case.model.activation_dtype)
    norm_ops = [op for op in mixed if op.recipe == "qk_l2norm"]
    assert len(norm_ops) == 1
    decode_scan = next(op for op in mixed if op.op_id.endswith("recurrent_scan1"))
    assert norm_ops[0].op_id in decode_scan.deps


def _kimi_situ_case():
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    raw["model"]["layers"] = 1
    raw["runtime"]["lowering_passes"] = [
        "activation_quantization", "norm_quant", "activation_quant"]
    raw["runtime"].setdefault("kernels", {}).setdefault(
        "lowerings", {})["rmsnorm_quant"] = "vllm_ascend_native_norm"
    case = parse_case(raw)
    quant = replace(
        case.model.quant, weight_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3", weight_block=(32, 32),
        activation_format="fp8_e4m3", activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        exclusions=())
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}),
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["situ_quant"]}}}}
    return replace(case, model=replace(case.model, quant=quant),
                   runtime=replace(case.runtime, kernels=kernels))


def test_generic_activation_quant_leaves_the_selected_native_group_intact():
    from mindie_sim.prediction.native_fusion import _groups
    case = _kimi_situ_case()
    ops = _ops(case, phase="decode", tokens=1, context=16,
               resource_fusion=False)
    groups = [(declaration, members) for declaration, members in _groups(ops)
              if declaration["role"] == "situ_quant"]
    assert groups
    assert all(len(members) == 2 for _declaration, members in groups)


def test_gmm_swiglu_quant_does_not_erase_a_selected_gate_up_native_group():
    raw = json.loads((ROOT / "examples/cases/v4f-a3-w8a8-tp8.json").read_text())
    case = parse_case(raw)
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}),
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {
                "roles": ["grouped_gate_up_activation_quant"],
                "traffic_profile": {
                    "grouped_gate_up_activation_quant":
                        "grouped_swiglu_quant_int8_split_workspace_ae923905"}}}}}
    runtime = replace(
        case.runtime,
        lowering_passes=tuple(dict.fromkeys(
            (*case.runtime.lowering_passes, "gmm_swiglu_quant"))),
        kernels=kernels)
    ops = _ops(replace(case, runtime=runtime), phase="decode", tokens=1,
               context=128)
    native = [op for op in ops if dict(op.metadata).get("fusion_role")
              == "grouped_gate_up_activation_quant"]
    assert native


def test_w4a16_dequant_work_follows_active_packed_weight_bytes():
    case = _case("kimi-k2-thinking", tp=1, dp=16, ep=16, num_devices=16)
    ops = _ops(case, phase="decode", tokens=1, context=128)
    expert = next(op for op in ops
                  if op.layer == 1 and op.op_id.endswith("expert_up"))
    metadata = dict(expert.metadata)
    expected = (float(metadata["dequant_weight_elements"])
                * float(metadata["dequant_ops_per_weight"]))
    assert expert.vector_ops == pytest.approx(expected)
    assert expert.vector_ops == pytest.approx(44040192.0)


def test_quantized_ports_attach_only_to_the_matrix_activation_operand():
    case = _case("deepseek-v3.2", tp=8, ep=1, quant="w8a8")
    ops = _ops(case, phase="decode", tokens=1, context=128)
    by_id = {op.op_id: op for op in ops}
    from mindie_sim.prediction.implementations import input_ports
    norm = by_id["s0.l1.q_a.quantize"]
    residual = [entry for entry in input_ports(norm)
                if entry.get("output") == "residual"]
    assert residual
    assert all(entry.get("output") == "residual" for entry in input_ports(norm))
    matrix_ports = {entry.get("output") for entry in input_ports(by_id["s0.l1.q_a"])}
    assert {"payload", "activation_scales"} <= matrix_ports


def test_moe_init_routing_keeps_the_int32_inverse_permutation_write():
    case = _case("kimi-k3", tp=1, dp=2, ep=2, num_devices=4,
                 quant="w8a8")
    ops = _ops(case, phase="prefill", tokens=64, context=0)
    by_id = {op.op_id: op for op in ops}
    pack = by_id["s0.l1.pack"]
    metadata = dict(pack.metadata)
    routed_slots = pack.local_shape[0] * pack.local_shape[1]
    assert metadata["inverse_index_bytes"] == pytest.approx(routed_slots * 4)
    assert metadata["inverse_index_bytes"] == pytest.approx(4096.0)
    assert "inverse_indices" in pack.returns
    assert pack.mte3_bytes == pytest.approx(3674112.0 + 4096.0)
    unpack = by_id["s0.l1.unpack"]
    assert pack.op_id in unpack.deps


# ---------------------------------------------------------------------------
# Fix 3: index pool, MTP selection, PCP, prefix and sparse attention


def _native_pool_case(scale="fp32", layers=4, **kwargs):
    raw = compose_case("glm-5.3-flash", hardware="a3", tp=8, ep=8,
                       num_devices=8, layers=layers,
                       quant={"name": "C8", "kv_dtype": "fp8_e4m3",
                              "index_dtype": "fp8_e4m3",
                              "index_scale_dtype": scale}, **kwargs)
    raw["runtime"].setdefault("kernels", {}).setdefault(
        "lowerings", {})["pool_key_indexer"] = {
            "implementation": "vllm_ascend_cann_keypool_caller",
            "options": {"tail_capacity": 9, "block_table_columns": 8195}}
    return parse_case(raw)


def test_native_keypool_binds_bf16_pool_projections_and_requires_fp32_scales():
    from mindie_sim.prediction.index_pool_native import (
        POOL_PROJECTION_CATEGORIES, supported,
    )
    case = _native_pool_case("fp32")
    overrides = dict(case.model.quant.category_overrides)
    for category in POOL_PROJECTION_CATEGORIES:
        assert dict(overrides[category])["weight_format"] == "bf16"
    ops = _ops(case, phase="prefill", tokens=8, context=0)
    gate = next(op for op in ops
                if op.layer == 3 and op.recipe == "index_pool_gate")
    assert dict(gate.metadata)["weight_format"] == "bf16"

    plan = case.model.layer_plans[0]
    # The native caller's hardcoded FP32 scale must not be selected for FP16.
    fp16 = _native_pool_case("fp16")
    views = ()
    assert not supported(views=views, plan=plan, model=fp16.model,
                         quant=fp16.model.quant)


def test_native_keypool_preserves_an_explicit_caller_category_override():
    from mindie_sim.prediction.index_pool_native import prepare_model
    from mindie_sim.prediction.implementations import select
    case = _native_pool_case("fp32")
    explicit = replace(case.model.quant, category_overrides=(
        ("indexer.pool_gate", (("weight_format", "fp8_e4m3"),
                               ("activation_format", "fp8_e4m3"))),))
    model = replace(case.model, quant=explicit)
    binding = select(case.runtime.kernels.get("lowerings", {}),
                     "pool_key_indexer")
    prepared = prepare_model(model, case.runtime, binding)
    fields = dict(dict(prepared.quant.category_overrides)["indexer.pool_gate"])
    assert fields["weight_format"] == "fp8_e4m3"


def test_shared_prefix_keypool_plan_matches_the_published_op_bytes():
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeBook
    raw = compose_case("glm-5.3-flash", hardware="a3", tp=8, ep=8,
                       num_devices=8, layers=4, concurrency=4,
                       prompt_tokens=131201, prefix_tokens=131072,
                       prefix_cache_mode="shared",
                       prefix_read_mode="batch_shared",
                       prefix_cache_key="shared_doc")
    raw["runtime"].setdefault("kernels", {}).setdefault(
        "lowerings", {})["pool_key_indexer"] = {
            "implementation": "vllm_ascend_cann_keypool_caller",
            "options": {"tail_capacity": 25, "block_table_columns": 8195}}
    case = parse_case(raw)
    ops = _ops(case, phase="decode", tokens=20, context=131201, sequences=4)
    scored = next(op for op in ops if op.recipe == "cann_pool_key_indexer")
    metadata = dict(scored.metadata)
    ledger = dict(metadata["pool_key_indexer_work"])
    assert scored.hbm_bytes > 0.0
    assert metadata["prefix_read_reuse_applied"] is True
    assert metadata["prefix_read_hbm_bytes"] < (
        ledger["waves"] * ledger["wave_cache_bytes"])
    book = RecipeBook.from_mapping(case.runtime.kernels)
    geometry = Geometry.from_devices(case.machine.dies[:8])
    prepared, plan = book.prepare(scored, geometry)
    assert scored.hbm_bytes == pytest.approx(prepared.hbm_bytes)
    assert scored.hbm_bytes == pytest.approx(plan.work.resources["hbm"])
    assert scored.mte2_bytes == pytest.approx(prepared.mte2_bytes)
    from mindie_sim.prediction.operator_analysis import operator_costs
    row = operator_costs(case, (scored,))["operators"][0]
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(prepared.hbm_bytes)
    # MTE2 keeps the complete per-wave refill plus query-row stream.
    assert prepared.mte2_bytes == pytest.approx(
        ledger["waves"] * ledger["wave_cache_bytes"]
        + ledger["waves"] * ledger["wave_descale_bytes"]
        + ledger["query_rows"] * ledger["heads"] * ledger["width"])


def test_row_shard_option_accepts_the_generic_alias():
    from mindie_sim.prediction.index_pool import row_shard_tp
    runtime = type("Runtime", (), {
        "metadata": {"row_shard_tp": 4},
        "parallel": type("Parallel", (), {"o_proj_tp": 8,
                                          "linear_proj_tp": 1})()})()
    assert row_shard_tp(options={"row_shard_tp": 2}, runtime=runtime) == 2
    assert row_shard_tp(options={}, runtime=runtime) == 4


def test_mtp_shared_selection_consumes_the_producer_geometry():
    case = parse_case(compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8, num_devices=8,
        o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, layers=4,
        concurrency=1, mtp_proposal_tokens=5, graph_capture_mode="full",
        max_num_seqs=8))
    graph = target_graphs(case, phase="decode", tokens=6, context=4,
                          scope="model")[0]
    attention = [op for op in graph if op.recipe.startswith("latent_attention")]
    draft0 = next(op for op in attention if op.op_id.startswith("draft0."))
    draft4 = next(op for op in attention if op.op_id.startswith("draft4."))
    topk = next(op for op in graph if op.recipe == "index_topk")
    assert topk.op_id == "draft0.s0.l0.index_topk"
    metadata = dict(draft0.metadata)
    selected = metadata["indexed_gather_rows"]
    assert selected == 5
    assert metadata["selection_index_read_bytes"] == selected * 4
    assert draft4.flops == pytest.approx(draft0.flops)
    assert dict(draft4.metadata)["indexed_gather_rows"] == selected
    assert dict(draft4.metadata)["selection_index_read_bytes"] == selected * 4


def test_packed_compressor_feeds_the_indexer_from_its_latent_port():
    case = _case("deepseek-v4.1-flash", hardware="a5", layers=3)
    ops = _ops(case, phase="decode", tokens=2, context=2048)
    by_id = {op.op_id: op for op in ops}
    compressor = by_id["s0.l2.compress_compress"]
    quantizer = by_id["s0.l2.index_k.quantize"]
    store = by_id["s0.l2.compressed_store"]
    from mindie_sim.prediction.implementations import input_ports, produced_values
    outputs = {value["output"] for value in produced_values(compressor)}
    assert outputs == {"output", "latent"}
    latent_ports = [entry for entry in input_ports(quantizer)
                    if entry.get("output") == "latent"]
    assert latent_ports
    assert not [entry for entry in input_ports(store)
                if entry.get("output") == "latent"]


def test_pcp_group_completions_follow_owned_query_ranges():
    from mindie_sim.prediction.generic_latent_attention import owned_completed_rows
    split = type("View", (), {
        "context": 0, "q_len": 2, "query_ranges": ((0, 1), (3, 4))})()
    assert owned_completed_rows(split, 4) == 1
    contiguous = type("View", (), {
        "context": 0, "q_len": 2, "query_ranges": ((0, 2),)})()
    assert owned_completed_rows(contiguous, 4) == 0


def test_pcp_compressed_and_index_stores_write_the_completed_group():
    case = _case("deepseek-v4-flash-0731", hardware="a5", tp=1, dp=2,
                 pcp=2, ep=2, num_devices=4, layers=3)
    ops = _ops(case, phase="prefill", tokens=4, context=0)
    by_id = {op.op_id: op for op in ops}
    assert by_id["s0.l2.compressed_store"].mte3_bytes == pytest.approx(1024.0)
    assert by_id["s0.l2.index_store"].mte3_bytes == pytest.approx(130.0)
    assert by_id["s0.l2.pcp_compressed_kv_gather"].link_bytes > 0


def test_pcp_kv_gather_uses_the_region_row_format():
    case = _case("glm-5.3-flash", hardware="a5", tp=1, dp=2, pcp=2, ep=2,
                 num_devices=4, layers=4,
                 quant={"name": "C8", "kv_dtype": "fp8_e4m3"})
    ops = _ops(case, phase="prefill", tokens=8, context=0)
    by_id = {op.op_id: op for op in ops}
    store = by_id["s0.l3.kv_store"]
    gather = by_id["s0.l3.pcp_kv_gather"]
    assert store.mte3_bytes == pytest.approx(2080.0)
    assert gather.link_bytes == pytest.approx(store.mte3_bytes)


def test_arch22_prolog_rejects_nope_and_sharded_cache_ownership():
    from mindie_sim.prediction.mla_prolog_arch22 import validate_declared_view
    from mindie_sim.schema.common import ModelingError
    view = {
        "token_dtype": "int8", "projection_weight_dtypes": ("int8",) * 3,
        "absorbed_weight_dtype": "bf16", "norm_dtype": "bf16",
        "cache_dtype": "bf16", "query_norm_dtype": "int8",
        "query_norm_output": False, "hidden": 4096, "query_rank": 1536,
        "latent_rank": 512, "rope_width": 64, "nope_width": 128,
        "heads": 8, "tokens": 1, "do_rope": True, "parallel_cp": 1,
    }
    assert validate_declared_view(view)["rope"] == 64
    with pytest.raises(ModelingError, match="requires_rope_and_unsharded"):
        validate_declared_view({**view, "do_rope": False})
    with pytest.raises(ModelingError, match="requires_rope_and_unsharded"):
        validate_declared_view({**view, "parallel_cp": 2})


def test_shared_prefix_union_merges_absolute_disjoint_swa_intervals():
    from mindie_sim.prediction.attention import AttentionPolicy, _memory_hierarchy, gqa
    from mindie_sim.prediction.prefix_reads import CacheAccess, plan_prefix_reads
    from mindie_sim.schema.workload import RequestView

    accesses = (
        CacheAccess("a", 1024, q_len=1, prefix_tokens=1023,
                    prefix_key="doc", row_bytes=2048.0, logical_rows=1024),
        CacheAccess("b", 3072, q_len=1, prefix_tokens=1023,
                    prefix_key="doc", row_bytes=2048.0, logical_rows=1024),
    )
    plan = plan_prefix_reads(accesses, mode="batch_shared",
                             kernel_reuse="tile")
    assert plan.unique_hbm_bytes == pytest.approx(4 * 1024 * 1024)
    # Per-tile refills stay private MTE2 service.
    assert plan.l2_to_core_bytes == pytest.approx(2 * 1024 * 2048.0)

    # SWA prefill (window=1024, context=2048, q_len=256): each request touches
    # prefix [1025, 2048) = 1,023 shared rows plus 256 private rows -> 1,535
    # unique rows across two requests (not 2 * 1,279 = 2,558).
    raw = compose_case("gemma-4-31b", hardware="a3", tp=8, ep=1,
                       num_devices=8, layers=1,
                       prefix_cache_mode="shared",
                       prefix_read_mode="batch_shared")
    case = parse_case(raw)
    swa_plan = case.model.layer_plans[0]
    views = tuple(
        RequestView(f"r{i}", 256, 2048, True, "prefill",
                    prefix_tokens=2048, prefix_cache_key="doc")
        for i in range(2)
    )
    hier = _memory_hierarchy(case.machine)
    res = gqa(prefix="s0.l0", plan=swa_plan, views=views, stage=0, deps="in",
              model=case.model, parallel=case.runtime.parallel,
              quant=case.model.quant, mode="prefill",
              policy=AttentionPolicy(), runtime=case.runtime,
              hierarchy=hier, padded_tokens=512)
    attn = next(op for op in res.ops if op.family == "attention")
    meta = dict(attn.metadata)
    assert meta["prefix_read_hbm_bytes"] == pytest.approx(3143680.0)
    assert meta["prefix_read_logical_bytes"] == pytest.approx(9428992.0)


def test_causal_window_tile_span_covers_every_query_in_the_tile():
    from mindie_sim.prediction.generic_latent_attention import window_tile_rows
    plan = type("Plan", (), {"window": 128, "noncausal_attention": False})()
    assert window_tile_rows(plan, context=128, query=128, start=0,
                            query_tile=128, kv_len=256) == 255
    noncausal = type("Plan", (), {"window": 128,
                                  "noncausal_attention": True})()
    assert window_tile_rows(noncausal, context=128, query=128, start=0,
                            query_tile=128, kv_len=256) == 128


def test_windowed_mla_charges_the_full_tile_window_span():
    from mindie_sim.catalog.cache_format import cache_format
    case = _case("deepseek-v4.1-flash", hardware="a5", layers=1)
    ops = _ops(case, phase="prefill", tokens=128, context=128)
    attention = next(op for op in ops
                     if op.layer == 0 and op.family == "attention")
    traffic = dict(attention.metadata)["attention_traffic_plan"]
    plan = case.model.layer_plans[0]
    window_row = cache_format(case.model, "window").row_bytes(
        plan.cache_row_elements)
    # The first query's window starts at absolute position 1 and the tile ends
    # at 256, so 255 distinct rows are live even though the window is 128.
    assert traffic["logical_read_bytes"] == pytest.approx(255 * window_row)
    assert traffic["logical_read_bytes"] > 128 * window_row


def test_lightning_indexer_charges_query_and_head_weight_operands():
    from mindie_sim.catalog.cache import index_row_bytes
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.generic_latent_attention import (
        semantic_index_selection, unique_index_key_rows,
    )
    from mindie_sim.prediction.lowering import Builder
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.workload import RequestView

    base_model = load_model("deepseek-v3.2")
    quant = replace(base_model.quant, index_dtype="fp8_e4m3",
                    index_scale_dtype="fp32")
    model = replace(base_model, quant=quant)
    plan = next(p for p in model.layer_plans if p.indexer)
    views = tuple(RequestView(str(i), 2, 128, False, "prefill")
                  for i in range(2))
    parallel = ParallelSpec()
    row = index_row_bytes(model, plan.index_dim)
    hbm_rows, refill_rows = unique_index_key_rows(
        views, share_prefix=False, noncausal=False, cp=1,
        interleave=128, compression=max(1, plan.compress_ratio))

    def build(raw_eager, kernels=None, arch="A5"):
        runtime = type("Runtime", (), {
            "kernels": dict(kernels or {}),
            "metadata": {"arch_family": arch, "raw_eager": raw_eager},
            "prefix_cache_mode": "private", "prefix_read_mode": "auto",
            "parallel": parallel})()
        builder = Builder("s0", plan, 0, views, model, "prefill")
        query = builder.work("q_operand", (), (4, 128), read=8, write=1024)
        weights = builder.work("w_operand", (), (128, 128), read=8, write=512)
        semantic_index_selection(
            builder, deps=(query, weights), views=views, plan=plan,
            model=model, quant=model.quant, parallel=parallel,
            shard=1, prefix="s0", padded_tokens=4, runtime=runtime)
        operand = sum(op.mte3_bytes for op in builder.ops
                      if op.op_id in (query, weights))
        return builder, operand

    native_builder, native_operands = build(False)
    native = next(op for op in native_builder.ops
                  if op.recipe == "quant_lightning_indexer")
    # Native: per-tile refill service plus the real Q/head-weight operands.
    assert native.mte2_bytes == pytest.approx(
        native_operands + refill_rows * row)
    assert native.hbm_bytes == pytest.approx(
        native_operands + hbm_rows * row + native.mte3_bytes)

    eager_builder, eager_operands = build(True)
    scores = next(op for op in eager_builder.ops if op.recipe == "index_scores")
    assert scores.mte2_bytes == pytest.approx(
        eager_operands + refill_rows * row)
    assert scores.hbm_bytes == pytest.approx(
        eager_operands + hbm_rows * row + scores.mte3_bytes)

    # When declared_native_fusion explicitly claims role "lightning_indexer"
    # (e.g. A2 native observations), semantic_index_selection must keep the
    # 2-member (index_scores, index_topk) pair intact instead of pre-fusing it.
    claimed_kernels = {
        "lowerings": {
            "native_fusion": {
                "implementation": "declared_native_fusion",
                "options": {"roles": ["lightning_indexer"]},
            }
        }
    }
    for arch in ("A2", "A5"):
        claimed_builder, _ = build(False, kernels=claimed_kernels, arch=arch)
        recipes = {op.recipe for op in claimed_builder.ops}
        assert "quant_lightning_indexer" not in recipes
        assert {"index_scores", "index_topk"} <= recipes


def test_sparse_gqa_topk_writes_int32_ids_and_attention_reads_them():
    case = _case("minimax-m3", hardware="a5", layers=4)
    ops = _ops(case, phase="decode", tokens=1, context=8192)
    by_id = {op.op_id: op for op in ops}
    topk = by_id["s0.l3.index_topk"]
    attention = by_id["s0.l3.attention"]
    plan = case.model.layer_plans[3]
    parallel = case.runtime.parallel
    heads = max(1, plan.num_heads // parallel.tp)
    kv_heads = max(1, plan.num_kv_heads // parallel.tp)
    width = plan.head_dim
    valid, selected = topk.local_shape
    # FP32 score read, INT32 selected-ID write.
    assert topk.mte2_bytes == pytest.approx(8193 * 4)
    assert topk.mte3_bytes == pytest.approx(valid * selected * 4)
    metadata = dict(attention.metadata)
    kv_row = kv_heads * width * 2 * 2
    query_bytes = valid * heads * width * 2
    ids = valid * selected * 4
    assert attention.mte2_bytes == pytest.approx(
        metadata["kv_sweeps"] * kv_row + query_bytes + ids)
