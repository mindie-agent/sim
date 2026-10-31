"""Ultra-long context (400K/1M), synthetic-expert MoE, and A2 hardware generalization tests."""
from dataclasses import replace
import math
import time

import pytest

from mindie_sim import graph_costs, operator_costs, target_operators
from mindie_sim.model_catalog import derive_expert_pruned, load_model
from mindie_sim.prediction.attention_decode import StagedGQADecode
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.fai_decode_mapping import host_partitions, partitions
from mindie_sim.prediction.grouped_dequant_arch35 import Config, assignments, build as build_dsq
from mindie_sim.prediction.index_pool_native import IMPLEMENTATION as KEYPOOL_IMPL
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_init_mxfp8 import RECIPE as MXFP8_INIT_RECIPE, build as build_mxfp8_init
from mindie_sim.prediction.paged_mla import build as build_paged_mla
from mindie_sim.prediction.paged_mla_arch22 import build as build_paged_mla_arch22
from mindie_sim.prediction.program import PipelineProgram
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import OpSpec


A2_SKUS = {
    "Ascend910B1": (1.85e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2C": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B3": (1.80e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910B4": (1.50e9, 20, 40, 34359738368, 0.8e12),
    "Ascend910B4-1": (1.65e9, 20, 40, 68719476736, 1.2e12),
}


def test_periodic_pipeline_compaction_matches_exact_pipeline_program():
    from test_attention_decode import make_op, run

    op = make_op([(1, 131072)] * 8, heads=16, kv_heads=2, dim=128)
    recipe = StagedGQADecode(key_tile=128, query_tile=16)

    import mindie_sim.prediction.attention_decode as ad
    orig_compact = ad.compact_periodic_pipeline
    try:
        ad.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
        eng_exact, end_exact, _, _, _ = run(op, recipe, units=24)
    finally:
        ad.compact_periodic_pipeline = orig_compact

    t0 = time.perf_counter()
    eng_compact, end_compact, _, _, _ = run(op, recipe, units=24)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5
    assert end_compact.time == pytest.approx(end_exact.time, rel=1e-11)
    assert eng_compact.serviced == pytest.approx(eng_exact.serviced, rel=1e-12)
    assert eng_compact.peak_allocated == pytest.approx(eng_exact.peak_allocated, rel=1e-12)


@pytest.mark.parametrize("context_len", (409600, 1048576))
def test_ultra_long_context_400k_and_1m_gqa_and_paged_mla_execute_in_o1_time(context_len):
    from test_attention_decode import make_op, run
    from test_paged_mla import attention as mla_attention, case as mla_case, execute as mla_execute

    t0 = time.perf_counter()
    op = make_op([(1, context_len)] * 8, heads=16, kv_heads=2, dim=128)
    eng, end, _, _, _ = run(op, StagedGQADecode(key_tile=128, query_tile=16), units=24)
    gqa_elapsed = time.perf_counter() - t0
    assert gqa_elapsed < 2.0
    assert eng.completed_count < 1000
    assert end.time > 0
    assert eng.summary()["live_tasks"] == 0
    assert all(v == 0 for v in eng.allocated.values())

    t1 = time.perf_counter()
    raw_mla = mla_case(splits=8)
    raw_mla["runtime"]["max_model_len"] = max(1048576, context_len + 16)
    mla_op, _ = mla_attention(raw_mla, batch=4, context=context_len)
    row = mla_execute(raw_mla, mla_op)
    mla_elapsed = time.perf_counter() - t1
    assert mla_elapsed < 2.0
    assert row["duration_ms"] > 0


def test_fai_decode_mapping_supports_q1_high_batch_and_up_to_64_cores():
    requests = [(1, 409600)] * 32
    mapped = host_partitions(requests, 16, 2, 32, key_tile=128, query_tile=64)
    assert len(mapped) == 64
    owners = {owner for spans in mapped.values() for owner, _, _ in spans}
    assert owners == set(range(32))
    # With tail_tasks = 2 <= 32 // 2 and kv_block_tile = 8, SplitCoreDecodeBS1GN2
    # unpacks the 2 tail batch tasks into single-KV-head tasks.
    requests_tail = [(1, 409600)] * 34
    mapped_tail = host_partitions(requests_tail, 16, 8, 32, key_tile=128, query_tile=64)
    tail_owners = [mapped_tail[(32, h)][0][0] for h in range(8)]
    assert len(set(tail_owners)) == 8
    flash_requests = [(1, 409600)] * 4
    flash_mapped = partitions(flash_requests, 16, 2, 32, key_tile=128, query_tile=64)
    assert len(flash_mapped) == 8
    assert max(owner for spans in flash_mapped.values() for owner, _, _ in spans) < 32


def test_attention_decode_compaction_conserves_causal_tail_and_ragged_multi_request_schedule():
    from test_attention_decode import make_op, run
    import mindie_sim.prediction.attention_decode as ad

    # Finding 1a: q > key_tile (q=12, key_tile=8) spans multiple causal tail tiles.
    op_causal = make_op([(12, 512)], heads=8, kv_heads=2, dim=128)
    recipe_causal = StagedGQADecode(key_tile=8, query_tile=16)
    orig_compact = ad.compact_periodic_pipeline
    try:
        ad.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
        eng_exact_c, end_exact_c, plan_c, _, _ = run(op_causal, recipe_causal, units=1)
    finally:
        ad.compact_periodic_pipeline = orig_compact
    eng_comp_c, end_comp_c, _, _, _ = run(op_causal, recipe_causal, units=1)
    assert eng_comp_c.serviced == pytest.approx(eng_exact_c.serviced, rel=1e-12)
    assert end_comp_c.time == pytest.approx(end_exact_c.time, rel=1e-11)

    # Finding 1b: Ragged multi-request decode [(1, 2048), (1, 1024)] preserves exact
    # staggered completion and UB pool high-water mark.
    op_ragged = make_op([(1, 2048), (1, 1024)], heads=8, kv_heads=2, dim=128)
    recipe_ragged = StagedGQADecode(key_tile=16, query_tile=16)
    try:
        ad.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
        eng_exact_r, end_exact_r, _, _, _ = run(op_ragged, recipe_ragged, units=1)
    finally:
        ad.compact_periodic_pipeline = orig_compact
    eng_comp_r, end_comp_r, _, _, _ = run(op_ragged, recipe_ragged, units=1)
    assert end_comp_r.time == pytest.approx(end_exact_r.time, rel=1e-12)
    assert eng_comp_r.peak_allocated == pytest.approx(eng_exact_r.peak_allocated, rel=1e-12)

    # 3-point periodic convergence control under slow mte1/mte3 regime (warm-up > 24 tiles):
    import test_attention_decode as td
    raw_slow = td.hardware(units=1)
    raw_slow["machine"]["device"]["rates"] = {
        "memory": 15e9, "vector": 26e6, "mte1": 4.5e6, "mte2": 31e6, "mte3": 1.5e6,
    }
    raw_slow["machine"]["device"]["compute_peaks"] = {"bf16": 1e11, "fp32": 1e12}
    recipe_slow = StagedGQADecode(key_tile=32, query_tile=64, slots=2)
    prev_compact_time = 0.0
    for ctx_len in (32800, 32832, 409632, 409664):
        op_slow = make_op([(12, ctx_len)], heads=2, kv_heads=1, dim=32)
        try:
            ad.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
            _, end_u, *_ = run(op_slow, recipe_slow, units=1, raw_case=raw_slow)
        finally:
            ad.compact_periodic_pipeline = orig_compact
        _, end_c, *_ = run(op_slow, recipe_slow, units=1, raw_case=raw_slow)
        assert end_c.time == pytest.approx(end_u.time, rel=1e-11)
        assert end_c.time > prev_compact_time
        prev_compact_time = end_c.time


def test_paged_mla_arch22_multi_job_owner_and_refill_steady_subblock_isolation():
    from test_paged_mla_arch22 import run as a2_run
    from mindie_sim.prediction.paged_mla_arch22_dispatch import source_dispatch
    import mindie_sim.prediction.paged_mla_arch22 as pma22
    from mindie_sim.prediction.refill_steady import _extract_stage
    from mindie_sim.prediction.program import Stage

    # Finding 1c: multi-job owner in paged_mla_arch22 matches exact PipelineProgram.
    raw = source_dispatch((32768, 32768), 16, 24)
    orig_compact = pma22.compact_periodic_pipeline
    try:
        pma22.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
        _, eng_exact, dur_exact = a2_run(raw)
    finally:
        pma22.compact_periodic_pipeline = orig_compact
    _, eng_comp, dur_comp = a2_run(raw)
    assert dur_comp == pytest.approx(dur_exact, rel=1e-12)
    assert eng_comp.serviced == pytest.approx(eng_exact.serviced, rel=1e-12)

    # Finding 6: unsupported nested Stage.program raises ValueError, and
    # multi-tile paged_mla (contexts=(8192,)) preserves per-AIV subblock
    # placement isolation under external contention.
    nested = Stage("bad", program=PipelineProgram(2, 1, lambda i: (Stage("s"),)))
    with pytest.raises(ValueError, match="unsupported_nested_stage_program:PipelineProgram"):
        _extract_stage(nested)

    from test_paged_mla import (
        BUFFERS as MLA_BUFFERS,
        Geometry as MlaGeometry,
        OpSpec as MlaOpSpec,
        RecipeContext as MlaRecipeContext,
        Resource as MlaResource,
        Engine as MlaEngine,
    )
    for heads in (1, 2, 16, 31, 32, 33, 48, 63, 64):
        op = MlaOpSpec(
            "mla", "attention", "decode", 0, 0, (), (), (), (), (),
            metadata=(("paged_mla_geometry", {"contexts": (8192,), "heads": heads, "kv_splits": 1, "page_size": 128}),),
        )
        units = {r: (2 if r in ("vector", "mte2_aiv", "mte3") else 1)
                 for r in ("hbm", "vector", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe")}
        rates = {r: (2 if r == "vector" else 1e20) for r in units}
        rates["cube"] = 1
        pools = {p: max(n, 1 << 20) for p, n in MLA_BUFFERS.items()}
        pu = {p: (2 if p == "ub" else 1) for p in pools}
        plan = build_paged_mla(op, MlaGeometry(1, units, pools, pu))
        engine = MlaEngine([MlaResource(r, v, units=units[r]) for r, v in rates.items()],
                           pools=pools, pool_units=pu)
        ctx = MlaRecipeContext(
            plan.work.apply(op), {}, {r: r for r in rates}, rates, 1,
            resource_units=units, compute_peaks={"bf16": 1e20},
            pool_names={p: p for p in pools}, pool_capacities=pools, pool_units=pu,
        )
        ready = engine.submit("input", release=7)
        competitor = engine.submit(
            "external", {"vector": plan.work.resources["vector"] * 3},
            after=(ready,), placement={"vector": (0,)},
        )
        done = plan.lower(ctx).start(engine, "mla", ready)
        engine.advance(until=(done,))
        expected = plan.work.resources["vector"] * ((heads + 1) // 2) / heads * 2
        assert done.time - 7 == pytest.approx(expected, rel=1e-12)
        engine.advance(until=(competitor,))
        assert not any(engine.allocated.values())


def test_native_keypool_caller_scales_block_table_columns_above_max_model_len():
    raw = {
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 8},
        "model": {"preset": "glm-5.3-flash", "layers": 4, "vocab_size": 128, "auxiliary": None},
        "workload": {"requests": [{"id": "r0", "prompt_tokens": 1024, "output_tokens": 1}]},
        "runtime": {
            "tp": 8,
            "max_model_len": 131072,
            "kernels": {
                "lowerings": {
                    "pool_key_indexer": {
                        "implementation": KEYPOOL_IMPL,
                        "options": {"tail_capacity": 4},
                    }
                }
            },
        },
    }
    ops = target_operators(raw, scope="body", phase="decode", tokens=1, context=409600, sequences=4)
    prep = next(op for op in ops if op.recipe == "cann_keypool_nonempty_inputs")
    meta = dict(prep.metadata)
    expected_cols = math.ceil((409600 + 1) / 128)
    assert meta["block_table_shape"] == (5, expected_cols)
    assert expected_cols > math.ceil(131072 / 128)


@pytest.mark.parametrize("experts,topk", ((16, 2), (32, 4), (48, 6), (64, 8), (96, 4), (128, 8)))
def test_derive_expert_pruned_preserves_dense_layers_and_sanitizes_router_groups(experts, topk):
    for base_key in ("glm-5.3", "deepseek-v4.1-flash", "qwen3.5-35b-a3b", "deepseek-v4-10t"):
        base = replace(
            load_model(base_key),
            expert_routing=tuple(1.0 / 256 for _ in range(256)),
            expert_routing_source="measured",
            layer_plans=tuple(
                replace(p, router_groups=8, router_group_topk=4) if p.ffn == "moe" else p
                for p in load_model(base_key).layer_plans
            ),
        )
        pruned = derive_expert_pruned(
            base, num_experts=experts, top_k=topk, model_id=f"{base_key}-e{experts}-k{topk}"
        )
        assert pruned.num_experts == experts
        assert pruned.top_k == topk
        if len(base.expert_routing) != experts:
            assert pruned.expert_routing == ()
            assert pruned.expert_routing_source == "uniform"
        for orig_plan, new_plan in zip(base.layer_plans, pruned.layer_plans):
            if orig_plan.ffn == "dense":
                assert new_plan.num_experts == 0
                assert new_plan.top_k == 0
            else:
                assert new_plan.num_experts == experts
                assert new_plan.top_k == topk
                if orig_plan.router_groups > 0 and experts % orig_plan.router_groups == 0 and (
                    (experts // orig_plan.router_groups) * orig_plan.router_group_topk >= topk
                ):
                    assert new_plan.router_groups == orig_plan.router_groups
                    assert new_plan.router_group_topk == orig_plan.router_group_topk


def test_moe_init_mxfp8_supports_topk_2_and_synthetic_expert_counts():
    from test_moe_init_mxfp8 import case as mxfp8_case, graph as mxfp8_graph

    raw = mxfp8_case(16)
    raw["model"]["num_experts"] = 16
    raw["model"]["top_k"] = 2
    ops = mxfp8_graph(raw, 16)
    op = next(o for o in ops if o.recipe == MXFP8_INIT_RECIPE)
    row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
    meta = row["operator"]["metadata"]["program_execution"]
    # For topk <= 3, useGatherCopy is False (ScatterCopyExpandedXandMXQuant):
    # each sorted assignment is loaded and quantized individually without cross-topk deduplication.
    assert meta["assignments"] == 16 * 4 * 2
    assert meta["quantized_input_rows"] == 16 * 4 * 2
    assert meta["source_branch"] == "onecore_sort_GATHER_count_mode_full_range_scatterCopy"
    assert meta["output_data_bytes"] == 16 * 4 * 2 * 4096
    assert row["duration_ms"] > 0
    assert row["live_tasks"] == 0


def test_grouped_dequant_arch35_handles_sparse_zero_expert_groups_in_special_mode():
    rows = [0] * 64
    rows[32] = 5
    rows[33] = 7
    rows[63] = 4
    cfg = Config(tuple(rows), 512, 64, 64, 253952)
    assert cfg.special
    assigned = [assignments(cfg, owner) for owner in range(cfg.blocks)]
    by_bank = [sum(n for owner in assigned for g, n in owner if g == bank) for bank in range(64)]
    assert by_bank == rows

    # Finding 4: For E = 128 > 64 AIVs, an empty first-wave group followed by a
    # non-empty second-wave group on the same owner stalls in the source kernel.
    rows_128_stall = [0] * 128
    rows_128_stall[64] = 8
    cfg_stall = Config(tuple(rows_128_stall), 512, 64, 64, 253952)
    with pytest.raises(Exception, match="arch35_dequant_zero_group_stalls_later_source_owner"):
        assignments(cfg_stall, 0)

    # Trailing empty groups after a non-empty first-wave group do not stall.
    rows_128_ok = [0] * 128
    rows_128_ok[0] = 8
    cfg_ok = Config(tuple(rows_128_ok), 512, 64, 64, 253952)
    assert assignments(cfg_ok, 0) == ((0, 8),)


@pytest.mark.parametrize("sku", A2_SKUS)
def test_a2_skus_declare_complete_arch2201_pipeline_and_precision_rates(sku):
    import json
    from pathlib import Path

    raw_profiles = json.loads(
        (Path(__file__).resolve().parents[1] / "mindie_sim/backends/profiles/ascend.json").read_text(encoding="utf-8")
    )
    sku_json = raw_profiles[sku]
    for kind in ("vector", "cube"):
        prov = sku_json["device"]["kernel_invocations"][kind]["provenance"]
        assert prov["basis"] == "arch2201_baseline_extrapolation"
    assert sku_json["source"]["pipeline"]["fields"]["hbm"]["rate_bytes_per_s"] == pytest.approx(A2_SKUS[sku][4])
    assert sku_json["source"]["pipeline"]["provenance"]["architecture_inference"].startswith("A2 npu_arch_2201")

    clock, aic, aiv, memory, hbm = A2_SKUS[sku]
    case = Case.from_mapping({
        "machine": {"backend": "ascend", "sku": sku},
        "model": {"preset": "qwen3.6-27b", "layers": 1, "vocab_size": 128, "auxiliary": None},
        "workload": {"requests": [{"id": "r0", "prompt_tokens": 1024, "output_tokens": 1}]},
        "runtime": {},
    })
    die = case.machine.dies[0]
    assert die.memory_bytes == memory
    assert die.compute_units == aic
    rates = dict(die.rates)
    assert rates["hbm"] == pytest.approx(hbm)
    assert rates["mte1"] == pytest.approx(128 * aic * clock)
    assert rates["mte2_aic"] == pytest.approx(32 * aic * clock)
    assert rates["mte2_aiv"] == pytest.approx(32 * aiv * clock)
    assert rates["mte3"] == pytest.approx(64 * aiv * clock)
    assert rates["fixpipe"] == pytest.approx(128 * aic * clock)
    modes = dict(die.service_modes)
    assert modes["mte1_l0a"] == ("mte1", pytest.approx(256 * aic * clock))
    assert modes["mte1_l0b"] == ("mte1", pytest.approx(128 * aic * clock))
    peaks = dict(die.compute_peaks)
    assert peaks["fp32"] == pytest.approx(peaks["bf16"] / 4)
    assert peaks["hf32"] == pytest.approx(peaks["bf16"] / 2)
    assert peaks["int4"] == pytest.approx(peaks["bf16"] * 4)


def test_indexed_attention_gt16_heads_splits_softmax_and_output_across_both_aivs():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.indexed_attention import build_pipeline
    from mindie_sim.prediction.program import RecipeContext

    rates = {
        "hbm": 1.6e12, "vector": 5e10, "cube": 300e12,
        "mte1": 5e12, "mte2_aic": 1.4e12, "mte2_aiv": 2.8e12,
        "mte3": 5e12, "fixpipe": 5e12,
    }
    res_units = {
        "hbm": 1, "vector": 40, "mte2_aiv": 40, "mte3": 40,
        "mte2_aic": 20, "mte1": 20, "fixpipe": 20, "cube": 20,
    }
    buffers = {"ub": 196608, "l2": 200_000_000}
    buf_units = {"ub": 40, "l2": 1}
    geom = Geometry(
        compute_units=20,
        resource_units=res_units,
        buffers=buffers,
        buffer_units=buf_units,
    )

    def _exec(plan):
        mapped = {}
        engine = Engine(
            [Resource(r, v, units=res_units[r]) for r, v in rates.items()],
            pools=buffers, pool_units=buf_units,
        )
        ctx = RecipeContext(
            OpSpec("idx", "attention", "decode", 0, 0, (), (), (), (), ()),
            {}, {r: r for r in rates}, rates, 20,
            resource_units=res_units, compute_peaks={"bf16": 300e12},
            pool_names={p: p for p in buffers}, pool_capacities=buffers,
            pool_units=buf_units, record_mapping=mapped.update,
        )
        done = plan.lower(ctx).start(engine, "idx", engine.submit("ready"))
        engine.advance(until=(done,))
        return done.time, mapped

    for heads, expected_ownership in (
        (16, "two_gather_AIVs; one_softmax_and_output_AIV"),
        (64, "two_gather_AIVs; two_softmax_and_output_AIVs"),
        (128, "two_gather_AIVs; two_softmax_and_output_AIVs"),
    ):
        shape = {
            "sequences": ((1, 4096),), "topk": 2048, "heads": heads,
            "latent": 512, "rope": 64, "window": 0, "compression_ratio": 1, "cp": 1,
        }
        plan_compact = build_pipeline(shape, geom, scratch_policy="offchip", compact=True)
        plan_exact = build_pipeline(shape, geom, scratch_policy="offchip", compact=False)
        dur_c, map_c = _exec(plan_compact)
        dur_e, _ = _exec(plan_exact)
        assert map_c["vector_ownership"] == expected_ownership
        assert dur_c == pytest.approx(dur_e, rel=0.05)

    # Uneven heads=17 (m0=16 on AIV0, m1=1 on AIV1) subblock contention control:
    from mindie_sim.prediction.indexed_attention import build as build_indexed
    units_1 = {
        r: (2 if r in ("vector", "mte2_aiv", "mte3") else 1)
        for r in ("hbm", "vector", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe")
    }
    rates_1 = {r: (2 if r == "vector" else 1e20) for r in units_1}
    rates_1["cube"] = 1
    shape_17 = dict(
        query=1, context=127, topk=2048, heads=17, latent=512, rope=64,
        query_dtype="bf16", cache_dtype="bf16", window=0, compression_ratio=1, cp=1,
    )
    op_17 = OpSpec("idx", "attention", "decode", 0, 0, (), (), (), (), (),
                   metadata=(("indexed_attention_geometry", shape_17),))
    for compact_mode in (False, True):
        plan_17 = build_indexed(op_17, Geometry(1, units_1, {}, {}), scratch_policy="offchip", compact=compact_mode)
        ctx_17 = RecipeContext(
            plan_17.work.apply(op_17), {}, {r: r for r in rates_1}, rates_1, 1,
            resource_units=units_1, compute_peaks={"bf16": 1e20},
        )
        prog_17 = plan_17.lower(ctx_17)
        times_17 = {}
        for comp_aiv in (None, 0, 1):
            eng_17 = Engine([Resource(r, v, units=units_1[r]) for r, v in rates_1.items()])
            ready_17 = eng_17.submit("ready")
            if comp_aiv is not None:
                eng_17.submit(
                    "competitor", {"vector": 3 * plan_17.work.resources["vector"]},
                    after=(ready_17,), placement={"vector": (comp_aiv,)},
                )
            done_17 = prog_17.start(eng_17, "indexed", ready_17)
            eng_17.advance(until=(done_17,))
            times_17[comp_aiv] = done_17.time
        assert times_17[None] == pytest.approx(65952.0, rel=1e-12)
        assert times_17[0] == pytest.approx(131904.0, rel=1e-12)
        if not compact_mode:
            assert times_17[1] == pytest.approx(74144.0, rel=1e-12)
        else:
            assert times_17[1] < times_17[0]

    # Balanced heads=32 (m0=16 on AIV0, m1=16 on AIV1) subblock contention control:
    shape_32 = dict(shape_17, heads=32)
    op_32 = OpSpec("idx32", "attention", "decode", 0, 0, (), (), (), (), (),
                   metadata=(("indexed_attention_geometry", shape_32),))
    for compact_mode in (False, True):
        plan_32 = build_indexed(op_32, Geometry(1, units_1, {}, {}), scratch_policy="offchip", compact=compact_mode)
        ctx_32 = RecipeContext(
            plan_32.work.apply(op_32), {}, {r: r for r in rates_1}, rates_1, 1,
            resource_units=units_1, compute_peaks={"bf16": 1e20},
        )
        prog_32 = plan_32.lower(ctx_32)
        for comp_aiv, expected_t in ((None, 65952.0), (0, 131904.0), (1, 131904.0)):
            eng_32 = Engine([Resource(r, v, units=units_1[r]) for r, v in rates_1.items()])
            ready_32 = eng_32.submit("ready")
            if comp_aiv is not None:
                eng_32.submit(
                    "competitor", {"vector": 3 * plan_32.work.resources["vector"]},
                    after=(ready_32,), placement={"vector": (comp_aiv,)},
                )
            done_32 = prog_32.start(eng_32, "indexed", ready_32)
            eng_32.advance(until=(done_32,))
            assert done_32.time == pytest.approx(expected_t, rel=1e-12)


@pytest.mark.parametrize("preset,tp,expected_families", (
    ("qwen3.6-27b", 4, {"gqa_decode", "gdn_decode"}),
    ("qwen3.5-35b-a3b", 4, {"gqa_decode", "gdn_decode"}),
    ("glm-5.3-flash", 4, {"indexed_mla"}),
    ("glm-5.3", 2, {"indexed_mla"}),
))
def test_a2_ascend910b2c_installs_default_body_pipelines_and_executes_400k_decode(
    preset, tp, expected_families,
):
    raw = {
        "machine": {"backend": "ascend", "sku": "Ascend910B2C", "num_devices": tp},
        "model": {
            "preset": preset, "quantization": "bf16",
            "layers": 4, "vocab_size": 128, "auxiliary": None,
        },
        "workload": {"requests": [{"id": "r0", "prompt_tokens": 409600, "output_tokens": 1}]},
        "runtime": {"tp": tp, "max_model_len": 524288},
    }
    parsed = parse_case(raw)
    _, selection = effective_body_declaration(parsed, "decode", tokens=1, scope="body")
    installed = {
        row["family"] for row in selection["selections"]
        if row["selection"] == "default_installed"
    }
    assert expected_families <= installed
    ops = target_operators(raw, scope="body", phase="decode", tokens=1, context=409600, sequences=4)
    report = operator_costs(raw, ops)
    assert sum(row["duration_ms"] for row in report["operators"]) > 0


def test_execution_prepare_and_program_cache_normalize_layer_prefixed_metadata():
    from mindie_sim.prediction.execution import _op_meta_key
    from mindie_sim.schema.operators import OpSpec

    op_l1 = OpSpec(
        "s0.l1.mc2_dispatch.send_read_quantize", "mc2_protocol", "decode", 0, 1,
        (), (1, 5120), (1, 5120), (1, 5120), (1, 5120),
        recipe="arch35_fullmesh_resource_phase",
        metadata=(
            ("device_kernel_group", {"id": "s0.l1.mc2_dispatch", "entry": False, "exit": False, "kernel": "MoeDistributeDispatchV3"}),
            ("fusion_group_key", "s0.l1.fusion"),
            ("fusion_semantic_ids", ["s0.l1.a", "s0.l1.b"]),
            ("structure_subgraphs", ({"key": "s0.l1.sub"},)),
        ),
    )
    op_l2 = OpSpec(
        "s0.l2.mc2_dispatch.send_read_quantize", "mc2_protocol", "decode", 0, 2,
        (), (1, 5120), (1, 5120), (1, 5120), (1, 5120),
        recipe="arch35_fullmesh_resource_phase",
        metadata=(
            ("device_kernel_group", {"id": "s0.l2.mc2_dispatch", "entry": False, "exit": False, "kernel": "MoeDistributeDispatchV3"}),
            ("fusion_group_key", "s0.l2.fusion"),
            ("fusion_semantic_ids", ["s0.l2.a", "s0.l2.b"]),
            ("structure_subgraphs", ({"key": "s0.l2.sub"},)),
        ),
    )
    assert _op_meta_key(op_l1) == _op_meta_key(op_l2)
    assert not hasattr(op_l1, "_frozen_meta_key")

    from mindie_sim.prediction.execution import _meta_diff, _meta_patch

    # 1. Deletions (e.g. Work.apply removing resource_work when zeroed) are replayed on cache hit
    orig_l0 = {"resource_work": {"vector": 100.0}, "fusion_group_key": "s0.l0.g"}
    res_l0 = {"fusion_group_key": "s0.l0.g", "program_execution": {"mode": "compact"}}
    diff_del = _meta_diff(orig_l0, res_l0)
    patched_l1 = _meta_patch({"resource_work": {"vector": 100.0}, "fusion_group_key": "s0.l1.g"}, diff_del)
    assert "resource_work" not in patched_l1
    assert patched_l1["fusion_group_key"] == "s0.l1.g"
    assert patched_l1["program_execution"] == {"mode": "compact"}

    # 2. Nested dictionaries (e.g. NativeFusionRecipe updating fusion_boundary.source_program.program)
    # preserve layer-specific sibling keys like fusion_semantic_ids and external_outputs.
    orig_fb_l0 = {
        "fusion_boundary": {
            "fusion_semantic_ids": ["s0.l0.q_a", "s0.l0.q_b"],
            "external_outputs": ["s0.l0.q_b"],
            "source_program": {"program": {"name": "prolog"}},
        }
    }
    res_fb_l0 = {
        "fusion_boundary": {
            "fusion_semantic_ids": ["s0.l0.q_a", "s0.l0.q_b"],
            "external_outputs": ["s0.l0.q_b"],
            "source_program": {"program": {"name": "prolog", "work_binding": "physical_device_geometry"}},
        }
    }
    diff_fb = _meta_diff(orig_fb_l0, res_fb_l0)
    orig_fb_l1 = {
        "fusion_boundary": {
            "fusion_semantic_ids": ["s0.l1.q_a", "s0.l1.q_b"],
            "external_outputs": ["s0.l1.q_b"],
            "source_program": {"program": {"name": "prolog"}},
        }
    }
    patched_fb_l1 = _meta_patch(orig_fb_l1, diff_fb)
    assert patched_fb_l1["fusion_boundary"]["fusion_semantic_ids"] == ["s0.l1.q_a", "s0.l1.q_b"]
    assert patched_fb_l1["fusion_boundary"]["external_outputs"] == ["s0.l1.q_b"]
    assert patched_fb_l1["fusion_boundary"]["source_program"]["program"]["work_binding"] == "physical_device_geometry"


