"""L2 subgraph fusion, contention and counterfactual monotonicity.

[Role] Exercise every L2 kind (A0-A4, K0-K1, I0-I1, F0-F4, N0, R0, R1)
  through native fusion, multi-stream fork/join, and counterfactual scaling.
[Invariants] Fused duration is at most the unfused duration. Dropped HBM
  equals dropped MTE2 plus MTE3, and cube/vector/mte1/fixpipe/link work is
  conserved. Contention makespan lies between the longest branch and the sum
  of the branches. Pipe and AIC/AIV scaling from 0.5x to 2.0x is
  non-increasing, and a unique stage bottleneck is strictly decreasing.
  Width and expert growth is non-decreasing at the L2 subgraph.
[Timeouts & Locks] Tier 0 (<5s), pure CPU arithmetic, no I/O or locks.
[Side-Effect Policy] Read-only catalog and in-memory model construction only.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache

from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec

KINDS = ("A0", "A1", "A2", "A3", "A4", "K0", "K1", "I0", "I1",
         "F0", "F1", "F2", "F3", "F4", "N0", "R0", "R1")
PIPES = ("cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe", "link")
COMPUTE = ("cube", "vector", "mte1", "fixpipe", "link")
# One canonical model and role set per kind. None keeps every role of that kind.
PICK = {
    "A0": ("dense", ("attention",)), "A1": ("mla", ("attention",)), "A2": ("wc", None),
    "A3": ("gdn", None), "A4": ("kda", None), "K0": ("dense", ("kv",)), "K1": ("wc", None),
    "I0": ("mla", ("index_produce",)), "I1": ("mla", ("index_select",)),
    "F0": ("dense", ("dense_ffn",)), "F1": ("moe", ("router",)), "F2": ("moe", None),
    "F3": ("moe", ("routed_ffn",)), "F4": ("moe", None), "N0": ("dense", ("attn_norm",)),
    "R0": ("dense", None), "R1": ("dense", ("residual",)),
}
FORKS = (
    ("gdn", "multistream_kda_proj", {"main", "aux"}),
    ("kda", "multistream_kda_proj", {"main", "aux"}),
    ("wc", "multistream_window_compressed", {"main", "aux"}),
    ("moe", "multistream_shared_expert", {"shared", "routed"}),
    ("mla", "multistream_attention_indexer", {"attention", "indexer"}),
)
BASE = {pipe: (2_000_000.0 if pipe.startswith("mte") or pipe == "fixpipe" else 1_000_000.0) for pipe in PIPES}
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4, split_target=1000, max_splits=4)
GEOM = WorkloadGeometry.homogeneous(2, 8, 32, phase="prefill")
SLACK = 1e-8


def _spec(plans, **overrides):
    base = dict(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128)
    base.update(overrides)
    return ModelSpec.from_mapping(dict(base, name="l2", layers=len(plans), layer_plans=plans))


def _gqa(**extra):
    plan = dict(index=0, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2, head_dim=16,
                intermediate_size=128, qk_norm=True)
    plan.update(extra)
    return plan


@lru_cache(maxsize=16)
def _catalog(hidden, experts, streams):
    inter = hidden * 2
    moe_inter = max(32, inter // 2)
    common = dict(hidden_size=hidden, intermediate_size=inter)
    recurrent = dict(linear_heads=4, linear_head_dim=16, linear_key_heads=4, linear_key_head_dim=16,
                     linear_value_heads=4, conv_kernel=4, intermediate_size=inter)
    specs = {
        "dense": (_spec([_gqa(owns_long_kv=True, hyper_connections=True, hc_mult=4, hc_sinkhorn_iters=2,
                              residual_block_size=2, intermediate_size=inter)], **common), {}),
        "mla": (_spec([dict(index=0, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1, head_dim=16,
                            intermediate_size=inter, indexer=True, index_heads=2, index_dim=16, index_topk=8,
                            q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8,
                            v_head_dim=16, mla_variant="latent_mla")],
                      **common, mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8,
                                         qk_rope_head_dim=8, v_head_dim=16, index_topk=8)),
                {"multistream_attention_indexer": streams}),
        "wc": (_spec([_gqa(cache_kind="window_compressed", window=8, compress_ratio=4,
                           compressor_kind="weighted_pool", indexer=True, index_heads=2, index_dim=8,
                           index_topk=4, owns_long_kv=True, intermediate_size=inter)], **common),
               {"multistream_window_compressed": streams}),
        "gdn": (_spec([_gqa(attention="gdn", **recurrent)], **common), {"multistream_kda_proj": streams}),
        "kda": (_spec([_gqa(attention="kda", gate_rank=4, **recurrent)], **common),
                {"multistream_kda_proj": streams}),
        "moe": (_spec([_gqa(ffn="moe", num_experts=experts, top_k=2, moe_intermediate_size=moe_inter,
                            num_shared_experts=1, shared_intermediate_size=moe_inter, intermediate_size=inter)],
                      **common, num_experts=experts, top_k=2, moe_intermediate_size=moe_inter,
                      num_shared_experts=1),
                {"ep": 2, "moe_recipe": "alltoall", "multistream_shared_expert": streams}),
    }
    built = {}
    for key, (spec, flags) in specs.items():
        features = FeatureConfig.from_model(
            spec, cores=2, token_tile=16, kv_tile=16, moe_row_tile=4, split_target=1000, max_splits=4, **flags)
        built[key] = (spec, features, instantiate_model(spec, features))
    return built


def _seed():
    state = StateView().mark_kv(0).mark_index(0).mark_selection(0)
    records = [(index, request.context, request.q_len) for index, request in enumerate(GEOM.requests)]
    state, _produced = state.update_compression(0, 4, records)
    return state


def _selected(catalog, kind):
    key, roles = PICK[kind]
    chosen = []
    for layer in catalog[key][2].layers:
        for item in layer.subgraphs:
            role = str(dict(item.metadata).get("role", ""))
            if item.kind == kind and (roles is None or role in roles):
                chosen.append(item)
    assert chosen, kind
    return chosen


def _compile(items, supply, policy=POLICY):
    compiler = OperatorPiecewiseCompiler(supply, policy)
    return [compiler.compile(item) for item in items]


def _eval(graphs, fusion=False):
    state = _seed()
    results = []
    for graph in graphs:
        names = tuple(contract.name for contract in graph.instance.template.fusions) if fusion else ()
        results.append(evaluate_serial_split(graph, GEOM, state, fusion=names))
    return results


def _duration(graphs, fusion=False):
    return sum((result.total_ms for result in _eval(graphs, fusion)), Fraction(0))


def _supply(rates=None):
    return PipeSupply(tuple((rates or BASE).items()))


def _retarget(graphs, supply):
    for graph in graphs:
        graph.supply = supply
        for operator in graph.operators:
            operator.supply = supply


def _work(pairs):
    total = {}
    for pipe, amount in pairs:
        total[pipe] = total.get(pipe, Fraction(0)) + amount
    return total


def _unique(results, supply):
    """Pipes that alone set a stage time. Scaling one of them must shorten the subgraph."""
    winners = set()
    for result in results:
        for cost in result.costs:
            stages = cost.stage_work or ((cost.work,) if cost.work else ())
            for stage in stages:
                times = {pipe: float(amount) / supply.effective(pipe) for pipe, amount in stage if amount > 0}
                if not times:
                    continue
                peak = max(times.values())
                tops = [pipe for pipe, elapsed in times.items() if abs(elapsed - peak) <= 1e-12 * max(1.0, peak)]
                if len(tops) == 1 and peak > 0:
                    winners.add(tops[0])
    return winners


def _bounds(envelope):
    times = [float(value) for _name, value in envelope.branches]
    span = float(envelope.makespan_ms)
    assert times and max(times) - SLACK <= span <= sum(times) + SLACK
    assert span + SLACK >= float(envelope.lower_bound_ms)


def _measure(hidden, experts):
    catalog = _catalog(hidden, experts, False)
    supply = _supply()
    return {kind: float(_duration(_compile(_selected(catalog, kind), supply))) for kind in KINDS}


def _hardware(aic, aiv):
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 2,
        "num_cores": aic, "aiv_count": aiv,
    })
    return PipeSupply.from_hardware(hardware)


def test_every_l2_kind_has_positive_duration():
    catalog = _catalog(64, 4, False)
    supply = _supply()
    for kind in KINDS:
        assert _duration(_compile(_selected(catalog, kind), supply)) > 0


def test_fusion_conserves_compute_and_pairs_hbm_with_mte():
    catalog = _catalog(64, 4, False)
    supply = _supply()
    checked = set()
    for kind in ("N0", "A0", "A1", "F0"):
        for graph in _compile(_selected(catalog, kind), supply):
            names = tuple(contract.name for contract in graph.instance.template.fusions)
            assert names
            off = _duration([graph], fusion=False)
            on = _duration([graph], fusion=True)
            assert on <= off
            for result in _eval([graph], fusion=True)[0].fusion_results:
                checked.add(result.name)
                assert result.removed_bytes > 0
                assert result.fused.duration_ms <= result.isolated_ms
                fused, member = _work(result.fused.work), {}
                for cost in result.semantic:
                    for pipe, amount in cost.work:
                        member[pipe] = member.get(pipe, Fraction(0)) + amount
                for pipe, amount in fused.items():
                    assert amount <= member.get(pipe, Fraction(0))
                for pipe in COMPUTE:
                    assert fused.get(pipe, Fraction(0)) == member.get(pipe, Fraction(0))
                dropped = {pipe: member.get(pipe, Fraction(0)) - fused.get(pipe, Fraction(0))
                           for pipe in ("hbm", "mte2", "mte3")}
                assert dropped["hbm"] == dropped["mte2"] + dropped["mte3"] > 0
    assert checked == {"add_rmsnorm_quant", "qkv_norm_rope", "mla_prolog", "dequant_swiglu_quant"}
    for graph in _compile(_selected(catalog, "A1"), _hardware(32, 64)):
        f_res = _eval([graph], fusion=True)[0].fusion_results
        assert len(f_res) > 0
        for result in f_res:
            fused, member = _work(result.fused.work), {}
            for cost in result.semantic:
                for pipe, amount in cost.work:
                    member[pipe] = member.get(pipe, Fraction(0)) + amount
            dropped = {pipe: member.get(pipe, Fraction(0)) - fused.get(pipe, Fraction(0))
                       for pipe in ("hbm", "mte2", "mte3")}
            # Dropped HBM is the traffic fusion actually removed.  The old
            # 4608-byte snapshot counted norm bytes the empirical scale tables
            # had added; those bytes are not in the structural program.
            assert dropped["hbm"] == dropped["mte2"] + dropped["mte3"] == result.removed_bytes > 0


def test_multistream_contention_stays_inside_branch_bounds():
    catalog = _catalog(64, 4, True)
    supply = _supply()
    seen = set()
    for kind in ("A2", "A3", "A4"):
        for graph in _compile(_selected(catalog, kind), supply):
            if "aux" not in {operator.operator.stream for operator in graph.operators}:
                continue
            envelope = evaluate_serial_split(graph, GEOM, _seed(), fusion=()).contention
            assert envelope is not None and set(dict(envelope.branches)) == {"main", "aux"}
            _bounds(envelope)
            seen.add(kind)
    assert seen == {"A2", "A3", "A4"}
    geometry = WorkloadGeometry.homogeneous(2, 4, 32, phase="prefill")
    for key, flag, expect in FORKS:
        spec, features, _inst = catalog[key]
        overlapped = assemble_model(spec, supply=supply, policy=POLICY, features=features).forward(geometry)
        serial = assemble_model(
            spec, supply=supply, policy=POLICY, features=replace(features, **{flag: False})).forward(geometry)
        envelope = overlapped.layers[0].contention
        assert envelope is not None and set(dict(envelope.branches)) == expect
        _bounds(envelope)
        assert overlapped.layers[0].duration_ms <= serial.layers[0].duration_ms


def test_pipe_scale_is_non_increasing_and_strict_on_unique_bottlenecks():
    catalog = _catalog(64, 4, False)
    base = _supply()
    compiled = {kind: _compile(_selected(catalog, kind), base) for kind in KINDS}
    unique = {kind: _unique(_eval(graphs), base) for kind, graphs in compiled.items()}
    assert all(unique.values())
    for pipe in PIPES:
        series = {}
        for factor in (0.5, 1.0, 2.0):
            rates = dict(BASE)
            rates[pipe] = BASE[pipe] * factor
            _retarget(list(graph for graphs in compiled.values() for graph in graphs), _supply(rates))
            for kind, graphs in compiled.items():
                series.setdefault(kind, []).append(float(_duration(graphs)))
        for kind, values in series.items():
            assert values[0] + SLACK >= values[1] >= values[2] - SLACK, (kind, pipe, values)
            if pipe in unique[kind]:
                assert values[0] > values[1] > values[2], (kind, pipe, values)


def test_width_and_expert_growth_is_non_decreasing():
    widths = [_measure(hidden, 4) for hidden in (64, 128, 256)]
    for kind in KINDS:
        values = [row[kind] for row in widths]
        assert values[0] <= values[1] + SLACK and values[1] <= values[2] + SLACK, (kind, values)
        if kind not in {"K0", "K1", "I1"}:
            assert values[0] < values[1] < values[2], (kind, values)
    experts = [_measure(64, count) for count in (4, 8, 16)]
    for kind in ("F1", "F2", "F3", "F4"):
        values = [row[kind] for row in experts]
        assert values[0] <= values[1] + SLACK and values[1] <= values[2] + SLACK, (kind, values)
        if kind in {"F1", "F3"}:
            assert values[0] < values[1] < values[2], (kind, values)


def test_aic_and_aiv_scale_is_non_increasing():
    catalog = _catalog(64, 4, False)
    items = {kind: _selected(catalog, kind) for kind in KINDS}

    def measure(aic, aiv):
        supply = _hardware(aic, aiv)
        policy = ImplementationPolicy(mt=16, nt=16, cores=aic, moe_row_tile=4, split_target=1000, max_splits=4)
        return {kind: float(_duration(_compile(items[kind], supply, policy))) for kind in KINDS}

    rows = {"aic": [measure(aic, aic * 2) for aic in (16, 32, 64)],
            "aiv": [measure(32, aiv) for aiv in (32, 64, 128)]}
    for label, group in rows.items():
        for kind in KINDS:
            values = [group[index][kind] for index in range(3)]
            assert values[0] + SLACK >= values[1] >= values[2] - SLACK, (label, kind, values)
