"""Directed dispatch/return work through ordinary graph and serving boundaries."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.collectives import PairwiseCollective
from mindie_sim.prediction.cost import CostModel, primitive_key
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.peer_traffic import PeerTraffic
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_communication_topology import exchange
from test_peer_traffic import peer_case, MATRIX, directed_wire


def pair(volume, *, family="alltoall", descriptor=None, prefix="body"):
    metadata = (("collective_id", prefix + ".dispatch"),)
    if descriptor is not None:
        metadata += (("peer_traffic", descriptor),)
    forward = replace(exchange(volume, family=family, axis="ep", metadata=metadata),
                      op_id=prefix + ".dispatch")
    reverse = replace(
        exchange(volume, family=family, axis="ep",
                 metadata=(("collective_id", prefix + ".combine"),
                           ("reverse_of_collective", prefix + ".dispatch"))),
        op_id=prefix + ".combine", deps=(forward.op_id,))
    return forward, reverse


def test_asymmetric_cohort_binds_before_graph_and_isolated_costing():
    raw = peer_case(devices=3)
    case = parse_case(raw)
    graphs = {r: pair(v) for r, v in enumerate((0, 4, 8))}
    bound = bind_cohort_traffic(case, graphs)
    assert [ops[1].link_bytes for ops in bound.values()] == [6, 4, 2]
    assert bind_cohort_traffic(case, bound) == bound
    assert [ops[1].link_bytes for ops in graphs.values()] == [0, 4, 8]
    recipes = RecipeBook(families={"alltoall": PairwiseCollective()})
    report = graph_costs(case, graphs, recipes=recipes, input_ready_ms=7)
    standalone = operator_costs(case, graphs, recipes=recipes, input_ready_ms=7)
    expected = directed_wire(MATRIX)
    for name, value in directed_wire(tuple(zip(*MATRIX))).items():
        expected[name] = expected.get(name, 0) + value
    actual = {name: value for name, value in report["execution"]["serviced"].items()
              if value}
    assert actual == pytest.approx(expected)
    assert sum(actual.values()) == 24
    assert report["duration_ms"] == 14
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    for result in (standalone, report["isolated"]):
        assert [row["communication"]["wire_bytes"] for row in result["operators"]] == [0, 4, 8, 6, 4, 2]
        assert not any(row["service_before_input"] for row in result["operators"])
    with pytest.raises(ModelingError, match="complete_dispatch_cohort"):
        bind_cohort_traffic(case, {0: graphs[0]})
    with pytest.raises(ValueError, match="complete_dispatch_cohort"):
        operator_costs(case, graphs[0])


def test_explicit_variable_traffic_is_required_and_changes_primitive_identity():
    case = parse_case(peer_case(devices=3))
    rows = {r: pair(v, family="alltoallv") for r, v in enumerate((0, 4, 8))}
    with pytest.raises(ModelingError, match="alltoallv_requires_explicit_peer_matrix"):
        bind_cohort_traffic(case, rows)
    descriptor = {"kind": "matrix", "payload_bytes": MATRIX}
    bound = bind_cohort_traffic(case, {
        r: pair(v, family="alltoallv", descriptor=descriptor)
        for r, v in enumerate((0, 4, 8))})
    assert [ops[1].link_bytes for ops in bound.values()] == [6, 4, 2]
    different_cells = {"kind": "matrix", "payload_bytes": ((0, 0, 0), (4, 0, 0), (0, 8, 0))}
    other = bind_cohort_traffic(case, {
        r: pair(v, family="alltoallv", descriptor=different_cells)
        for r, v in enumerate((0, 4, 8))})
    assert primitive_key(case, bound[1][0], "none") != primitive_key(case, other[1][0], "none")
    # Even the idle sender must not evade row validation when its link work
    # is filtered out of CostModel's demand mapping.
    bad = replace(bound[1][0], link_bytes=0)
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        Execution(case, CostModel(case))._context(bad, 1, "none")


def test_ordinary_unequal_and_idle_dp_views_use_return_columns_in_execution():
    raw = peer_case(devices=3)
    raw["model"]["layers"] = 1
    raw["machine"]["links"] = [
        {"name": "fabric", "members": [0, 1, 2],
         "capacity_bytes_per_ms": 1000, "semantics": "shared"}]
    case = parse_case(raw)
    views = {r: (() if not r else (
        RequestView(f"opaque{r}", r, 0, True, "prefill", draft_prepare=False),))
        for r in range(3)}
    global_views = tuple(v for batch in views.values() for v in batch)
    works = {r: Frontend()(
        IterationPlan(batch), case.model, case.runtime, case.machine,
        global_tokens=3, global_views=global_views) for r, batch in views.items()}
    seen = {}
    execution = Execution(
        case, CostModel(case), include_submission=False,
        recipes=RecipeBook(families={"alltoall": PairwiseCollective()}),
        on_operator=lambda op, r, *_: seen.update({(r, op.recipe): op})
        if op.family == "alltoall" else None)
    steps = execution.add_group(works, 5)
    execution.engine.advance(until=tuple(s.completion for s in steps.values()), first=False)
    forward = [seen[r, "expert_dispatch"].link_bytes for r in range(3)]
    backward = [seen[r, "expert_combine"].link_bytes for r in range(3)]
    assert forward == pytest.approx([0, 128 / 3, 256 / 3])
    assert backward == pytest.approx([64, 128 / 3, 64 / 3])
    for r in range(3):
        meta = dict(seen[r, "expert_combine"].metadata)
        assert PeerTraffic.from_mapping(meta["peer_traffic"], 3).row_total(r) == pytest.approx(backward[r])
    assert execution.engine.summary()["live_tasks"] == 0
    result = predict(raw, event_limit=1000)
    assert result["status"] == "modeled"
    assert result["execution"]["live_tasks"] == 0
    equal = target_graphs(case, phase="prefill", tokens=2)
    assert all("peer_traffic" in dict(op.metadata) for ops in equal.values()
               for op in ops if op.family == "alltoall")
    # Without source-to-chunk ownership the cohort total (3) cannot describe
    # the actual per-chunk totals (2, 1, 0). Do not execute phantom expert rows.
    chunked = replace(case.runtime, moe_recipe="mc2", mc2_chunks=3)
    with pytest.raises(ModelingError, match="mc2_cross_dp_tails"):
        Frontend()(IterationPlan(views[1]), case.model, chunked, case.machine,
                   global_tokens=3, global_views=global_views)


def test_bound_local_return_is_an_independently_priceable_physical_call():
    raw = peer_case(dp=1, tp=3, ep=3, devices=3)
    raw["model"].update(num_heads=6, num_kv_heads=3, hidden_size=48)
    case = parse_case(raw)
    bound = bind_cohort_traffic(case, {0: pair(4)})[0]
    standalone = operator_costs(case, (bound[1],))["operators"][0]
    together = operator_costs(case, bound)["operators"]
    assert standalone["duration_ms"] == together[1]["duration_ms"]
    assert standalone["serviced"] == together[1]["serviced"]
    assert sum(standalone["serviced"].values()) == 12


def test_repeated_invocations_keep_distinct_pairing_but_share_physical_cost():
    case = parse_case(peer_case(devices=3))
    graphs = {r: (*pair(v, prefix="target"), *pair(v, prefix="draft0"))
              for r, v in enumerate((0, 4, 8))}
    result = operator_costs(case, graphs, recipes=RecipeBook(
        families={"alltoall": PairwiseCollective()}))
    assert len(result["collective_cohorts"]) == 2
    assert all(row["occurrences"] == 2 for row in result["collective_cohorts"])
    assert result["operator_occurrences"] == 12
