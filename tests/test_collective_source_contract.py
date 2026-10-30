"""Independent source-volume contracts for the common expert alltoall path."""
import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from test_prediction_system import case


EXCHANGE_RECIPES = {"expert_dispatch", "expert_combine"}


def source_case(*, dp=1, tp=1, ep=4, balanced=False, tokens=4):
    raw = case(dp=dp, tp=tp, ep=ep, prompt=tokens)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=8, top_k=2, moe_intermediate_size=64,
        num_heads=tp, num_kv_heads=tp, head_dim=32 // tp,
    )
    if balanced:
        raw["runtime"].update(routing={"policy": "balanced"},
                              max_num_batched_tokens=max(16, tokens))
    return raw


def topology(raw):
    parsed = parse_case(raw)
    return Topology(parsed.machine, parsed.runtime.parallel)


def exchanges(ops):
    return [op for op in ops if op.recipe in EXCHANGE_RECIPES]


def metadata_contract(op, *, partitions, basis):
    metadata = dict(op.metadata)
    assert metadata["source_assignment_partitions"] == partitions
    assert metadata["traffic_model"] == "uniform_peer_payload"
    assert metadata["routing_basis"] == basis
    # No duplicate keys may carry a conflicting value for this contract.
    for key in ("source_assignment_partitions", "traffic_model", "routing_basis"):
        assert sum(name == key for name, _ in op.metadata) == 1


@pytest.mark.parametrize("balanced,basis", [
    (False, "probability_moments"),
    (True, "forced_balanced_integer_counts"),
])
def test_dp1_tp4_ep4_declares_one_source_partition_volume(balanced, basis):
    raw = source_case(tp=4, ep=4, balanced=balanced)
    ops = target_graphs(raw, phase="prefill", tokens=4, scope="body")[0]
    both = exchanges(ops)
    assert len(both) == 2
    assert [op.link_bytes for op in both] == [96.0, 96.0]
    for op in both:
        metadata_contract(op, partitions=4, basis=basis)

    # 4 tokens * top-k 2 * hidden 32 * BF16 2 B = 512 B of assignments.
    # Four physical source ranks own 128 B each and send 3/4 remotely: 96 B.
    topo = topology(raw)
    for op in both:
        work = topo.accounting(op, 0)
        assert work.to_dict()["useful_bytes"] == 384.0
        assert work.useful_by_rank == {rank: 96.0 for rank in range(4)}

    # Source partitioning must not change the expert arithmetic: two rows per
    # physical rank in both probability-moment and forced-balanced paths.
    expert_ops = [op for op in ops if op.family == "grouped_gemm"]
    assert expert_ops and all(op.local_shape[0] == 2 for op in expert_ops)


@pytest.mark.parametrize("balanced,basis", [
    (False, "probability_moments"),
    (True, "forced_balanced_integer_counts"),
])
def test_tp8_ep4_keeps_two_independent_etp_group_copies(balanced, basis):
    raw = source_case(tp=8, ep=4, balanced=balanced)
    ops = target_graphs(raw, phase="prefill", tokens=4, scope="body")[0]
    both = exchanges(ops)
    assert len(both) == 2
    assert [op.link_bytes for op in both] == [96.0, 96.0]
    for op in both:
        metadata_contract(op, partitions=4, basis=basis)

    topo = topology(raw)
    assert topo.groups("ep", 0, 0) == ((0, 1, 2, 3), (4, 5, 6, 7))
    for op in both:
        work = topo.accounting(op, 0)
        # Two independent ETP groups each send 4 * 96 B; neither copy is
        # removed by the min(TP, EP) source partition.
        assert work.to_dict()["useful_bytes"] == 768.0
        assert work.useful_by_rank == {rank: 96.0 for rank in range(8)}
        group_totals = [
            sum(work.useful_by_rank[rank] for rank in group)
            for group in topo.groups("ep", 0, 0)
        ]
        assert group_totals == [384.0, 384.0]


@pytest.mark.parametrize("balanced,basis", [
    (False, "probability_moments"),
    (True, "forced_balanced_integer_counts"),
])
def test_dp2_tp2_ep4_charges_each_equal_dp_source_once(balanced, basis):
    raw = source_case(dp=2, tp=2, ep=4, balanced=balanced)
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="body")
    assert tuple(graphs) == (0, 1)

    topo = topology(raw)
    dispatch_totals, source_ranks = [], set()
    for replica, ops in graphs.items():
        both = exchanges(ops)
        assert len(both) == 2
        # 4 local tokens * top-k 2 * hidden 32 * BF16 2 B = 512 B.
        # Two local source partitions own 256 B each; 3/4 are remote: 192 B.
        assert [op.link_bytes for op in both] == [192.0, 192.0]
        for op in both:
            metadata_contract(op, partitions=2, basis=basis)
        dispatch = next(op for op in both if op.recipe == "expert_dispatch")
        work = topo.accounting(dispatch, replica)
        assert work.to_dict()["useful_bytes"] == 384.0
        assert set(work.useful_by_rank) <= ({0, 1} if replica == 0 else {2, 3})
        assert set(work.useful_by_rank.values()) == {192.0}
        dispatch_totals.append(work.to_dict()["useful_bytes"])
        source_ranks.update(work.useful_by_rank)

    # Each of the four physical sources contributes 192 B once across the
    # shared EP cohort: 768 B, not two separate 768 B per-DP totals.
    assert sum(dispatch_totals) == 768.0
    assert source_ranks == {0, 1, 2, 3}


def test_ep1_has_no_expert_exchange_and_keeps_tp_reduction():
    raw = source_case(tp=4, ep=1)
    ops = target_graphs(raw, phase="prefill", tokens=4, scope="body")[0]
    assert not exchanges(ops)
    assert not any(op.family == "alltoall" for op in ops)

    tp_reduce = next(op for op in ops if op.op_id.endswith(".tp_attn"))
    assert tp_reduce.family == "allreduce"
    # Unchanged TP allreduce volume: 2*(TP-1)/TP * 4*32*2 = 384 B per rank.
    assert tp_reduce.link_bytes == 384.0
    work = topology(raw).accounting(tp_reduce, 0)
    assert work.to_dict()["useful_bytes"] == 1536.0
    assert work.useful_by_rank == {rank: 384.0 for rank in range(4)}


def test_documented_recipes_select_pairwise_and_expose_round_tasks():
    raw = source_case(tp=4, ep=4)
    raw["runtime"]["kernels"] = {"recipes": {
        "expert_dispatch": {"kind": "pairwise_collective"},
        "expert_combine": {"kind": "pairwise_collective"},
    }}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    report = graph_costs(raw, graphs)
    rows = [row for row in report["isolated"]["operators"]
            if row["operator"]["recipe"] in EXCHANGE_RECIPES]
    assert {row["operator"]["recipe"] for row in rows} == EXCHANGE_RECIPES
    assert all(row["operator"]["family"] == "alltoall" for row in rows)
    assert all(row["program"] == "PairwiseCollective" for row in rows)
    assert sum(row["communication"]["wire_bytes"] for row in rows) == 768.0

    raw["workload"]["requests"] = [
        {"id": "declared", "prompt_tokens": 4, "output_tokens": 1}]
    served = predict(raw, event_limit=5000)
    assert served["status"] == "modeled"
    tasks = [event["task"] for event in served["events"] if ":i0:" in event["task"]]
    for name in ("dispatch", "combine"):
        service = [task for task in tasks
                   if f"s0.l0.{name}:pairwise:program:group0:" in task
                   and task.endswith(":round0:hop0:service")]
        assert len(service) == 4
