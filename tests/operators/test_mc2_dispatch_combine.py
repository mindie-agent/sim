"""MC2 routing, caller boundaries, and executable transport contracts.

[Role] Exercise the governed MC2 entrypoint and the ordinary MoE caller.
[Invariants] Directed payload, matrix work, ownership and causal edges survive
  planning, binding and execution across declared architectures.
[Timeouts & Locks] Finite analytical graphs with bounded engine traces.
[Side-Effect Policy] No device access. C11 uses the isolated test cache.
"""
from collections import Counter
from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest
from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import hardware_case, machine

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.moe_routing.moe_token_partition import KIND, select as select_partition
from mindie_sim.prediction.operators import mc2
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.registry import FamilyDispatcher, FamilyRegistry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0
GATHER, SCATTER, DISPATCH, COMBINE = (
    "all_gather_matmul", "matmul_reduce_scatter", "dispatch_v2", "combine_v2")
CLIENT, SIO_SWITCH, SHMEM, WINDOW, CCU = ("a2_hccl_client", "a3_sio_hccs", "a5_shmem", "a5_mte", "a5_ccu")
CLIENT_BINDING = {"caller_abi": "hccl_client", "server_mode": "aiv"}
FRAME, DATA, FLAG, ROW = 512, 480, 32, 240
M, K, N = 24, 32, 48
CHUNK_SETTINGS = dict(internal_chunks=2, independent_buffers=True, independent_handles=True)
RELEASE, RATE, FAST = 7, 64000, 1e9
MESH_PEERS, ROUTING_WIDTH, INJECTION_RATE = 8, 4096, 32000
ROLES = ("hbm", "l2", "mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe", "vector", "link", "cube")
BASE_SKU, DUAL_SKU, MESH_SKU = REPRESENTATIVE_SKUS.values()

def call(abi, peers, *, shape=(), **fields):
    shape = (M, K, N) if abi in (GATHER, SCATTER) else shape
    return OpSpec("call", "mc2", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", recipe=f"mc2_{abi}",
                  metadata=(("mc2", dict(abi=abi, world_size=peers, **fields)),))

def geometry(sku, peers=1):
    hw = machine(sku, num_devices=peers)
    return Geometry.from_devices(hw.dies[:1], links=hw.links,
                                 transfer_channels=hw.transfer_channels)

def lower_plan(op, geo, **modes):
    rates = dict.fromkeys(ROLES, RATE)
    units = {role: geo.resource_units.get(role, 1) for role in rates}
    context = RecipeContext(op, {}, {r: r for r in rates}, rates, geo.compute_units,
                            resource_units=units, compute_peaks={"bf16": RATE * 1000},
                            service_modes=modes, datapath=geo.datapath)
    return mc2.build(op, geo), context

def execute(program, rates, units, *, backend="python"):
    engine = Engine([Resource(r, rate, units=units[r]) for r, rate in rates.items()],
                    trace_limit=10000, backend=backend)
    ready = engine.submit("input", release=RELEASE)
    done = program.start(engine, "call", ready)
    engine.advance(horizon=RELEASE - 1)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    engine.advance()
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    return done.time - RELEASE, engine

def stages(program):
    return {stage.name: stage for stage in program.stages}

def timings(engine):
    return {stage["task"].rsplit(":", 1)[-1]: stage for stage in engine.trace}

def ancestors(graph, name):
    pending, seen = list(graph[name].after), set()
    while pending:
        parent = pending.pop()
        if parent not in seen:
            seen.add(parent)
            pending.extend(graph[parent].after)
    return seen

@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("abi", (GATHER, SCATTER, DISPATCH, COMBINE))
@pytest.mark.parametrize("peers", (1, 4))
def test_abi_conservation_and_occurrence_bound_dispatch(sku, abi, peers):
    geo = geometry(sku, peers)
    active_arch = geo.datapath.compute_arch.arch_id == ARCH_IDS[-1].values[0]
    binding = {} if active_arch else {"transport_binding": CLIENT_BINDING}
    remote, flops = 0, 0
    if abi in (GATHER, SCATTER):
        operand = M * (K if abi == GATHER else N) * 2
        remote, flops = (peers - 1) * operand / peers, 2 * M * K * N
    else:
        bindings = dict(width=ROW + 1, quant_mode="per_token", payload_dtype="int8",
                        combine_width=ROW + 2, combine_dtype="fp16", status_bytes=0)
        binding.update(bindings, assignment_matrix={(0, dst): 2 for dst in range(peers)})
        remote = (peers - 1) * 2 * FRAME * (1 if abi == DISPATCH else 2)
    op = call(abi, peers, **binding)
    occurrence = OpOccurrence(3, op.stage, 0, op.op_id)
    ctx = OperatorPlanningContext(occurrence, geo, geo.datapath)
    resolved = FamilyDispatcher(FamilyRegistry((mc2.register_family(),))).resolve_plan(op, ctx)
    plan, bind = lower_plan(op, geo)
    descriptor = resolved.descriptor
    assert isinstance(descriptor, FusedPipelineDescriptor)
    assert descriptor.owner == occurrence and descriptor.member_op_ids == (op.op_id,)
    assert resolved.recipe == plan and descriptor.sources == plan.pipeline_spec.sources
    assert plan.metadata["transport_abi"] == ((CCU if flops else WINDOW) if active_arch else CLIENT)
    assert descriptor.implementation == plan.metadata["implementation"]
    assert resolved.op.link_bytes == plan.metadata["peer_payload_bytes"] == remote
    assert resolved.op.flops == sum(plan.work.matrix.values()) == flops
    if abi in (DISPATCH, COMBINE):
        pack = stages(plan.program(bind))["cohort0_pack"]
        assert pack.work.get("mte2_aiv") == 2 * peers * (2 * (ROW + 1) + 4 if abi == DISPATCH else 2 * (ROW + 2) + 12)
    work = Counter()
    for phase in descriptor.phases:
        for item in phase.work:
            assert item.unit == ("flop" if item.resource == "matrix" else "op" if item.resource == "vector" else "bytes")
            work[item.resource] += item.amount
    assert +work == +Counter({**plan.work.resources, "matrix": flops})
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    program = restored.program(bind)
    assert {p.phase_id: p.after for p in descriptor.phases} == {s.name: s.after for s in program.stages}
    _, engine = execute(restored.lower(bind), bind.rates, bind.resource_units)
    assert +Counter(engine.serviced) == pytest.approx(+Counter(bind.bind(plan.work)))

@pytest.mark.parametrize("algorithm,hops", (("ring", ("east", "west")), ("two_tier", ("local", "spine"))))
def test_hccl_algorithm_requires_binding_and_explicit_hops(algorithm, hops):
    geo = geometry(BASE_SKU, 4)
    base = call(GATHER, 4, transport_binding=CLIENT_BINDING)
    assert mc2.build(base, geo).metadata["topology_algorithm"] == "fullmesh"
    binding = {**CLIENT_BINDING, "algorithms": (algorithm,)}
    fields = dict(transport_binding=binding, topology_algorithm=algorithm)
    for changes in ({}, {"transport_binding": CLIENT_BINDING,
                         "topology_hops": {(0, p): hops for p in range(1, 4)}}):
        with pytest.raises(UnmodeledCapabilityError, match="binding_and_group_hops"):
            mc2.build(call(GATHER, 4, **{**fields, **changes}), geo)
    plan, ctx = lower_plan(call(GATHER, 4, **fields, topology_hops={(0, p): hops for p in range(1, 4)}), geo)
    transfers = [s for s in plan.program(ctx).stages if s.work.get("link")]
    assert len(transfers) == 3 and all(s.metadata["topology_hops"] == "/".join(hops) for s in transfers)
    assert plan.metadata["peer_credits"] == (1 if algorithm == "ring" else len(transfers))

@pytest.mark.parametrize("packages,expected", (((0, 0, 1, 1), ("SIO", "HCCS_SW")),
                                                ((0, 1, 0, 1), ("HCCS_SW", "SIO"))))
def test_die_routes_follow_placement_at_fixed_group_size(packages, expected):
    geo = geometry(DUAL_SKU, 4)
    with pytest.raises(UnmodeledCapabilityError, match="peer_hops_required"):
        mc2.build(call(GATHER, 4, transport_abi=SIO_SWITCH), geo)
    group = dict(package_ids=packages, node_ids=(0, 0, 0, 1), cross_domain_hops=("edge", "pod"))
    plan, ctx = lower_plan(call(GATHER, 4, group=group), geo)
    actual = {s.metadata["destination_rank"]: s.metadata["topology_hops"]
              for s in plan.program(ctx).stages if s.work.get("link")}
    assert plan.metadata["transport_abi"] == SIO_SWITCH
    assert actual == dict(enumerate((*expected, "edge/pod"), 1))

@pytest.mark.parametrize("hops", (("sio", "hccs_sw"), ("sio", "hccs_sw", "sio")))
@pytest.mark.parametrize("physical", (False, True))
def test_each_directed_hop_carries_the_full_transfer(hops, physical):
    op = call(COMBINE, 2, width=ROW, transport_abi=SIO_SWITCH,
              assignment_matrix={(1, 0): 2},
              topology_hops={(0, 1): ("hccs_sw",), (1, 0): hops})
    plan, ctx = lower_plan(op, geometry(DUAL_SKU, 2))
    if physical:
        ctx = replace(ctx, rates={**ctx.rates, "sio": RATE * 2, "hccs_sw": RATE},
                      resource_names={**ctx.resource_names, "sio": "sio", "hccs_sw": "hccs_sw"},
                      resource_units={**ctx.resource_units, "sio": 1, "hccs_sw": 1})
    transfers = sorted((stage for stage in plan.program(ctx).stages if "hop_index" in stage.metadata),
                       key=lambda stage: stage.metadata["hop_index"])
    assert tuple(stage.metadata["physical_hop"] for stage in transfers) == hops
    assert plan.work.resources["link"] == len(hops) * 2 * FRAME
    assert transfers[-1].name == "cohort0_recv_link_1"
    _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units)
    trace = timings(engine)
    for i, (stage, hop) in enumerate(zip(transfers, hops)):
        resource = hop if physical else "link"
        assert stage.work == {resource: 2 * FRAME}
        assert stage.metadata["hop_bytes"] == 2 * FRAME
        assert (stage.metadata["source_rank"], stage.metadata["destination_rank"]) == (1, 0)
        if i:
            prev = transfers[i - 1].name
            assert stage.after == (prev,)
            assert trace[stage.name]["start_ms"] >= trace[prev]["end_ms"]
    assert trace["cohort0_unpack"]["start_ms"] >= trace[transfers[-1].name]["end_ms"]
    assert sum(engine.serviced.get(role, 0) for role in ("link", "sio", "hccs_sw")) == len(hops) * 2 * FRAME

@pytest.mark.parametrize("arch", ARCH_IDS)
def test_foreign_transport_is_rejected_at_public_boundary(arch):
    sku = MESH_SKU if arch == ARCH_IDS[-1].values[0] else BASE_SKU
    geo = geometry(sku)
    invalid = (CLIENT,) if sku == MESH_SKU else (SHMEM, WINDOW, CCU)
    for transport in invalid:
        with pytest.raises(UnmodeledCapabilityError, match="transport_unsupported") as failure:
            mc2.build(call(DISPATCH, 1, width=ROW, transport_abi=transport), geo)
        assert failure.value.onboarding_hint["schema"] == "mindie-unmodeled-structure-hint/1"

@pytest.mark.parametrize("backend", ("python", "c"))
@pytest.mark.parametrize("declare_ports", (False, True))
def test_direct_mesh_alignment_and_active_peers_share_injection(backend, declare_ports):
    peers = MESH_PEERS
    geo = geometry(MESH_SKU, peers)
    fields = dict(transport_abi=SHMEM, width=ROW, status_bytes=0)
    binding = {"caller_abi": "shmem", "window_offset_bytes": FRAME}
    for change, error in (({}, "shmem_caller"), ({**binding, "window_offset_bytes": 1}, "aligned_window")):
        with pytest.raises(UnmodeledCapabilityError, match=error):
            mc2.build(call(DISPATCH, peers, **fields, transport_binding=change), geo)
    op = call(DISPATCH, peers, **fields, transport_binding=binding)
    with pytest.raises(UnmodeledCapabilityError, match="peer_capability"):
        mc2.build(op, replace(geo, datapath=replace(geo.datapath, ub_peer_edges=())))
    for active in (1, 2, peers - 1):
        op = call(DISPATCH, peers, **fields, transport_binding=binding,
                  assignment_matrix={(0, peer): 1 for peer in range(1, active + 1)})
        plan, ctx = lower_plan(op, geo, mc2_injection=("link", INJECTION_RATE * 1000),
                              mc2_peer=("link", RATE * 1000))
        pairs = [f"link:mesh:pair0>{peer}" for peer in range(1, peers)]
        injection = "link:mc2_injection:0"
        if declare_ports:
            injection = "link:mesh:tx:port0"
            rates = {**ctx.rates, **dict.fromkeys(pairs, RATE), injection: INJECTION_RATE}
            ctx = replace(ctx, rates=rates, resource_units=dict.fromkeys(rates, 1),
                          service_modes={"mc2_peer": ("link", RATE * 1000)})
        plan = pickle.loads(pickle.dumps(plan))
        sends = [stage for stage in plan.program(ctx).stages if injection in stage.work]
        assert plan.metadata["window_alignment_bytes"] == FRAME
        assert plan.metadata["peer_credits"] == peers - 1
        assert len(sends) == active
        _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, backend=backend)
        if backend == "c" and declare_ports:
            assert engine.summary()["native_steps"] > 0
        trace = timings(engine)
        assert len({trace[stage.name]["start_ms"] for stage in sends}) == 1
        for stage in sends:
            peer = pairs[stage.metadata["destination_rank"] - 1] if declare_ports else "link"
            assert stage.work == {peer: 2 * FRAME, injection: 2 * FRAME}
            assert stage.caps == {peer: RATE}
            assert stage.metadata["window_offset_bytes"] == FRAME
            transfer = trace[stage.name]
            assert transfer["end_ms"] - transfer["start_ms"] == pytest.approx(active * 2 * FRAME / INJECTION_RATE)
        assert engine.serviced[injection] == active * 2 * FRAME
    gemm = mc2.build(call(GATHER, peers, transport_abi=SHMEM, transport_binding=binding), geo)
    assert gemm.metadata["peer_payload_bytes"] == (peers - 1) * M * K * 2 / peers
    assert gemm.work.resources["link"] == (peers - 1) * FRAME

def test_credits_gate_each_peer_window():
    plan, ctx = lower_plan(call(DISPATCH, 4, width=ROW, peer_credits=2,
                                assignment_matrix={(0, peer): 1 for peer in range(1, 4)}),
                           geometry(MESH_SKU, 4))
    graph = stages(plan.program(ctx))
    assert "cohort0_status_1" not in ancestors(graph, "cohort0_send_2")
    assert "cohort0_status_1" in ancestors(graph, "cohort0_send_3")
    assert "cohort0_status_3" in ancestors(graph, "cohort0_slot_release")

@pytest.mark.parametrize("credits", (0, True, -2))
def test_invalid_peer_credit_limit_is_rejected(credits):
    with pytest.raises(UnmodeledCapabilityError, match="invalid_peer_credit_limit"):
        mc2.build(call(DISPATCH, 2, width=ROW, peer_credits=credits), geometry(MESH_SKU, 2))

@pytest.mark.parametrize("abi,operation", ((DISPATCH, "write_nbi"), (COMBINE, "read_nbi")))
@pytest.mark.parametrize("transport", (WINDOW, CCU))
def test_window_publication_and_consumption_precede_reuse(abi, operation, transport):
    groups = (mc2.Mc2CohortSlice(0, 0, 1, 1, 1, 1, 0),
              mc2.Mc2CohortSlice(1, 1, 2, 1, 1, 1, 0))
    matrix = ((0, 1), (1, 0))
    op = call(abi, 2, width=ROW, cohort_slices=groups, status_bytes=FLAG, transport_abi=transport,
              cohort_assignment_matrices={part.cohort_id: matrix for part in groups})
    plan, ctx = lower_plan(op, geometry(MESH_SKU, 2))
    graph = stages(plan.program(ctx))
    send, recv, visible, consume = (graph["cohort0_" + name] for name in
                                    ("send_1", "recv_link_1", "recv_1_ready", "recv_1_consume"))
    packet = FRAME * (1 if abi == COMBINE else 2) if transport == WINDOW else (
        ROW * 2 if abi == COMBINE else FRAME)
    status_bytes = FLAG if transport == WINDOW else (FRAME if abi == DISPATCH else 0)
    if status_bytes:
        status = graph["cohort0_status_1"]
        assert status.after == (send.name,) and status.work == {"link": status_bytes}
    assert recv.work == {"link": packet + status_bytes}
    assert visible.after == (recv.name,) and visible.work == {"hbm": packet}
    assert consume.after == (visible.name,) and consume.work == {"hbm": packet, "mte2_aiv": packet}
    assert consume.name in ancestors(graph, "cohort0_slot_release")
    assert "cohort0_slot_release" in graph["cohort1_pack"].after
    assert send.work == {"link": packet}
    if transport == WINDOW:
        assert graph[send.after[0]].work == {"mte3": packet}
        assert send.metadata["direct_remote_gm"] is (abi == COMBINE)
    else:
        assert send.metadata["transfer"] == operation
        assert consume.name in graph["cohort0_cross_core_ready"].after
        assert graph["cohort0_unpack"].after == ("cohort0_cross_core_ready",)
    py_span, py = execute(plan.lower(ctx), ctx.rates, ctx.resource_units)
    c_span, native = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, backend="c")
    assert native.summary()["native_steps"] > 0
    assert c_span == pytest.approx(py_span)
    assert native.serviced == pytest.approx(py.serviced)

@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("abi,producer,consumer", ((GATHER, "ag", "gemm"), (SCATTER, "gemm", "rs")))
def test_two_chunks_overlap_with_independent_slots(sku, abi, producer, consumer):
    geo = geometry(sku, 4)
    plan, ctx = lower_plan(call(abi, 4, transport_binding=CLIENT_BINDING, **CHUNK_SETTINGS), geo)
    graph = stages(plan.program(ctx))
    for chunk in (0, 1):
        send, use = f"{producer}_chunk{chunk}", f"{consumer}_chunk{chunk}"
        assert send in ancestors(graph, use)
        assert graph[f"slot_release_{chunk}"].after == (use,)
    assert f"{consumer}_chunk0" not in ancestors(graph, f"{producer}_chunk1")
    assert "slot_release_0" not in ancestors(graph, f"{producer}_chunk1")
    assert {p.slot for p in plan.pipeline_spec.phases if p.phase_id.startswith("slot_release")} == {0, 1}
    assert plan.work.matrix["bf16"] == 2 * M * K * N
    assert plan.work.resources["link"] == 3 * M * (K if abi == GATHER else N) * 2 / 4
    bad = call(abi, 4, transport_binding=CLIENT_BINDING, internal_chunks=2, independent_handles=False)
    with pytest.raises(UnmodeledCapabilityError, match="independent_buffers_and_handles"):
        mc2.build(bad, geo)

def test_cohort_helper_passes_parent_dependencies_and_preserves_directed_counts():
    part = mc2.Mc2CohortSlice(1, 4, 7, 3, 2, 2, 1, 1)
    ledger = mc2.Mc2Contract(DISPATCH, WINDOW, "fullmesh", (0, 1), 0, 1, False, False, "bf16",
        row_layout=mc2.dispatch_row_layout(ROW, input_dtype="bf16", payload_dtype="bf16", quant_mode="none"),
        cohorts=(part,), assignments=((1, 0, 0, 1), (1, 0, 1, 2), (1, 1, 0, 1)), status_bytes=FLAG)
    transport = SimpleNamespace(routes=(SimpleNamespace(source=0, destination=1, hops=("UB_WINDOW",)),),
        aiv_staging=True, credits=1, peer_bytes=lambda n: n, peer_limit=1, window_alignment=1,
        window_offset=0, direct_remote_gm=False, ccu=False, abi=WINDOW, incoming_routes=())
    cohort = {p.name: p for p in mc2.build_cohort_transport_stages(
        ledger, part, transport, after=("router",), peer_after=("prior_slot",))}
    assert cohort["cohort1_pack"].after == ("router",)
    assert cohort["cohort1_send_1_stage_ub"].after == ("cohort1_pack", "prior_slot")
    assert cohort["cohort1_send_1"].work.resources == {"link": 4 * FRAME}
    assert cohort["cohort1_recv_link_1"].after == ("router", "prior_slot")
    assert cohort["cohort1_recv_1_ready"].work.resources == {"hbm": 2 * FRAME}
    assert cohort["cohort1_unpack"].work.resources["mte3"] == 3 * (DATA + 12)

def test_cohort_plan_is_compact_and_rebinds_service_rates():
    width, rows = 481, 4096
    cells = {(0, 1): 1}
    op = call(DISPATCH, 2, width=width, payload_dtype="int8", quant_mode="per_token",
              assignment_matrix=cells)
    geo = geometry(MESH_SKU, 2)
    small, ctx = lower_plan(op, geo)
    cells[0, 1] = rows
    large = mc2.build(op, geo)
    assert tuple(stages(large.program(ctx))) == tuple(stages(small.program(ctx)))
    assert large.metadata["peer_payload_bytes"] == rows * small.metadata["peer_payload_bytes"]
    assert large.work.resources["vector"] == rows * 7 * width
    assert small.work.resources["vector"] == 7 * width
    slow, before = execute(large.lower(ctx), ctx.rates, ctx.resource_units)
    fast_ctx = replace(ctx, rates={role: rate * 2 for role, rate in ctx.rates.items()})
    fast, after = execute(large.lower(fast_ctx), fast_ctx.rates, fast_ctx.resource_units)
    assert slow == pytest.approx(2 * fast)
    assert after.serviced == pytest.approx(before.serviced)

@pytest.mark.parametrize("backend", ("python", "c"))
@pytest.mark.parametrize("transport", (WINDOW, CCU))
def test_receive_only_rank_waits_for_incoming_transfer(backend, transport):
    geo = geometry(MESH_SKU, 2)
    route = dict(source_rank=1, destination_rank=0, top_k_slot=0, destination_expert=0)
    routes = tuple(dict(route, source_token_id=token) for token in range(2))
    fields = dict(transport_abi=transport, source_tokens_by_rank={1: len(routes)}, shape=(0, ROUTING_WIDTH))
    op = call(DISPATCH, 2, routes=routes, **fields)
    for token, error in ((-1, "invalid_source_token_id"), (len(routes), "source_token_id_out_of_range")):
        bad = call(DISPATCH, 2, routes=(dict(route, source_token_id=token),), **fields)
        with pytest.raises(ModelingError, match=error):
            mc2.build(bad, geo)
    plan, ctx = lower_plan(op, geo)
    graph = stages(plan.program(ctx))
    _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, backend=backend)
    if backend == "c":
        assert engine.summary()["native_steps"] > 0
    trace = timings(engine)
    recv = graph["cohort0_recv_link_1"]
    payload, status = (2 * ROUTING_WIDTH * 2, FRAME) if transport == CCU else (2 * 18 * FRAME, FLAG)
    assert plan.metadata["peer_payload_bytes"] == 0
    assert recv.work["link"] == payload + status
    outgoing = FRAME if transport == CCU else 0
    assert engine.serviced["link"] == payload + status + outgoing
    assert (recv.metadata["source_rank"], recv.metadata["destination_rank"]) == (1, 0)
    assert (recv.metadata["payload_bytes"], recv.metadata["status_bytes"]) == (payload, status)
    assert "cohort0_send_1" not in graph
    for name in ("cohort0_recv_1_ready", "cohort0_recv_1_consume", "cohort0_unpack"):
        assert recv.name in ancestors(graph, name)
        assert trace[name]["start_ms"] >= trace[recv.name]["end_ms"]

def test_zero_payload_dispatch_exchanges_counts_with_each_peer():
    peers = 3
    geo = geometry(MESH_SKU, peers)
    for abi in (DISPATCH, COMBINE):
        plan, ctx = lower_plan(call(abi, peers, shape=(1, ROW), transport_abi=CCU,
                                   assignment_matrix={(rank, rank): 1 for rank in range(peers)}), geo)
        graph = stages(plan.program(ctx))
        _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, backend="c")
        assert engine.serviced["link"] == (2 * (peers - 1) * FRAME if abi == DISPATCH else 0)
        for peer in range(1, peers):
            sent, received = f"cohort0_status_{peer}", f"cohort0_recv_link_{peer}"
            if abi == DISPATCH:
                assert graph[sent].work == graph[received].work == {"link": FRAME}
                assert sent in ancestors(graph, "cohort0_slot_release")
                assert received in graph["cohort0_cross_core_ready"].after
            else:
                assert sent not in graph and received not in graph

@pytest.mark.parametrize("use_shape", (False, True))
def test_combine_returns_one_row_per_local_source_token(use_shape):
    tokens = (0, 1, 2, 3, 0, 1)
    count = len(set(tokens))
    routes = tuple(dict(source_rank=0, destination_rank=1, source_token_id=token,
                        top_k_slot=index // count, destination_expert=1)
                   for index, token in enumerate(tokens))
    fields = (dict(shape=(count, ROUTING_WIDTH), assignment_matrix={(1, 0): len(tokens)})
              if use_shape else dict(routes=routes))
    op = call(COMBINE, 2, width=ROUTING_WIDTH, top_k=2, **fields)
    plan, ctx = lower_plan(op, geometry(MESH_SKU, 2))
    assert stages(plan.program(ctx))["cohort0_unpack"].work["mte3"] == count * ROUTING_WIDTH * 2

@pytest.mark.parametrize("backend", ("python", "c"))
def test_interleaved_cohorts_use_exact_rows_and_restore_order(backend):
    slices = (mc2.Mc2CohortSlice(5, 0, 3, 2, 0, 2, 0, row_ids=(0, 2)),
              mc2.Mc2CohortSlice(9, 1, 4, 2, 0, 2, 0, row_ids=(1, 3)))
    routes = tuple(dict(cohort_id=part.cohort_id, source_rank=0, destination_rank=1,
                        source_token_id=row, top_k_slot=0, destination_expert=1)
                   for part in slices for row in part.row_ids)
    geo = geometry(MESH_SKU, 2)
    forward = call(DISPATCH, 2, width=ROW, routes=routes, cohort_slices=slices)
    plan, ctx = lower_plan(forward, geo)
    _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, backend=backend)
    trace = timings(engine)
    for part in slices:
        sent = trace[f"cohort{part.cohort_id}_send_1"]["metadata"]
        assert sent["cohort_row_ids"] == sent["cohort_restore_map"] == ",".join(map(str, part.row_ids))
        assert sent["cohort_compact_indices"] == "0,1"
    assert trace["cohort9_pack"]["start_ms"] >= trace["cohort5_slot_release"]["end_ms"]
    returns = tuple(replace(part, source_assignments=0, remote_assignments=0, received_assignments=2)
                    for part in slices)
    backward = mc2.build(call(COMBINE, 2, width=ROW, routes=routes, cohort_slices=returns), geo)
    outputs = {p.phase_id: next(w.amount for w in p.work if w.resource == "mte3")
               for p in backward.pipeline_spec.phases if p.phase_id.endswith("_unpack")}
    assert outputs == dict.fromkeys(("cohort5_unpack", "cohort9_unpack"), 2 * ROW * 2)
    wrong = ({**routes[0], "source_token_id": 1}, routes[1], {**routes[2], "source_token_id": 0}, routes[3])
    with pytest.raises(ModelingError, match="source_token_outside_cohort"):
        mc2.build(call(DISPATCH, 2, width=ROW, routes=wrong, cohort_slices=slices), geo)
    overlap = (slices[0], replace(slices[1], row_ids=(2, 3)))
    with pytest.raises(ModelingError, match="overlapping_cohort_rows"):
        mc2.build(call(DISPATCH, 2, width=ROW, routes=routes, cohort_slices=overlap), geo)

@pytest.mark.parametrize("row_ids", ((-1,), (0, 0), (True,), (0.5,), None))
def test_cohort_rejects_invalid_token_ids(row_ids):
    with pytest.raises(ModelingError, match="row_id"):
        mc2.Mc2CohortSlice(0, 0, 2, 0, 0, 0, 0, row_ids=row_ids)

@pytest.mark.parametrize("abi", (GATHER, SCATTER))
@pytest.mark.parametrize("shape", ({"m": 3}, {"m": 4, "gathered_bytes": 6},
                                   {"m": 8, "gathered_bytes": 4, "internal_chunks": 2}))
def test_collectives_require_integer_shard_rows_and_bytes(abi, shape):
    with pytest.raises(ModelingError, match="^mc2_uneven_shard_not_integer$"):
        mc2.build(call(abi, 4, **shape), geometry(MESH_SKU, 4))

def test_reduce_scatter_publishes_temp_gm_before_hccl_reads():
    geo = geometry(MESH_SKU, 4)
    assert geo.datapath.has_l0c_to_ub
    plan, ctx = lower_plan(call(SCATTER, 4), geo)
    graph = stages(plan.program(ctx))
    gemm, read, link = (graph[name] for name in ("gemm_chunk0", "rs_chunk0_hccl_read", "rs_chunk0_peer_1"))
    assert gemm.work["fixpipe"] == M * N * 2
    assert gemm.work["hbm"] == (M * K + K * N + M * N) * 2
    assert read.work["hbm"] == 3 * (M // 4) * N * 2
    assert link.work["link"] == (M // 4) * N * 2
    assert gemm.name in ancestors(graph, read.name)
    assert read.name in ancestors(graph, link.name)
    _, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units)
    trace = timings(engine)
    assert trace[gemm.name]["end_ms"] <= trace[read.name]["start_ms"]
    assert trace[read.name]["end_ms"] <= trace[link.name]["start_ms"]

NORMAL = {"kind": "arch35_fullmesh", "dispatch_quant": "none"}
V2_NATIVE = {"kind": "arch22_v2_caller", "dispatch_quant": "per_token_int8",
             "max_inflight_per_rank": 2, "global_bs": 6}


def caller_case(tp=1, dp=2, protocol=NORMAL):
    raw = hardware_case()
    raw["machine"].update(num_devices=tp * dp, links=[dict(name="fabric", capacity_bytes_per_ms=1e6,
        members=list(range(tp * dp)), semantics="per_pair", latency_us=5)])
    device = raw["machine"]["device"]
    device["compute_peaks"].update(bf16=FAST, fp32=FAST, int8=FAST)
    device["rates"].update(memory=FAST, vector=FAST, mte2_aic=FAST, mte2_aiv=FAST, mte3=FAST)
    device["resource_units"] = dict(vector=4, mte2_aiv=4, mte3=4)
    raw["model"].update(hidden_size=32, num_heads=max(4, tp), num_kv_heads=max(4, tp), head_dim=8,
        ffn="moe", num_experts=16 if tp > 1 else 4, top_k=2, moe_intermediate_size=32)
    kernels = {"native_moe_protocol": dict(protocol)} if protocol else {}
    if tp > 1:
        raw["model"]["num_shared_experts"] = 1
        kernels["moe_token_partition"] = {"kind": "vllm_ascend_mc2_tp_slice"}
    raw["runtime"].update(dp=dp, tp=tp, ep=tp * dp, kernels=kernels, routing={"policy": "balanced"})
    return raw

@pytest.mark.parametrize("tp,rows", ((2, 2), (4, 8), (4, 64)))
def test_tensor_partition_preserves_full_router_expert_work_and_one_gather(tp, rows):
    raw = caller_case(tp, dp=1)
    opts = dict(phase="decode", tokens=1, sequences=rows, context=128, scope="body")
    declared = target_graphs(raw, **opts)
    ops = {op.op_id.removeprefix("s0.l0."): op for op in declared[0]}
    assert ops["router"].local_shape[0] == ops["shared.gate_up"].local_shape[0] == rows
    assert ops["topk"].local_shape == (rows // tp, 16) and ops["topk"].deps == (ops["mc2_logits_view"].op_id,)
    assert all(ops[name].hbm_bytes == 0 for name in ("mc2_input_view", "mc2_logits_view"))
    ledger = dict(ops["expert_up"].metadata)["routing_realization"]
    assert sum(ledger["global_expert_rows"]) == rows * 2
    assert ops["expert_up"].flops * tp == 2 * rows * 2 * 64 * 32
    gather = ops["mc2_tp_gather"]
    assert gather.deps == (ops["unpack"].op_id,) and gather.local_shape == (rows, 32)
    assert gather.link_bytes == (tp - 1) * (rows // tp) * 32 * 2
    assert set(ops["join"].deps) == {gather.op_id, ops["shared.tp"].op_id}
    assert sum(o.family == "allgather" for o in ops.values()) == 1
    kernels = raw["runtime"]["kernels"]
    kernels["moe_token_partition"].update(activation_tensor_rows=rows * 2, activation_owners=4)
    kernels["hccl"] = {"allgather": {"expansion": "AICPU_TS", "template": "mesh", "axes": ["tp"]}}
    capped = {o.op_id.removeprefix("s0.l0."): o for o in target_graphs(raw, **opts)[0]}
    assert capped["expert_act"].local_shape[0] == rows * 2
    selection = dict(capped["mc2_tp_gather"].metadata)["hccl_selection"]
    assert (selection["payload_bytes"], selection["ranks"]) == (rows // tp * 32 * 2, tp)
    for name in ("expert_up", "expert_down", "dispatch", "unpack"):
        actual, original = capped[name], ops[name]
        assert (actual.flops, actual.hbm_bytes, actual.link_bytes) == (
            original.flops, original.hbm_bytes, original.link_bytes)
    raw["runtime"]["kernels"] = {}
    baseline = {op.op_id: op for op in target_graphs(raw, **opts)[0]}
    for name in ("router", "expert_up", "expert_act", "expert_down", "shared.gate_up", "shared.tp"):
        op, other = ops[name], baseline[ops[name].op_id]
        assert (op.flops, op.hbm_bytes, op.vector_ops) == (other.flops, other.hbm_bytes, other.vector_ops)

@pytest.mark.parametrize("protocol,width", ((NORMAL, 32), (V2_NATIVE, 2048), (None, 32)))
@pytest.mark.parametrize("sequences", ((1, 1), (0, 3), (1, 3)))
def test_native_joint_calls_charge_each_channel_once_and_gate_consumer(protocol, width, sequences):
    legacy = protocol == V2_NATIVE
    raw = caller_case(protocol=protocol)
    if protocol is None:
        raw["machine"].update(architecture="A5", sku=MESH_SKU)
        raw["runtime"].update(max_num_seqs=64, max_num_batched_tokens=8192, graph_buckets=[1, 2, 4, 8, 16, 32, 64],
                              metadata={"arch_family": "A5", "additional_config": {"enable_fused_mc2": 0}})
    raw["model"].update(hidden_size=width, head_dim=width // 4)
    raw["machine"]["device"]["launch_us"] = 3
    raw["runtime"]["kernels"]["invocations"] = [
        {"match": {"kernel": f"MoeDistribute{kind}{'V2' if legacy else 'V3'}"}, "minimum_ms": .030}
        for kind in ("Dispatch", "Combine")]
    parsed = parse_case(raw)
    if legacy:
        parsed = replace(parsed, model=replace(parsed.model, quant=QuantSpec(
            "row-int8", weight_format="int8", expert_weight_format="int8",
            activation_format="int8", scale_format="fp32", scale_bytes=4,
            exclusions=("router", "norm"))))
    graphs = target_graphs(parsed, phase="decode", tokens=1, sequences=sequences, context=128, scope="body")
    result = graph_costs(parsed, graphs, external_inputs={"analysis_input": RELEASE},
                         input_ready_ms=RELEASE, isolated=False, include_submission=True)
    x = result["execution"]
    expected = sum(4 if protocol is None and rows == 3 else rows for rows in sequences)
    expected *= width + 4 + 12 + width * 2 if legacy else 2 * FRAME
    expected += 16 if legacy else 4 * FLAG
    assert sum(v for k, v in x["serviced"].items() if k.startswith("link:")) == expected
    assert x["live_tasks"] == x["live_programs"] == 0
    timeline = {row["id"]: row for row in result["timeline"]}
    assert min(row["ready_ms"] for row in timeline.values()) >= RELEASE
    if not legacy:
        native = x["native_groups"]
        assert len(native) == 2
        for direction in ("tx", "rx"):
            assert sum(g[f"transport_{direction}_bytes"] for g in native) == expected
    for rank, graph in graphs.items():
        indexed = {op.op_id: op for op in graph}
        dispatch, gate = (indexed["s0.l0." + name] for name in ("dispatch", "expert_up"))
        assert (dict(dispatch.metadata).get("moe_comm_selection", {}).get("mode") == "dynamic") is (protocol is None)
        group = [dict(o.metadata)["device_kernel_group"] for o in graph if "device_kernel_group" in dict(o.metadata)]
        assert sum(g["entry"] for g in group) == sum(g["exit"] for g in group) == 2
        prefix = f"r{rank}:"
        assert timeline[prefix + gate.op_id]["ready_ms"] >= timeline[prefix + dispatch.op_id]["complete_ms"]
        if legacy:
            meta = dict(dispatch.metadata)
            assert meta["activation_scale_shape"] == (sum(sequences),)
            assert meta["activation_scale_bytes"] == 4 * sum(sequences)
            assert meta["assist_info_shape"] == (sum(sequences), 3)
            assert {"activation_scales", "assist_info", "expert_token_nums", "ep_send_counts", "expand_scales"} <= set(dispatch.returns)
            assert dict(indexed["s0.l0.combine"].metadata)["expand_scales_buffer_producer"] == dispatch.op_id

@pytest.mark.parametrize("parallel", ({"tp": 1, "ep": 1}, {"dp": 2}, {"ep": 8}))
def test_partition_requires_matching_expert_and_tensor_placement(parallel):
    fields = {"tp": 4, "ep": 4, "dp": 1, "etp": 1, "cp": 1, "pcp": 1, **parallel}
    runtime = SimpleNamespace(kernels={"moe_token_partition": {"kind": KIND, "padding": "right_to_tp"}},
                              moe_recipe="alltoall")
    placement = dict(parallel=SimpleNamespace(**fields), plan=SimpleNamespace(scope="body", expert_latent_size=0),
                    model=SimpleNamespace(hidden_size=32), tokens=1)
    if fields["tp"] == 1:
        assert select_partition(runtime, **placement) is None
    else:
        with pytest.raises(ModelingError, match="dp1_ep_equals_tp"):
            select_partition(runtime, **placement)
