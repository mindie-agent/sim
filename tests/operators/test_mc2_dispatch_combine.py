"""MC2 routing, caller boundaries, and executable transport contracts.

[Role] Exercise the governed MC2 entrypoint and the ordinary MoE caller.
[Invariants] Directed payload, matrix work, ownership and causal edges survive
  planning, binding and execution across declared architectures.
[Timeouts & Locks] Finite analytical graphs with bounded engine traces.
[Side-Effect Policy] No device access. C11 uses the isolated test cache.
"""
from collections import Counter
from dataclasses import replace
from math import ceil
import pickle
from types import SimpleNamespace

import pytest
from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import hardware_case, machine

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_graph import NETWORK, lower, recipes, select
from mindie_sim.prediction.mc2_phase import build as phase_build, contract
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_token_partition import KIND, select as select_partition
from mindie_sim.prediction.operators import mc2
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.mc2_fullmesh import assignment_prefix_work, dispatch_layout, dispatch_plan
from mindie_sim.prediction.operators.registry import FamilyDispatcher, FamilyRegistry
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
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
M, K, N = 20, 32, 48
CHUNK_SETTINGS = dict(internal_chunks=2, independent_buffers=True, independent_handles=True)
RELEASE, RATE, FAST = 7, 64000, 1e9
ROLES = ("hbm", "l2", "mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe", "vector", "link", "cube")
BASE_SKU, DUAL_SKU, MESH_SKU = REPRESENTATIVE_SKUS.values()

def call(abi, peers, **fields):
    shape = (M, K, N) if abi in (GATHER, SCATTER) else ()
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

def execute(program, rates, units, *, backend="python", external=None):
    engine = Engine([Resource(r, rate, units=units[r]) for r, rate in rates.items()],
                    trace_limit=10000, backend=backend)
    ready = engine.submit("input", release=RELEASE)
    if external:
        engine.submit("rival", work={"scalar_ub_access": external},
                      placement={"scalar_ub_access": (0,)}, release=RELEASE)
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

@pytest.mark.parametrize("arch", ARCH_IDS)
def test_foreign_transport_is_rejected_at_public_boundary(arch):
    sku = MESH_SKU if arch == ARCH_IDS[-1].values[0] else BASE_SKU
    geo = geometry(sku)
    invalid = (CLIENT,) if sku == MESH_SKU else (SHMEM, WINDOW, CCU)
    for transport in invalid:
        with pytest.raises(UnmodeledCapabilityError, match="transport_unsupported") as failure:
            mc2.build(call(DISPATCH, 1, width=ROW, transport_abi=transport), geo)
        assert failure.value.onboarding_hint["schema"] == "mindie-unmodeled-structure-hint/1"

def test_direct_mesh_alignment_credits_and_shared_injection():
    peers, credits = 8, 2
    geo = geometry(MESH_SKU, peers)
    fields = dict(transport_abi=SHMEM, width=ROW, status_bytes=0, peer_credits=credits,
                  assignment_matrix={(src, dst): 1 for src in range(peers) for dst in range(peers)})
    binding = {"caller_abi": "shmem", "window_offset_bytes": FRAME}
    for change, error in (({}, "shmem_caller"), ({**binding, "window_offset_bytes": 1}, "aligned_window")):
        with pytest.raises(UnmodeledCapabilityError, match=error):
            mc2.build(call(DISPATCH, peers, **fields, transport_binding=change), geo)
    op = call(DISPATCH, peers, **fields, transport_binding=binding)
    with pytest.raises(UnmodeledCapabilityError, match="peer_capability"):
        mc2.build(op, replace(geo, datapath=replace(geo.datapath, ub_peer_edges=())))
    injection = FRAME * 1000
    plan, ctx = lower_plan(op, geo, mc2_injection=("link", injection), mc2_peer=("link", RATE * 1000))
    graph = stages(plan.program(ctx))
    send = [s for s in graph.values() if s.work.get("link")]
    assert plan.metadata["window_alignment_bytes"] == FRAME
    assert plan.metadata["peer_credits"] == credits
    assert {(s.metadata["source_rank"], s.metadata["destination_rank"]) for s in send} == {(0, p) for p in range(1, peers)}
    for i, s in enumerate(send):
        assert s.work["link"] == 2 * FRAME
        assert s.metadata["window_offset_bytes"] == FRAME
        assert s.caps["link"] == pytest.approx(injection / 1000 / (peers - 1))
        if i >= credits:
            assert send[i - credits].name + "_consume" in s.after
    elapsed, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units)
    assert engine.serviced["link"] == 2 * FRAME * (peers - 1)
    assert elapsed >= engine.serviced["link"] / (injection / 1000)
    gemm = mc2.build(call(GATHER, peers, transport_abi=SHMEM, transport_binding=binding), geo)
    assert gemm.metadata["peer_payload_bytes"] == (peers - 1) * M * K * 2 / peers
    assert gemm.work.resources["link"] == (peers - 1) * FRAME

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
    send, visible, status, consume = (graph["cohort0_" + name] for name in
                                      ("send_1", "send_1_ready", "status_1", "send_1_consume"))
    packet = FRAME * (1 if abi == COMBINE else 2)
    assert visible.after == (send.name,) and visible.work == {"hbm": packet}
    assert status.after == (visible.name,) and status.work == {"link": FLAG}
    assert consume.after == (status.name,) and consume.work == {"hbm": packet, "mte2_aiv": packet}
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
        window_offset=0, direct_remote_gm=False, ccu=False)
    cohort = {p.name: p for p in mc2.build_cohort_transport_stages(
        ledger, part, transport, after=("router",), peer_after=("prior_slot",))}
    assert cohort["cohort1_pack"].after == ("router",)
    assert cohort["cohort1_send_1_stage_ub"].after == ("cohort1_pack", "prior_slot")
    assert cohort["cohort1_send_1"].work.resources == {"link": 4 * FRAME}
    assert cohort["cohort1_send_1_ready"].work.resources == {"hbm": 2 * FRAME}
    assert cohort["cohort1_unpack"].work.resources["mte3"] == 3 * (DATA + 12)

def phase_inputs(phase, *, kind="dispatch", tokens=1, top_k=8, units=64, experts=16,
                 ep=8, received=None, width=ROW, roles=(), issue_units=None, supply=1, copy_units=None, rate=RATE):
    counts = tokens * top_k
    other = counts if received is None else received
    raw = dict(kind=kind, width=width, dtype="bf16", source_tokens=tokens, top_k=top_k,
               source_assignments=counts if kind == "dispatch" else other,
               received_assignments=other if kind == "dispatch" else counts, experts=experts, ep=ep, phase=phase)
    op = replace(call(DISPATCH, 1), family="mc2_protocol", recipe="", metadata=(("mc2_phase", raw),))
    physical = dict(hbm=1, mte2_aiv=copy_units or units, mte3=copy_units or units, vector=units)
    physical.update(dict.fromkeys(roles, issue_units or units))
    rates = dict(hbm=max(FAST, rate), mte2_aiv=rate * supply, mte3=rate * supply, vector=rate * supply)
    rates.update({r: supply * physical[r] for r in roles})
    context = RecipeContext(op, {}, {r: r for r in rates}, rates, units, resource_units=physical)
    return raw, op, Geometry(units, physical), context

def run_phase(phase, *, external=None, **kwargs):
    raw, op, geo, ctx = phase_inputs(phase, **kwargs)
    plan = phase_build(op, geo)
    span, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units, external=external)
    expected = Counter(plan.work.resources)
    if external:
        expected["scalar_ub_access"] += external
    assert +Counter(engine.serviced) == pytest.approx(+expected)
    return span, engine, plan, {p.name: p for p in contract(raw, geo.compute_units).phases}

PREFIX_FACTOR, STATUS_WORK = 5, 9
ISSUE = ("scalar_ub_access", "scalar_alu_issue", "pipe_barrier")

@pytest.mark.parametrize("tokens,top_k,units", ((0, 8, 64), (1, 8, 56), (2, 3, 8),
                                               (4, 8, 64), (17, 5, 16), (128, 8, 56)))
def test_strided_prefix_and_instruction_work_follow_assignment_indices(tokens, top_k, units):
    def oracle(owners):
        expected = dict.fromkeys(range(owners), 0)
        for index in range(top_k, tokens * top_k):
            expected[index % owners] += PREFIX_FACTOR * index - 1
        return expected

    assert dict(assignment_prefix_work(tokens, top_k, tokens * top_k, tuple(range(units)))) == oracle(units)
    _, engine, plan, phases = run_phase("send_read_quantize", tokens=tokens, top_k=top_k, units=units)
    count = phases["count_compute"]
    count_owners = len(count.owner_domain)
    payload = phases["send_read_quantize"]
    assert set(count.owner_domain).isdisjoint(payload.owner_domain)
    assert set(count.owner_domain + payload.owner_domain) == set(range(units))
    expected = oracle(len(payload.owner_domain))
    assert engine.serviced["vector"] == pytest.approx(sum(expected.values()))
    assert engine.serviced["hbm"] == pytest.approx(tokens * top_k * DATA)
    assert dict(count.work)["vector"] == count_owners * STATUS_WORK * 16
    for owner in plan.metadata["active_owner_domain"]:
        own = [r for r in engine.trace if f"owner{owner}:" in r["task"]]
        for scan in (r for r in own if r["task"].endswith(":assignment_prefix_scan")):
            assert scan["end_ms"] - scan["start_ms"] == pytest.approx(expected[owner] * units / RATE)
            assert next(r for r in own if r["task"].endswith(":read"))["start_ms"] >= scan["end_ms"]
    _, priced, plan, _ = run_phase("send_read_quantize", tokens=tokens, top_k=top_k, units=units, roles=ISSUE)
    scanned = max(0, (tokens - 1) * top_k)
    assert priced.serviced["scalar_ub_access"] == pytest.approx(tokens * top_k + scanned)
    assert priced.serviced["pipe_barrier"] == pytest.approx(4 * scanned)
    for owner, entries in plan.metadata["primitive_services"]["owners"].items():
        indices = range(owner, tokens * top_k, len(payload.owner_domain))
        scanned_owner = sum(index >= top_k for index in indices)
        assert sum(x["counts"].get("scalar_ub_access", 0) for x in entries.values()) == len(indices) + scanned_owner

def test_retained_counts_do_not_invent_prefix_indices_and_large_scans_stay_compact():
    row = dispatch_layout(ROW)
    for tokens, retained in ((2, 3), (2, .5), (.5, 4)):
        plan = dispatch_plan(row, source_tokens=tokens, top_k=8, source_assignments=retained,
                             received_assignments=8, experts=16, ep=8, aiv_count=64)
        phase = next(p for p in plan.phases if p.name == "send_read_quantize")
        assert dict(phase.details)["assignment_prefix_scan"]["owner_vector_work"] is None
        assert dict(phase.work)["hbm"] == retained * DATA
        assert any("retained valid assignment indices" in item for item in plan.unmodeled_features)
    results = [run_phase("send_read_quantize", tokens=4096, supply=f) for f in (.5, 1, 2)]
    assert results[0][0] > results[1][0] > results[2][0]
    for _, engine, _, _ in results:
        assert engine.summary()["completed_tasks"] < 4 * 64
        assert engine.serviced["vector"] == sum(PREFIX_FACTOR * i - 1 for i in range(8, 4096 * 8))
    plan = run_phase("send_read_quantize", tokens=4)[2]
    entries = [x for owner in plan.metadata["primitive_services"]["owners"].values() for x in owner.values()]
    assert sum(x["unpriced_service_work"].get("scalar_ub_access", 0) for x in entries) == 56

@pytest.mark.parametrize("experts,units,ep,owners", ((16, 4, 4, 1), (100, 16, 4, 6), (384, 64, 32, 8)))
def test_idle_senders_publish_counts_and_prefix_to_every_owner(experts, units, ep, owners):
    opts = dict(tokens=0, experts=experts, units=units, ep=ep, roles=ISSUE)
    _, counts, _, phases = run_phase("count_compute", **opts)
    _, prefix, _, _ = run_phase("prefix_counts", **opts)
    assert counts.serviced["scalar_ub_access"] == experts
    assert counts.serviced["pipe_barrier"] == owners
    assert prefix.serviced["scalar_alu_issue"] == experts
    assert prefix.serviced["scalar_ub_access"] == pytest.approx(2 * experts + 4 * owners + 2 * (owners - 1))
    assert prefix.serviced["pipe_barrier"] == 2 * owners - 1
    assert not any(dict(phases["routing_metadata"].work).values())
    assert dict(phases["receive_counts"].work)["mte3"] == (experts + owners**2) * FLAG
    assert dict(phases["prefix_counts"].work)["mte3"] == (owners**2 + owners * units) * FLAG + (units + 1) * experts * 4
    assert dict(phases["prefix_ready"].work)["mte2_aiv"] == owners * units * FLAG
    assert dict(phases["receive_metadata"].work)["mte2_aiv"] == units * experts * 4
    assert dict(phases["expert_counts"].work)["mte3"] == experts // ep * 8
    assert dict(phases["count_status_exchange"].work) == {"mte3": experts * FLAG, "hbm": experts * FLAG}

@pytest.mark.parametrize("tokens,top_k,width", ((1, 1, ROW), (3, 8, ROW), (65, 3, ROW), (1, 8, 3584)))
def test_successful_flag_poll_is_ordered_and_separate_from_slot_payload(tokens, top_k, width):
    opts = dict(kind="combine", tokens=tokens, top_k=top_k, width=width)
    span, engine, plan, phases = run_phase("receive_flags", **opts)
    blocks = ceil(width * 2 / DATA)
    compared = ceil(top_k * blocks * 8 / 64) * 64
    assert engine.serviced["vector"] == tokens * 2 * compared
    assert engine.serviced["mte2_aiv"] == pytest.approx(tokens * top_k * blocks * FLAG)
    details = plan.metadata["source_service_details"]
    assert set(details["scalar_instruction_counts"].values()) == {tokens * compared // 64}
    assert span == pytest.approx(ceil(tokens / 64) * (2 * compared + top_k * blocks * FLAG) / (RATE / 64))
    own = [r for r in engine.trace if "owner0:" in r["task"]]
    init, read, check = (next(r for r in own if r["task"].endswith(":" + s))
                         for s in ("initialize_flags", "read_flags", "compare_flags"))
    assert init["end_ms"] <= read["start_ms"] <= read["end_ms"] <= check["start_ms"]
    assert not phases["receive_slots"].work
    assert dict(phases["weighted_reduce"].work)["mte2_aiv"] == tokens * top_k * (4 + blocks * DATA)
    assert phases["clear_data_flags"].after == ("weighted_reduce",)
    assert dict(phases["publish_output"].work)["mte3"] == tokens * width * 2

@pytest.mark.parametrize("tokens,units", ((1, 2), (1.25, 2), (3, 2), (8.75, 4), (65, 64), (257, 2)))
def test_token_reduction_matches_eager_arithmetic_and_bounds_live_work(tokens, units):
    top_k = 3
    _, op, geo, ctx = phase_inputs("weighted_reduce", kind="combine", tokens=tokens, top_k=top_k, units=units)
    lazy, engine = execute(phase_build(op, geo).lower(ctx), ctx.rates, ctx.resource_units)
    eager = []
    active = min(units, ceil(tokens))
    base, tail = divmod(tokens, active)
    for owner in range(active):
        rows = base + min(1, max(0, tail - owner))
        previous = ()
        terms = [("mte2_aiv", rows * top_k * 4)]
        for token in range(ceil(rows)):
            fraction = min(1, rows - token)
            terms += [("vector", fraction * ROW)]
            terms += [("mte2_aiv", fraction * DATA), *(("vector", fraction * ROW),) * 3] * top_k
        for index, (role, amount) in enumerate(terms):
            demand = {role: amount, **({"hbm": amount} if role == "mte2_aiv" else {})}
            name = f"owner{owner}:{index}"
            eager.append(Stage(name, demand, after=previous, placement=ctx.owned(owner, units, (role,))))
            previous = (name,)
    expected, control = execute(KernelProgram(tuple(eager)), ctx.rates, ctx.resource_units)
    assert lazy == pytest.approx(expected, abs=1e-11)
    assert +Counter(engine.serviced) == pytest.approx(+Counter(control.serviced))
    assert engine.serviced["vector"] == tokens * (3 * top_k + 1) * ROW
    if tokens > 100:
        assert engine.summary()["peak_live_tasks"] < 100
        assert control.summary()["peak_live_tasks"] > 1000

def test_scalar_pipe_contention_and_local_copy_obey_physical_owner_supply():
    opts = dict(kind="combine", tokens=2, top_k=3, units=4, issue_units=2, roles=ISSUE[:1], rate=1e12)
    ordinary, _, _, _ = run_phase("weighted_reduce", **opts)
    rival, engine, _, _ = run_phase("weighted_reduce", **opts, external=6)
    assert ordinary == pytest.approx(6, abs=1e-6) and rival == pytest.approx(9, abs=1e-6)
    assert next(row for row in engine.trace if row["task"] == "rival")["end_ms"] == pytest.approx(19)
    assert all(row["placement"]["scalar_ub_access"] == (0,) for row in engine.trace if "primitive" in row["task"])
    for factor in (.5, 1, 2):
        span, _, plan, _ = run_phase("receive_local_copy", received=1, supply=factor)
        assert span == pytest.approx((2 * DATA + DATA + 12) / (1000 * factor))
        assert plan.metadata["active_owner_domain"] == (0,)

def test_phase_templates_snapshot_geometry_and_bind_fresh_rates():
    raw, op, geo, ctx = phase_inputs("weighted_reduce", kind="combine", tokens=8, top_k=6, width=5120, roles=ISSUE)
    plan = phase_build(op, geo)
    assert phase_build(replace(op, op_id="another_layer", layer=1), geo) is plan
    slow, engine = execute(plan.lower(ctx), ctx.rates, ctx.resource_units)
    faster = replace(ctx, rates={role: value * 2 for role, value in ctx.rates.items()})
    fast, replay = execute(plan.lower(faster), faster.rates, faster.resource_units)
    assert slow == pytest.approx(2 * fast)
    assert engine.serviced == pytest.approx(replay.serviced)
    raw.update(source_tokens=16, source_assignments=96, received_assignments=96)
    assert phase_build(op, geo).work.resources["hbm"] == 2 * plan.work.resources["hbm"]
    entry = next(iter(plan.metadata["primitive_services"]["owners"][0].values()))
    with pytest.raises(TypeError):
        entry["counts"]["scalar_ub_access"] = 0
    smaller = phase_build(op, Geometry(4, {r: 4 for r in ROLES if r != "cube"}))
    assert len(smaller.metadata["active_owner_domain"]) == 4
    bounded = replace(op, metadata=(("mc2_phase", dict(raw, source_tokens=1_000_000.25,
                    source_assignments=6_000_001.5, received_assignments=6_000_001.5)),))
    compact = phase_build(bounded, geo).program(ctx)
    tiles = [s.program for s in compact.stages if s.program is not None]
    assert len(tiles) == 64
    assert all(p.tile(0) is p.tile(p.count // 2) for p in tiles)
    assert sum(p.tile(0) is not p.tile(p.count - 1) for p in tiles) == 1

NORMAL = {"kind": "arch35_fullmesh", "dispatch_quant": "none"}
V2_NATIVE = {"kind": "arch22_v2_caller", "dispatch_quant": "per_token_int8",
             "max_inflight_per_rank": 2, "global_bs": 6}

@pytest.mark.parametrize("window", (None, 4))
def test_protocol_selection_keeps_explicit_credit_override(window):
    choice = dict(NORMAL) if window is None else dict(NORMAL, max_inflight_per_rank=window)
    declaration = {"native_moe_protocol": choice}
    assert select(declaration) == choice
    assert recipes(declaration)[NETWORK].max_inflight_per_rank == window
    assert select({"native_moe_protocol": V2_NATIVE}) == V2_NATIVE
    with pytest.raises(ModelingError):
        lower([], runtime=SimpleNamespace(kernels=declaration, moe_recipe="alltoall"),
              prefix="p", tokens=1, top_k=2, width=ROW, experts=16, ep=1,
              source_partitions=1, received_assignments=2, dtype="bf16")

@pytest.mark.parametrize("value", (0, True, -2))
def test_invalid_native_peer_window_is_rejected(value):
    for invalid in (dict(NORMAL, max_inflight_per_rank=value), dict(NORMAL, extra=1),
                    dict(NORMAL, dispatch_quant="int4"), dict(V2_NATIVE, dispatch_quant="none"),
                    {"kind": "unknown"}):
        with pytest.raises(ModelingError):
            select({"native_moe_protocol": invalid})

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
