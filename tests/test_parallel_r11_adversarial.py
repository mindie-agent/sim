"""Round-11 absolute ownership, physical replicas and stage-exact regressions.

[Role]
  Exercise all five R11 defects through independent byte/owner oracles and
  public admission, including heterogeneous windows and parallel shared drafts.
[Invariants]
  Every destination copy and physical message is counted; cache offsets do
  not renumber pages. Per-rank memory and per-stage calls follow actual work.
[Timeouts & Locks] Bounded synthetic arithmetic; no subprocesses or locks.
[Side-Effect Policy] Immutable Cases and fresh ledgers only; no file writes.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import math

import pytest

from test_parallel_r7_adversarial import _dense, _draft, _link, _machine_with, _sparse, _speculative
from test_parallel_r8_adversarial import _facts
from test_parallel_r10_adversarial import _instance, _compose, _bridge, _solve
from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.parallel_bridge import bridge_lifetime_plan, bridge_remap_bytes, phase_bridge
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_placement import build_placement, with_world
from mindie_sim.parallel_prefill_cost import prefill_step_ms
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_draft import priced_draft_step_ms, shared_target_draft
from mindie_sim.xpyd_transfer import logical_transfer_bytes, kv_transfer_contention


def _slow_last(case):
    n = len(case.machine.dies)
    link = replace(_link("one", range(n)),
                   endpoint_rates=tuple((r, 1000.0 if r == n - 1 else 1e9) for r in range(n)))
    return replace(case, machine=_machine_with(case, [link]))


def _exact_lifetime(case, p, d, transfer, eplb=0):
    bridge = _bridge(p, d, transfer, layer_extents=transfer["layer_extents"],
                     replicated=d.pcp_lane_policy == "replica_lanes")
    return bridge_lifetime_plan(bridge, case=case, transfer=transfer,
                                eplb_rebalance_peak_bytes=eplb)


def test_r11_01_all_540_absolute_owner_cases():
    case = _dense(layers=1, kv_heads=1, num_devices=64, max_seqs=1)
    p_specs = {pcp: ParallelSpec(pcp=pcp) for pcp in (1, 2, 3)}
    p_insts = {pcp: _instance(case, p_specs[pcp], range(pcp)) for pcp in (1, 2, 3)}
    d_specs = {(cp, dp): ParallelSpec(tp=cp, cp=cp, dp=dp)
               for cp, dp in product((1, 2, 4), (1, 2, 3))}
    d_insts = {
        (cp, dp, pcp): _instance(case, d_specs[cp, dp], range(pcp, pcp + d_specs[cp, dp].world_size))
        for cp, dp, pcp in product((1, 2, 4), (1, 2, 3), (1, 2, 3))
    }
    transfers = {
        (offset, count): logical_transfer_bytes(
            case, prompt_tokens=(offset + count) * 128,
            destination_cached_tokens=offset * 128, prefix_digest="d" * 64)
        for offset, count in product(range(4), range(1, 6))
    }
    checks = 0
    for cp, dp, pcp, offset, count in product((1, 2, 4), (1, 2, 3), (1, 2, 3),
                                             range(4), range(1, 6)):
        p, d = p_specs[pcp], d_specs[cp, dp]
        transfer = transfers[offset, count]
        bridge = _bridge(p, d, transfer)
        result = _compose(case, [p_insts[pcp]], [d_insts[cp, dp, pcp]], bridge, transfer)
        assert result["qualified"], result
        for replica, lane in product(range(dp), range(cp)):
            rank = pcp + replica * cp + lane
            pages = sum(page % cp == lane for page in range(offset, offset + count))
            assert result["handoff_demands"].get(f"port:{rank}:ingress", 0) == pytest.approx(
                pages * (65_536 + 64) / dp)
        for lane in range(pcp):
            pages = sum(page % pcp == lane for page in range(offset, offset + count))
            assert result["handoff_demands"].get(f"port:{lane}:egress", 0) == pytest.approx(pages * 65_600)
        assert result["wire_bytes"] == pytest.approx(transfer["transfer_total_bytes"])
        checks += 1
    assert checks == 540


def test_r11_01_cached_page_port_admission_and_unstable_rate():
    case = _slow_last(_dense(layers=4, kv_heads=1, num_devices=3, max_seqs=1))
    result = _solve(case, d=ParallelSpec(tp=2, cp=2), isl=256, osl=4,
                     destination_cached_tokens=128, prefix_digest="d" * 64)
    best = result["best_configuration"]
    assert result["qualified"]
    assert best["remap_bridge"]["destination_pages"] == ((), (1,))
    assert best["handoff_cut_demands"]["port:2:ingress"] == 262_400
    demand = 262_400 + 99_072
    assert best["shared_cut_demands"]["port:2:ingress"] == demand
    assert best["admitted_rps"] == pytest.approx(.9 * (1 - 1e-9) * 1e6 / demand)
    rho = best["admitted_rps"] * demand / 1e6
    assert best["transfer_queue_ms"] == pytest.approx(rho * demand / 1000 / (2 * (1 - rho)))
    old = kv_transfer_contention(262_400, n_prefill=1, n_decode=1,
        nic_bw_gbps=.001, hbm_bw_bytes_per_s=1e12, p_world=1, d_world=2,
        decode_step_ms=1, osl=4, requests_per_s=3.429878048784299,
        port_demands={"port:2:ingress": demand}, port_capacities={"port:2:ingress": 1e6})
    assert not old["feasible"] and math.isinf(old["queue_ms"])


def test_r11_01_explicit_evidence_order_never_changes_owners_or_allows_incomplete_plans():
    p, d = ParallelSpec(pcp=3), ParallelSpec(tp=4, cp=4)
    first = phase_bridge(p, d, kv_bytes=128, page_ids=(1, 2), source_page_ids=(2, 1),
                         destination_page_ids=(0, 2, 1), cached_page_ids=(0,))
    second = phase_bridge(p, d, kv_bytes=128, page_ids=(1, 2))
    assert first.edge_bytes == second.edge_bytes == ((1, 1, 64), (2, 2, 64))
    assert first.destination_pages == ((), (1,), (2,), ())
    for values in ((), (1,)):
        with pytest.raises(ModelingError):
            phase_bridge(p, d, kv_bytes=128, page_ids=(1, 2), destination_page_ids=values)


def test_r11_01_replicated_pcp_uses_absolute_dcp_and_source_dcp_members():
    case = _dense(layers=1, kv_heads=1, num_devices=12)
    p = ParallelSpec(tp=2, cp=2, pcp=2)
    d = ParallelSpec(tp=2, cp=2, pcp=3, pcp_lane_policy="replica_lanes")
    t = logical_transfer_bytes(case, prompt_tokens=256, destination_cached_tokens=128,
                               prefix_digest="d" * 64)
    b = _bridge(p, d, t, replicated=True)
    z = _compose(case, [_instance(case, p, range(4))],
                  [_instance(case, d, range(4, 10))], b, t, bridge_remap_bytes(b)["total_bytes"])
    assert z["qualified"]
    assert z["handoff_rank_edges"] == ((3, 5, 65_600),)
    # Data stays on DCP member 1 through every PCP replica; control has its
    # independently declared leader and does not change these data amounts.
    assert (5, 7, 65_536) in z["remap_rank_edges"]
    assert (5, 9, 65_536) in z["remap_rank_edges"]


@pytest.mark.parametrize("heads,tp", tuple(product((1, 2, 4), (1, 2, 4, 8))))
def test_r11_02_every_tp_consumer_receives_its_heads(heads, tp):
    case = _dense(layers=1, kv_heads=heads, num_devices=16)
    p, d = ParallelSpec(tp=4), ParallelSpec(tp=tp)
    t = logical_transfer_bytes(case, prompt_tokens=128)
    z = _compose(case, [_instance(case, p, range(4))],
                  [_instance(case, d, range(4, 4 + tp))], _bridge(p, d, t), t)
    assert z["qualified"], z
    per_head = 65_536
    local = max(1, heads // tp)
    for rank in range(4, 4 + tp):
        assert z["handoff_demands"][f"port:{rank}:ingress"] >= per_head * local + 64
    assert z["unique_logical_bytes"] == per_head * heads
    assert z["destination_replica_bytes"] == per_head * (tp * local - heads)
    assert z["wire_bytes"] == pytest.approx(z["unique_logical_bytes"] +
        z["destination_replica_bytes"] + z["handoff_header_bytes"])
    canonical = {h * 4 // heads for h in range(heads)}
    assert {s for s, _, _ in z["handoff_rank_edges"]} == canonical


def test_r11_02_replica_public_admission_and_header_conservation():
    case = _slow_last(_dense(layers=4, kv_heads=1, num_devices=3, max_seqs=1))
    r = _solve(case, d=ParallelSpec(tp=2), osl=4)
    b = r["best_configuration"]
    assert r["qualified"]
    assert b["handoff_cut_demands"]["port:1:ingress"] == 262_400
    assert b["handoff_cut_demands"]["port:2:ingress"] == 262_400
    assert b["handoff_cut_demands"]["port:0:egress"] == 524_800
    assert b["unique_logical_bytes"] == b["destination_replica_bytes"] == 262_144
    assert b["handoff_header_bytes"] == 512 and b["wire_bytes"] == 524_800
    demand = 262_400 + 65_536
    assert b["shared_cut_demands"]["port:2:ingress"] == demand
    assert b["admitted_rps"] * demand < .9 * 1e6
    assert b["admitted_rps"] < 3.429878048784299


def test_r11_02_mla_index_and_compressor_replicas_are_physical_copies():
    case = _sparse(layers=1)
    case = replace(case, model=replace(case.model, layer_plans=tuple(replace(l,
        cache_kind="window_compressed", compress_ratio=16, owns_long_kv=True,
        compressor_state_rows=16, compressor_width=1) for l in case.model.layer_plans)))
    p, d = ParallelSpec(tp=2), ParallelSpec(tp=4)
    t = logical_transfer_bytes(case, prompt_tokens=128)
    assert t["transferred_index_bytes"] > 0 and t["fixed_state_transfer_bytes"] > 0
    z = _compose(case, [_instance(case, p, (0, 1))], [_instance(case, d, range(2, 6))],
                  _bridge(p, d, t), t)
    assert z["qualified"], z
    assert z["handoff_rank_edges"] == tuple((0, rank, t["transfer_total_bytes"]) for rank in range(2, 6))


@pytest.mark.parametrize("flag", ("mtp_shared_weights", "draft_shared_weights", "eagle_shared_weights"))
def test_r11_03_shared_target_geometry_prices_mandatory_gathers(flag):
    case = _dense(layers=1, hidden=256, heads=2, kv_heads=1, pcp=2, num_devices=3, max_seqs=1)
    case = replace(case, machine=_machine_with(case, [_link("one", range(3), switch=1000)]))
    case = _speculative(case, None, k=3, cost=.001)
    metadata = dict(case.runtime.metadata)
    metadata.pop("mtp_shared_weights")
    case = replace(case, runtime=replace(case.runtime, metadata={**metadata, flag: True}))
    draft = shared_target_draft(case.model)
    assert draft.layers == 1 and draft.target_shared_lm_head and draft.target_shared_embedding
    assert draft.q_width == 256 and draft.kv_width == 128
    facts = _facts(case, "decode", isl=256, osl=1, hit=1)
    assert facts["feasible"]
    assert facts["draft_step_ms"] == pytest.approx(.512)
    assert facts["draft_comm_demands"]["switch:intra_node_hccs"] == 512
    assert facts["step_ms"] == pytest.approx(5.6607296)
    r = _solve(case, d=ParallelSpec(pcp=2), isl=256, destination_cached_tokens=256,
                 prefix_digest="d" * 64, tpot_slo_ms=5)
    assert not r["qualified"] and r["best_configuration"] is None
    control = _speculative(case, replace(_draft(256), num_heads=2, num_kv_heads=1, kv_width=128))
    control_facts = _facts(control, "decode", isl=256, osl=1, hit=1)
    assert control_facts["draft_step_ms"] == facts["draft_step_ms"]
    assert control_facts["draft_comm_demands"] == facts["draft_comm_demands"]
    assert control_facts["step_ms"] >= facts["step_ms"] > 5


@pytest.mark.parametrize("value", (0, -1, None, True, 1.5))
def test_r11_03_unusable_shared_geometry_fails_before_pricing(value):
    case = _speculative(_dense(pcp=2, num_devices=3, max_seqs=1), None)
    case = replace(case, model=replace(case.model, intermediate_size=value))
    assert shared_target_draft(case.model) is None
    facts = _facts(case, "decode", isl=256, osl=1, hit=1)
    assert not facts["feasible"] and not facts["route_qualified"]
    assert facts["route_evidence"] == "missing_draft_geometry"


def test_r11_03_shared_draft_without_scalar_uses_roofline_and_moe_remains_admissible():
    case = _speculative(_dense(experts=8, top_k=2, moe_inter=4096,
                               dp=2, ep=2, num_devices=3, max_seqs=1), None, cost=0)
    case = replace(case, machine=_machine_with(case, [_link("one", range(3), switch=1000)]))
    draft = shared_target_draft(case.model)
    assert draft.num_experts == 8 and draft.top_k == 2 and draft.expert_intermediate_size == 4096
    f = _facts(case, "decode", isl=256, osl=1, hit=1)
    assert f["feasible"] and f["draft_step_ms"] > 0 and f["draft_comm_demands"]


def _window_case():
    case = _dense(layers=2, kv_heads=1, num_devices=3, max_seqs=1, max_model_len=8192)
    return replace(case, model=replace(case.model, layer_plans=tuple(
        replace(l, window=128) if i == 0 else l for i, l in enumerate(case.model.layer_plans))))


def test_r11_04_window_stage_lifetime_and_public_negative_headroom():
    case = _window_case()
    p, d = ParallelSpec(), ParallelSpec(pp=2)
    t = logical_transfer_bytes(case, prompt_tokens=4096, destination_cached_tokens=3968,
                               prefix_digest="d" * 64)
    lifetime = _exact_lifetime(case, p, d, t)
    assert lifetime.phase_rank_peak_bytes("decode") == (139_264, 139_264)
    direct = with_world(case, d)
    base = _facts(direct, "decode", isl=4096, osl=1, hit=1)["memory"]
    charged = _facts(direct, "decode", isl=4096, osl=1, hit=1, transfer_plan=lifetime)["memory"]
    assert tuple(a[3] - b[3] for a, b in zip(charged.per_rank_bytes, base.per_rank_bytes)) == (139_264, 139_264)
    budget = base.per_rank_bytes[0][3] + 32_768
    tight = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=math.ceil(budget / .9)) if i == 1 else die
        for i, die in enumerate(case.machine.dies))))
    r = _solve(tight, d=d, isl=4096, destination_cached_tokens=3968, prefix_digest="d" * 64)
    assert not r["qualified"] and r["best_configuration"] is None
    row = r["rows"][0]["decode_instance_rows"][0]["per_rank_bytes"][0]
    assert row[4] - row[3] == pytest.approx(-106_495.7)


def test_r11_04_tail_window_dcp_replica_memory_and_eplb_are_not_averaged():
    case = replace(_window_case(), machine=_dense(num_devices=17).machine)
    p, d = ParallelSpec(), ParallelSpec(tp=4, cp=2, dp=2, pp=2)
    t = logical_transfer_bytes(case, prompt_tokens=1024, destination_cached_tokens=512,
                               prefix_digest="d" * 64)
    lifetime = _exact_lifetime(case, p, d, t, eplb=4096)
    peaks = lifetime.phase_rank_peak_bytes("decode")
    for rank in range(d.world_size):
        stage = rank // 8
        lane = rank % 2
        # Window data only on absolute page 7; full-context pages 4..7 are
        # equally split. DP request routing never divides a rank allocation.
        retained = (65_536 if lane else 0) if stage == 0 else 131_072
        assert peaks[rank] == 2 * retained + 8192 + 4096
    z = _compose(case, [_instance(case, p, (0,))], [_instance(case, d, range(1, 17))],
                  _bridge(p, d, t), t)
    assert z["qualified"]
    assert z["handoff_demands"]["port:2:ingress"] > z["handoff_demands"]["port:1:ingress"]


@pytest.mark.parametrize("recipe", ("alltoall", "mc2"))
@pytest.mark.parametrize("partition,mask", (((2, 2), (True, True, False, False)),
    ((3, 1), (False, True, False, True)), ((2, 2, 1), (True, False, False, False, True))))
def test_r11_05_decode_ep_counts_and_wire_are_stage_local(recipe, partition, mask):
    case = _dense(layers=len(mask), experts=32, top_k=4, moe_inter=4096,
                  dp=2, ep=2, pp=len(partition), num_devices=8, max_seqs=1)
    case = replace(case, model=replace(case.model, layer_plans=tuple(
        replace(l, ffn="moe" if active else "dense") for l, active in zip(case.model.layer_plans, mask))),
        runtime=replace(case.runtime, pp_layer_partition=partition, moe_recipe=recipe),
        machine=_machine_with(case, [_link("one", range(8), switch=1000)]))
    step = DecodeCostModel.from_case(case).step(1)
    ops = [o for o in step.comm_ledger.operations if o.role.startswith("ep_")]
    offset = 0
    for stage, count in enumerate(partition):
        owned = sum(mask[offset:offset + count])
        selected = [o for o in ops if (2 * stage, 2 * stage + 1) in o.groups]
        assert len(selected) == (0 if not owned else 1 if recipe == "mc2" else 2)
        assert all(o.count == owned for o in selected)
        offset += count
    f = _facts(case, "decode", isl=256, osl=1, hit=1)
    assert f["shared_cut_demands"]["switch:intra_node_hccs"] == 32_768 * sum(mask) + 4096 * (len(partition) - 1)


@pytest.mark.parametrize("phase", ("prefill", "decode"))
@pytest.mark.parametrize("partition", ((4,), (2, 2), (3, 1)))
def test_r11_05_dense_tp_wire_and_timing_use_the_same_stage_counts(phase, partition):
    case = _dense(layers=4, tp=2, pp=len(partition), num_devices=8, max_seqs=1)
    case = replace(case, runtime=replace(case.runtime, pp_layer_partition=partition),
                   machine=_machine_with(case, [_link("one", range(8), switch=1000)]))
    if phase == "prefill":
        step = prefill_step_ms(case, isl=256, prefix_hit_rate=0, batch=1)
        ledger, stages, factor = step["comm_ledger"], step["pp_stage_ms"], 256
    else:
        step = DecodeCostModel.from_case(case).step(1)
        ledger, stages, factor = step.comm_ledger, step.pp_stage_ms, 1
    for stage, count in enumerate(partition):
        ops = [o for o in ledger.operations if o.kind == "all_reduce" and (2 * stage, 2 * stage + 1) in o.groups]
        assert len(ops) == 1 and ops[0].count == count
        assert stages[stage] >= ops[0].demands()["switch:intra_node_hccs"] / 1000
    expected = 2 * 4096 * 4 * factor + 2048 * factor * (len(partition) - 1)
    assert ledger.cut_demands()["switch:intra_node_hccs"] == expected


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_r11_05_all_axes_and_sequential_pcp_gathers_use_their_layer_stage(phase):
    spec = ParallelSpec(tp=4, dp=2, ep=2, expert_tp=2, expert_dp=2, cp=2, pcp=2, pp=2)
    case = with_world(_dense(layers=4, kv_heads=1, experts=8, top_k=2, moe_inter=4096, num_devices=32), spec)
    case = replace(case, model=replace(case.model, layer_plans=tuple(
        replace(l, ffn="moe" if i < 2 else "dense") for i, l in enumerate(case.model.layer_plans))))
    step = (prefill_step_ms(case, isl=256, prefix_hit_rate=0, batch=1) if phase == "prefill"
            else DecodeCostModel.from_case(case).step(1))
    ledger = step["comm_ledger"] if phase == "prefill" else step.comm_ledger
    plan = build_placement(case, spec).to_dict()
    for op in ledger.operations:
        if op.kind == "peer":
            continue
        stages = {plan["rank_map"][g[0]][0] for g in op.groups}
        assert len(stages) == 1
        assert op.count == (1 if op.role == "pcp_sequential" else 2)
        if op.role in {"etp", "edp"} or op.role.startswith("ep_"):
            assert stages == {0}


@pytest.mark.parametrize("layers", (1, 2, 5))
def test_r11_05_draft_pp_layers_are_partitioned_once(layers):
    case = with_world(_dense(layers=4, num_devices=4), ParallelSpec(tp=2, pp=2))
    case = replace(case, machine=_machine_with(case, [_link("one", range(4), switch=1000)]))
    draft = replace(_draft(256), layers=layers)
    case = _speculative(case, draft)
    price = priced_draft_step_ms(case, draft)
    assert price.groups == ((2, 3),)
    # One dense TP reduction per draft layer, plus one legitimate PP hop.
    expected = layers * 2 * 512 + 256
    assert price.comm_demands["switch:intra_node_hccs"] == expected
    assert price.mandatory_ms >= expected / 1000
    assert price.execution_duration(.001)[0] >= expected / 1000


def test_r11_05_dense_dp_omits_inactive_expert_collectives():
    case = _dense(dp=2, num_devices=2, max_seqs=1)
    step = DecodeCostModel.from_case(case).step(1)
    assert not step.comm_ledger.operations
    facts = _facts(case, "decode", isl=128, osl=1, hit=1)
    assert facts["feasible"] and not facts["shared_cut_demands"]


def test_r11_02_incomplete_extents_cannot_certify_conservation():
    case = _dense(layers=1, kv_heads=1, num_devices=3)
    p, d = ParallelSpec(), ParallelSpec(tp=2)
    t = logical_transfer_bytes(case, prompt_tokens=128)
    b = _bridge(p, d, t)
    altered = dict(t, layer_extents=({**t["layer_extents"][0], "kv_bytes": 1},))
    z = _compose(case, [_instance(case, p, (0,))], [_instance(case, d, (1, 2))], b, altered)
    assert not z["qualified"] and z["reason"] == "bridge_layer_extent_not_conserved"
