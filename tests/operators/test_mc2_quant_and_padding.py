"""MC2 row framing, activation scales, and padded caller boundaries.

[Role] Exercise framed rows and quantized producers through real consumers.
[Invariants] Framing precedes assignment multiplication. Padding retains local
  work and allocation while masked assignments contribute no peer payload.
[Timeouts & Locks] Finite analytical graphs and bounded source fixture reads.
[Side-Effect Policy] Read-only data fixtures and in-memory graph execution.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import ceil
from pathlib import Path

import pytest
from _support.axes import PHASES, REPRESENTATIVE_SKUS
from _support.hardware import hardware_case, machine

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.instructions import cal_token_send_expert_work, get_sff_value_u64_i64_work
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.peer_traffic import PeerTraffic
from mindie_sim.prediction.operators import mc2
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_matrix_contracts import matrix_comparison_scope

pytestmark = pytest.mark.tier0
DATA, FRAME, FLAG, TRIPLE, FP32 = 480, 512, 32, 12, 4
ROW_ALIGN, MX_ALIGN, GROUP_ALIGN = 32, 256, 128
HIDDEN, EXPERTS = 32, 16
MESH_SKU = REPRESENTATIVE_SKUS["a5"]
PREFIX = "s0.l0."
WAIT = 7
ROOT = Path(__file__).resolve().parents[2]
ROW_INT8 = "per_token_int8"
UNFUSED_IMPLEMENTATION = "vllm_ascend_mc2_unfused_w8a8"
NORMAL = {"kind": "arch35_fullmesh", "dispatch_quant": "none"}
PARTITION = {"kind": "vllm_ascend_mc2_tp_slice", "padding": "right_to_tp"}
SOURCE = {"kind": "vllm_ascend_mc2_tp", "global_bs": 0}
SELECTOR = dict(kind="separate_quantized_payload", payload_dtype="int8", scale_dtype="fp32",
                scale_granularity="row", histogram_dtype="int64")

def align(amount, block):
    return ceil(amount / block) * block

def row_plan(abi, width, rows, transport, **fields):
    hw = machine(MESH_SKU, num_devices=2)
    op = OpSpec("rows", "mc2", "decode", 0, 0, (), (), (), (), (), dtype="bf16",
                metadata=(("mc2", dict(abi=abi, width=width, world_size=2, transport_abi=transport,
                                      assignment_matrix=((0, rows), (0, 0)), **fields)),))
    return mc2.build(op, Geometry.from_devices((hw.dies[0],)))


@pytest.mark.parametrize("abi", ("dispatch_v2", "combine_v2"))
def test_direct_rows_and_full_mesh_frames_use_their_own_wire_layout(abi):
    width, rows = 4096, 2
    direct = row_plan(abi, width, rows, "a5_ccu")
    framed = row_plan(abi, width, rows, "a5_mte")
    row = width * 2
    dispatch = abi == "dispatch_v2"
    framed_row = ceil((row + (ROW_ALIGN if dispatch else 0)) / DATA) * FRAME
    for plan, payload, status in ((direct, row * rows, FRAME if dispatch else 0),
                                  (framed, framed_row * rows, FLAG if dispatch else 0)):
        phases = {p.phase_id: {w.resource: w.amount for w in p.work} for p in plan.pipeline_spec.phases}
        assert plan.metadata["peer_payload_bytes"] == phases["cohort0_send_1"]["link"] == payload
        assert phases.get("cohort0_status_1", {}).get("link", 0) == status
        incoming_count = status if plan is direct else 0
        assert phases.get("cohort0_recv_link_1", {}).get("link", 0) == incoming_count
        assert plan.work.resources["link"] == payload + status + incoming_count


@pytest.mark.parametrize("mode,payload_dtype", (("none", "fp8_e4m3"), ("per_token", "int8"), ("mx", "fp8_e4m3")))
@pytest.mark.parametrize("width", (256, 513, 5120))
def test_quantized_direct_stride_aligns_payload_and_scales_per_assignment(mode, payload_dtype, width):
    incoming = payload_dtype if mode == "none" else "bf16"
    given_scales = 64 if mode == "none" else 0
    scales = {"none": given_scales, "per_token": FP32, "mx": align(ceil(width / ROW_ALIGN), 2)}[mode]
    args = dict(input_dtype=incoming, payload_dtype=payload_dtype, quant_mode=mode, input_scale_bytes=given_scales)
    alignment = MX_ALIGN if mode == "mx" else ROW_ALIGN
    plan = row_plan("dispatch_v2", width, 3, "a5_ccu", **args)
    assert plan.work.resources["link"] == 3 * align(align(width, alignment) + scales, FRAME) + 2 * FRAME
    if mode != "none":
        with pytest.raises(ModelingError, match="requires_unscaled_float_row"):
            row_plan("dispatch_v2", width, 3, "a5_ccu", **{**args, "input_scale_bytes": scales})

@pytest.mark.parametrize("width", (1, 31, 32, 127, 128, 239, 240, 255, 256, 448, 480, 481, 1536, 7168))
@pytest.mark.parametrize("mode,incoming,outgoing,input_scale", (
    ("none", "bf16", "bf16", 0), ("none", "fp8_e4m3", "fp8_e4m3", 64),
    ("per_token", "bf16", "int8", 0), ("mx", "bf16", "fp8_e4m3", 0),
    ("mx", "bf16", "fp4", 0), ("per_group", "fp16", "fp8_e5m2", 0)))
def test_rows_reserve_scales_and_triple_before_wire_framing(width, mode, incoming, outgoing, input_scale):
    payload = width * 2 if outgoing == "bf16" else ceil(width / 2) if outgoing == "fp4" else width
    alignment, scales = {
        "none": (ROW_ALIGN, input_scale), "per_token": (ROW_ALIGN, FP32),
        "mx": (MX_ALIGN, align(ceil(width / ROW_ALIGN), 2)),
        "per_group": (GROUP_ALIGN, ceil(width / GROUP_ALIGN) * FP32),
    }[mode]
    offset = align(payload, alignment)
    triple = align(offset + scales, ROW_ALIGN)
    blocks = ceil((triple + ROW_ALIGN) / DATA)
    args = dict(input_dtype=incoming, payload_dtype=outgoing, quant_mode=mode, input_scale_bytes=input_scale)
    row = mc2.dispatch_row_layout(width, **args)
    weighted = mc2.dispatch_row_layout(width, **args, has_expert_scales=True)
    assert (row.payload_bytes, row.scale_offset_bytes, row.scale_bytes) == (payload, offset, scales)
    assert (row.triple_offset_bytes, row.blocks, row.packet_bytes) == (triple, blocks, blocks * FRAME)
    assert (row.data_region_bytes, row.flag_bytes) == (blocks * DATA, blocks * FLAG)
    assert row.materialized_bytes == payload + scales + TRIPLE
    assert weighted.packet_bytes == row.packet_bytes and weighted.materialized_bytes == row.materialized_bytes + FP32
    returned = mc2.combine_row_layout(width, dtype="fp16")
    assert returned.payload_bytes == returned.materialized_bytes == width * 2
    assert returned.packet_bytes == ceil(width * 2 / DATA) * FRAME
    assert returned.scale_bytes == returned.triple_offset_bytes == 0

@pytest.mark.parametrize("abi", ("dispatch_v2", "combine_v2"))
@pytest.mark.parametrize("kept", (0, 3))
def test_governed_masked_assignments_keep_pack_work_and_zero_peer_payload(abi, kept):
    width, masked, top_k = 481, 2, 1
    routes = tuple(dict(source_rank=0, destination_rank=1, source_token_id=i, top_k_slot=0,
                        destination_expert=1, active=i < kept) for i in range(kept + masked))
    dispatch = abi == "dispatch_v2"
    owner = 0 if dispatch else 1
    # Combine reverses the original routes. Its pack owner still keeps allocated rows.
    base = dict(abi=abi, world_size=2, rank=owner, width=width, top_k=top_k, status_bytes=0,
                source_tokens=kept + masked, quant_mode="per_token", payload_dtype="int8",
                combine_dtype="fp16", routes=routes)
    hw = machine(MESH_SKU, num_devices=2)
    geo = Geometry.from_devices((hw.dies[owner],))
    for padding in (0, masked):
        fields = dict(base, routes=routes[:kept + padding], masked_padding_rows=padding)
        op = OpSpec("rows", "mc2", "decode", 0, 0, (), (), (), (), (),
                    dtype="bf16", metadata=(("mc2", fields),))
        plan = mc2.build(op, geo)
        phases = {p.phase_id: {w.resource: w.amount for w in p.work} for p in plan.pipeline_spec.phases}
        packet = (ceil((align(align(width, ROW_ALIGN) + FP32, ROW_ALIGN) + ROW_ALIGN) / DATA)
                  if dispatch else ceil(2 * width / DATA)) * FRAME
        assert plan.work.resources.get("link", 0) == plan.metadata["peer_payload_bytes"] == kept * packet
        assert phases["cohort0_pack"].get("mte2_aiv", 0) == (kept + padding) * (width * 2 + (4 if dispatch else 12))
        assert phases["cohort0_pack"].get("vector", 0) == (kept + padding) * (7 * width if dispatch else 1)

def case(tp=1, dp=2, *, width=HIDDEN, mode="none", native=True, source=False, experts=None):
    raw = hardware_case()
    size = tp * dp
    raw["machine"].update(num_devices=size, links=[dict(name="fabric", capacity_bytes_per_ms=1e6,
        members=list(range(size)), semantics="per_pair", latency_us=5)])
    dev = raw["machine"]["device"]
    dev["compute_peaks"].update(dict.fromkeys(("bf16", "fp32", "int8", "fp8_e4m3"), 1e9))
    dev["rates"].update(dict.fromkeys(("memory", "vector", "mte2_aic", "mte2_aiv", "mte3"), 1e9))
    dev["resource_units"] = dict(vector=4, mte2_aiv=4, mte3=4)
    raw["model"].update(hidden_size=width, num_heads=max(4, tp), num_kv_heads=max(4, tp),
        head_dim=width // max(4, tp), intermediate_size=width, num_experts=experts or 2 * size,
        top_k=2, ffn="moe", moe_intermediate_size=32 if width == HIDDEN else width // 4,
        num_shared_experts=int(tp > 1))
    kernels = {}
    if native:
        kernels["native_moe_protocol"] = dict(NORMAL, dispatch_quant=mode, max_inflight_per_rank=8)
    if tp > 1:
        kernels["moe_token_partition"] = dict(PARTITION)
    if source:
        kernels["force_eplb_source"] = dict(SOURCE)
    raw["runtime"].update(tp=tp, dp=dp, ep=size, kernels=kernels, routing={"policy": "balanced"})
    return raw

def graphs(raw, sequences=1, *, tokens=1, phase="decode", scope="body"):
    return target_graphs(raw, phase=phase, tokens=tokens, sequences=sequences, context=128, scope=scope)

def index(ops):
    return {o.op_id.removeprefix(PREFIX): o for o in ops}

def costs(raw, grouped, **options):
    report = graph_costs(raw, grouped, external_inputs={"analysis_input": WAIT}, input_ready_ms=WAIT, **options)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert all(row["ready_ms"] >= WAIT for row in report["timeline"])
    return report

def wire(report):
    return sum(v for r, v in report["execution"]["serviced"].items() if r.startswith("link:"))

@pytest.mark.parametrize("rows_allocated,rows", ((4, 1), (4, 3), (4, 4), (8, 5), (8, 8)))
@pytest.mark.parametrize("native,source,padding", ((False, False, "concat"), (True, True, "concat"),
                                                  (False, True, "pad_v3")))
def test_full_row_caller_precedes_padding_and_restores_only_valid_output(rows_allocated, rows, native, source, padding):
    raw = case(4, 1, experts=EXPERTS, native=native, source=source)
    settings = raw["runtime"]["kernels"]["moe_token_partition"]
    settings.update(padding_implementation=padding, logits_dtype="float32")
    ops = index(graphs(raw, rows)[0])
    assert all(ops[key].local_shape[0] == rows for key in ("qkv", "router", "shared.gate_up"))
    local = rows_allocated // 4
    valid = tuple(min(local, max(0, rows - rank * local)) for rank in range(4))
    gather = ops["mc2_tp_gather"]
    assert gather.local_shape == (rows_allocated, HIDDEN) and gather.link_bytes == 3 * local * HIDDEN * 2
    assert dict(gather.metadata)["moe_token_partition"]["valid_rows_per_rank"] == valid
    assert ops["topk"].local_shape == (local, EXPERTS)
    ledger = dict(ops["expert_up"].metadata)["routing_realization"]
    assert sum(ledger["global_expert_rows"]) == sum(ledger["rank_assignment_rows"]) == rows * 2
    if rows == rows_allocated:
        assert all(name not in ops for name in ("mc2_input_pad", "mc2_logits_pad", "mc2_output_trim"))
        return
    for role, width, dtype_bytes in (("input", HIDDEN, 2), ("logits", EXPERTS, 4)):
        pad = ops[f"mc2_{role}_pad"]
        assert pad.local_shape == (rows_allocated, width)
        assert pad.mte2_bytes == (rows if padding == "pad_v3" else rows_allocated) * width * dtype_bytes
        assert pad.mte3_bytes == rows_allocated * width * dtype_bytes
        assert ops[f"mc2_{role}_view"].deps == (pad.op_id,)
    trim = ops["mc2_output_trim"]
    assert trim.local_shape == (rows, HIDDEN) and trim.deps == (gather.op_id,) and trim.hbm_bytes == 0
    assert trim.op_id in ops["join"].deps
    if native:
        forward, backward = (dict(ops[name].metadata)["mc2_phase"] for name in ("dispatch", "unpack"))
        matrix = forward["route_matrix"]
        assert backward["route_matrix"] == matrix and backward["route_reverse"]
    else:
        matrix, reverse = (dict(ops[name].metadata)["peer_traffic"]["payload_bytes"]
                           for name in ("dispatch", "combine"))
        assert reverse == tuple(zip(*matrix))
    assert all(count or not any(matrix[rank]) for rank, count in enumerate(valid))

def test_capture_rows_and_rank_copies_preserve_distinct_physical_boundaries():
    raw = case(4, 1, native=False, experts=EXPERTS)
    raw["runtime"].update(graph_mode="auto", graph_capture_mode="full", graph_capture_keys=[4])
    ops = index(graphs(raw, 3)[0])
    assert ops["router"].local_shape[0] == 4
    assert sum(dict(ops["expert_up"].metadata)["routing_realization"]["global_expert_rows"]) == 8
    assert "mc2_input_pad" not in ops
    raw["runtime"]["graph_mode"] = "off"
    settings = raw["runtime"]["kernels"]["moe_token_partition"]
    settings["gather_output_materialization"] = "per_rank_copy"
    ops = index(graphs(raw, 3)[0])
    copies = [op for name, op in ops.items() if "mc2_gather_copy" in name]
    assert len(copies) == 4
    assert sum(o.mte2_bytes for o in copies) == sum(o.mte3_bytes for o in copies) == 4 * HIDDEN * 2
    assert set(ops["mc2_gather_complete"].deps) == {op.op_id for op in copies}
    assert ops["mc2_output_trim"].deps == (ops["mc2_gather_complete"].op_id,)
    settings.update(padding="none", padding_implementation="pad_v3")
    with pytest.raises(ModelingError, match="pad_v3_requires_mc2_right_to_tp_padding"):
        graphs(raw, 4)

def quant_case(mode, *, send=True):
    width = 5120 if mode == "mx" else 2048
    raw = parse_case(case(width=width, mode=mode if send else "none"))
    fmt = ("fp8_e4m3", "mxfp4", "fp8_e8m0", 1) if mode == "mx" else ("int8", "int8", "fp32", 4)
    activation, expert, scale, size = fmt
    options = dict(expert_block=32, weight_block=(32, 32), activation_group_size=32,
                   activation_scale_format=scale, activation_scale_pack=2) if mode == "mx" else {}
    quant = QuantSpec("explicit", weight_format="bf16" if mode == "mx" else "int8",
        expert_weight_format=expert, activation_format=activation, scale_format=scale, scale_bytes=size,
        exclusions=("router", "norm"), **options)
    return replace(raw, model=replace(raw.model, quant=quant))

def protocol_only(declared):
    selected = {}
    for rank, ops in declared.items():
        keep = {o.op_id for o in ops if "device_kernel_group" in dict(o.metadata)}
        selected[rank] = tuple(replace(o, deps=tuple(d for d in o.deps if d in keep))
                               for o in ops if o.op_id in keep)
    return selected

@pytest.mark.parametrize("mode", (ROW_INT8, "mx"))
@pytest.mark.parametrize("sequences", ((1, 0), (0, 3), (1, 3)))
def test_quantized_wire_and_consumer_preserve_received_scales(mode, sequences):
    raw = quant_case(mode)
    before = deepcopy(raw)
    declared = graphs(raw, sequences)
    plain = graphs(quant_case(mode, send=False), sequences)
    assert raw == before
    width = raw.model.hidden_size
    scales = width // ROW_ALIGN if mode == "mx" else FP32
    alignment = MX_ALIGN if mode == "mx" else ROW_ALIGN
    send_bytes = ceil((align(align(width, alignment) + scales, ROW_ALIGN) + ROW_ALIGN) / DATA) * FRAME
    return_bytes = ceil(width * 2 / DATA) * FRAME
    report = costs(raw, protocol_only(declared), isolated=False)
    assert wire(report) == sum(sequences) * (send_bytes + return_bytes) + 4 * FLAG
    for rank, ops in declared.items():
        chosen, default = index(ops), index(plain[rank])
        up, dispatch = chosen["expert_up"], chosen["dispatch"]
        assert "expert_up.quantize" not in chosen
        if up.local_shape[0]:
            assert "expert_up.quantize" in default
        assert chosen.get("expert_down.quantize") == default.get("expert_down.quantize")
        assert (up.flops, up.mte2_bytes, up.mte3_bytes) == (
                default["expert_up"].flops, default["expert_up"].mte2_bytes, default["expert_up"].mte3_bytes)
        meta = dict(up.metadata)
        assert up.deps == (meta["activation_scale_producer"],) == (dispatch.op_id,)
        assert meta["activation_scale_bytes"] == up.local_shape[0] * scales
        if mode == ROW_INT8:
            assert meta["activation_scale_format"] == "fp32" and meta["activation_scale_shape"] == (up.local_shape[0],)
        assert dict(chosen["expert_down"].metadata)["output_format"] == "bf16"
        assert dict(dispatch.metadata)["activation_scale_bytes"] == up.local_shape[0] * scales

@pytest.mark.parametrize("mode,mutation,message", (
    (ROW_INT8, {"activation_group_size": 128}, "matching_int8_fp32_per_row"),
    ("mx", {"activation_scale_format": "fp32", "activation_scale_pack": 1}, "matching_e4m3_e8m0")))
def test_quantized_consumer_waits_for_receive_and_rejects_foreign_scale_layout(mode, mutation, message):
    raw = quant_case(mode)
    declared = graphs(raw, (1, 1))
    declared[0] = tuple(replace(o, deps=(*o.deps, "late")) if o.op_id == PREFIX + "dispatch" else o for o in declared[0])
    report = graph_costs(raw, declared, external_inputs={"analysis_input": 0, "late": 17}, isolated=False)
    times = {r["id"]: r for r in report["timeline"]}
    received, consumer = (times["r0:" + PREFIX + name] for name in ("dispatch", "expert_up"))
    assert received["ready_ms"] >= 17 and consumer["ready_ms"] >= received["complete_ms"]
    bad = replace(raw, model=replace(raw.model, quant=replace(raw.model.quant, **mutation)))
    with pytest.raises(ModelingError, match=message):
        graphs(bad)

@pytest.mark.parametrize("mode", (ROW_INT8, "mx"))
@pytest.mark.parametrize("role", ("vector", "hbm", "mte2_aiv", "mte3", "link"))
def test_quantized_protocol_scales_with_each_pipe_and_conserves_wire(mode, role):
    raw = quant_case(mode)
    reports = []
    for factor in (.5, 1, 2):
        hw = raw.machine
        if role == "link":
            hw = replace(hw, links=(replace(hw.links[0], capacity_bytes_per_ms=100 * factor),))
        else:
            hw = replace(hw, dies=tuple(replace(d, rates=tuple({**dict(d.rates), role: 1e7 * factor}.items()))
                                        for d in hw.dies))
        changed = replace(raw, machine=hw)
        reports.append(costs(changed, protocol_only(graphs(changed)), isolated=False))
    assert reports[0]["duration_ms"] > reports[1]["duration_ms"] > reports[2]["duration_ms"]
    assert [wire(r) for r in reports] == pytest.approx([wire(reports[0])] * len(reports))
    for left, right in zip(reports, reports[1:]):
        # Slower arrivals may add flag polls on these three physical roles.
        ignored = ("memory@", "mte2_aiv@", "vector@") if role in ("vector", "link") else ()
        stable = lambda r: {k: v for k, v in r["execution"]["serviced"].items() if not k.startswith(ignored)}
        assert stable(left) == pytest.approx(stable(right))

@pytest.mark.parametrize("calls", (17, 30))
def test_short_row_scalar_calls_use_shared_clock(calls):
    die = machine(MESH_SKU).dies[0]
    rates = {key: value / 1000 for key, value in die.rates}
    units = {key: die.units(key) for key in rates}
    ctx = RecipeContext(None, {}, {r: r for r in rates}, rates, units["vector"], resource_units=units)
    scalar = get_sff_value_u64_i64_work(calls)
    clock = 1.65e9
    assert rates["pipe_s_issue"] * 1000 == units["pipe_s_issue"] * clock
    assert dict(die.service_modes)["get_sff_value_u64_i64"] == ("pipe_s_issue", units["pipe_s_issue"] * clock)
    for count, shared in ((1, True), (2, True), (2, False)):
        engine = Engine([Resource(r, v, units=units[r]) for r, v in rates.items()])
        stages = tuple(Stage(str(i), ctx.bind(scalar), placement=ctx.owned(
            0 if shared else i, units["vector"], scalar.resources)) for i in range(count))
        done = KernelProgram(stages).start(engine, "scalar", engine.submit("input"))
        engine.advance(until=(done,))
        assert done.time == pytest.approx(calls * 1000 / clock * (count if shared else 1), rel=1e-9)
        assert engine.serviced["pipe_s_issue"] == calls * count
    for rows in (6, 16):
        work = cal_token_send_expert_work(rows)
        assert work.resources == dict(pipe_v_lanes=6 * rows, pipe_v_issue=4, ub_read_bytes=4 * rows, ub_write_bytes=4)
        assert max(amount / (rates[r] / units[r]) for r, amount in work.resources.items()) == 4 / (
            rates["pipe_v_issue"] / units["pipe_v_issue"])

def allocation_case(*, selected=True, masked=True):
    kernels = dict(moe_token_partition=dict(PARTITION), native_moe_protocol=dict(NORMAL, dispatch_quant=ROW_INT8,
                   max_inflight_per_rank=4), lowerings={})
    if selected:
        kernels["lowerings"]["grouped_int8_caller"] = dict(implementation=UNFUSED_IMPLEMENTATION, options={"global_bs": 0})
    if masked:
        kernels["force_eplb_source"] = dict(SOURCE)
    return dict(machine=dict(backend="ascend", sku=MESH_SKU, num_devices=4),
        model=dict(catalog_key="qwen3.5-35b-a3b", quantization="w8a8_dynamic", layers=1),
        runtime=dict(tp=4, ep=4, routing={"policy": "balanced"}, graph_mode="auto",
                     graph_capture_mode="full", graph_capture_keys=[4, 8, 12, 16, 24, 32], kernels=kernels),
        workload={"requests": [{"id": "r", "prompt_tokens": 4096, "output_tokens": 8}]})

@pytest.mark.parametrize("sequences,physical", ((1, 4), (4, 16), (7, 32), (8, 32)))
def test_unfused_storage_scans_capacity_while_matrices_use_active_rows(sequences, physical):
    raw = allocation_case()
    original = deepcopy(raw)
    declared = graphs(raw, sequences, tokens=4)
    ops = index(declared[0])
    assert raw == original
    local, ep, top_k, banks = physical // 4, 4, 8, 64
    counts = Counter()
    for token in range(sequences * 4):
        rank, slot = divmod(token, local)
        for choice in range(top_k):
            cursor = (local * top_k + 1) * rank + slot * top_k + choice
            if cursor % ep == 0:
                counts[cursor // ep % banks] += 1
    valid, active, capacity = sum(counts.values()), len(counts), physical * top_k
    for name, n, k in (("expert_up", 1024, 2048), ("expert_down", 2048, 512)):
        op = ops[name]
        meta = dict(op.metadata)
        assert op.local_shape == op.launch_shape == (valid, n, k)
        assert op.padded_shape == (capacity, n, k) and op.flops == 2 * valid * n * k
        assert tuple(meta["matrix_group_rows"]) == tuple(counts[i] for i in range(banks))
        assert op.mte2_bytes == valid * k + active * n * (k + 2) + valid * FP32
        assert op.mte3_bytes == valid * n * 2 and meta["matrix_epilogue"]["column_scale_bytes"] == active * n * 2
        cast = ops[name + ".weight_scale_cast"]
        assert (cast.mte2_bytes, cast.mte3_bytes) == (banks * n * FP32, banks * n * 2)
        assert cast.op_id in op.deps
    for name, read, output, vector in (("expert_act", 2048, 1024, 4 * 512),
                                      ("expert_down.quantize", 1024, 512 + 4, 4 * 512 + 2)):
        op = ops[name]
        assert op.local_shape == (valid, 512) and op.padded_shape == op.launch_shape == (capacity, 512)
        assert (op.mte2_bytes, op.mte3_bytes, op.vector_ops) == (capacity * read, capacity * output, capacity * vector)
    assert ops["router"].local_shape[0] == physical

@pytest.mark.parametrize("scope", ("body", "model"))
def test_padded_allocation_requires_explicit_mask_and_preserves_capture_contract(scope):
    with pytest.raises(ValueError, match="padded_call_requires_explicit_source_mask_ledger"):
        graphs(allocation_case(masked=False), 7, tokens=4, scope=scope)
    raw = allocation_case()
    ops = index(graphs(raw, 7, tokens=4, scope=scope)[0])
    gate, act = ops["expert_up"], ops["expert_act"]
    assert gate.local_shape[0] == 56 and gate.padded_shape[0] == 256
    ledger = dict(gate.metadata)["routing_realization"]
    assert ledger["valid_tokens_per_rank"] == (8, 8, 8, 4)
    assert sum(ledger["global_expert_rows"]) == 28 * 8
    capture = json.loads((ROOT / "tests/fixtures/mc2_unfused/a5_qwen35_q28_physical32.json").read_text())
    swiglu = next(n["abi"] for n in capture["native"] if n["abi"]["type"] == "SwiGlu")
    assert swiglu["input_shapes"] == '"256,1024"'
    assert predicted_contract(act.to_dict()) == ("swiglu", (256, 512), "bf16")
    quant = predicted_contract(ops["expert_down.quantize"].to_dict())
    assert quant[2:] == (((256, 512), "bf16", "ND"), ((256, 512), "int8", "ND"), ((256,), "fp32", "ND"))
    assert predicted_contract(gate.to_dict())[:2] == ("grouped_matrix", (64, 256, 1024, 2048))
    assert "valid_grouped_rows_not_proven_by_tensor_capacity" in matrix_comparison_scope(gate.to_dict())["not_compared"]
    old = index(graphs(allocation_case(selected=False), 7, tokens=4, scope=scope)[0])
    assert old["expert_act"].local_shape == (56, 512)
    assert sum(o.flops for o in ops.values()) == sum(o.flops for o in old.values())

@pytest.mark.parametrize("options,match", (({}, "explicit_global_bs0"), ({"global_bs": False}, "explicit_global_bs0"),
                                            ({"global_bs": 32}, "explicit_global_bs0")))
def test_unfused_caller_rejects_implicit_or_nonuniform_allocation(options, match):
    raw = allocation_case()
    raw["runtime"]["kernels"]["lowerings"]["grouped_int8_caller"]["options"] = options
    with pytest.raises(ValueError, match=match):
        graphs(raw, 7, tokens=4)

@pytest.mark.parametrize("concurrency", (4, 64, 128))
def test_tp_quantized_receiver_feeds_integer_gate_and_next_quantized_matrix(concurrency):
    folder = ROOT / "examples/cases/high-concurrency-20261030/qwen35-hccl-conditional"
    raw = json.loads((folder / f"qwen35-35b-a5-tp4-c{concurrency}-hccl-source.json").read_text())
    layers = 1 if concurrency == 4 else 4
    raw["model"]["layers"] = layers
    raw["runtime"]["pp_layer_partition"] = [layers]
    raw["runtime"]["kernels"].update(native_moe_protocol=dict(NORMAL, max_inflight_per_rank=4, dispatch_quant=ROW_INT8),
        moe_token_partition=dict(PARTITION, padding="none",
                                 gather_output_materialization="per_rank_copy" if concurrency == 128 else "view"))
    declared = graphs(raw, concurrency, tokens=4)
    ops = {op.op_id: op for op in declared[0]}
    for layer in range(layers):
        prefix = f"s0.l{layer}."
        recv, up, act, down, gather = (ops[prefix + name] for name in
                                       ("dispatch", "expert_up", "expert_act", "expert_down", "mc2_tp_gather"))
        rows = concurrency * 8
        assert ops[prefix + "router"].local_shape[0] == concurrency * 4
        assert ops[prefix + "topk"].local_shape[0] == concurrency
        assert dict(recv.metadata)["activation_payload_shape"] == (rows, 2048)
        assert dict(recv.metadata)["activation_scale_bytes"] == rows * FP32
        assert up.deps == (recv.op_id,) and dict(up.metadata)["output_format"] == "int32"
        assert prefix + "expert_up.quantize" not in ops
        assert act.deps == (up.op_id, recv.op_id) and act.recipe == "grouped_int32_dequant_swiglu"
        assert dict(act.metadata)["input_row_scale_producer"] == recv.op_id
        assert down.deps == (act.op_id,) and dict(down.metadata)["activation_scale_producer"] == act.op_id
        assert dict(gather.metadata)["hccl_selection"]["payload_bytes"] == concurrency * 2048 * 2
    assert sum("mc2_gather_copy" in op for op in ops) == (4 * layers if concurrency == 128 else 0)
    if concurrency == 4:
        times = {t["op_id"]: t for t in costs(raw, declared)["timeline"]}
        assert times[PREFIX + "expert_act"]["ready_ms"] >= times[PREFIX + "expert_up"]["complete_ms"]
        assert times[PREFIX + "mc2_tp_gather.hccl.entry"]["ready_ms"] >= times[PREFIX + "unpack"]["complete_ms"]

def transport_case(dp=2, tp=1, *, selected=True):
    raw = case(tp, dp, native=False, experts=8)
    raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
    raw["runtime"]["kernels"] = {"moe_transport": dict(SELECTOR)} if selected else {}
    return raw

@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("dp,tp,sizes", ((2, 1, (4, 4)), (1, 2, (4,)), (1, 8, (8,)), (3, 1, (4, 2, 0))))
def test_separate_quantized_transport_conserves_records_and_transposes_return_rows(phase, dp, tp, sizes):
    raw = transport_case(dp, tp)
    baseline = graphs(transport_case(dp, tp, selected=False), sizes, phase=phase)
    declared = graphs(raw, sizes, phase=phase)
    assert not any("moe_histogram" in o.op_id for o in baseline[0])
    ep = dp * tp
    for rank, group in declared.items():
        ops = index(group)
        hist, gather, pack, quant, scale, send, gate, down, back = (ops[k] for k in
            ("moe_histogram", "moe_histogram_gather", "pack", "moe_transport_quantize",
             "moe_scale_dispatch", "moe_payload_dispatch", "expert_up", "expert_down", "expert_combine"))
        rows = sizes[rank] * 2 / tp
        meta = dict(hist.metadata)
        assert meta["histogram_dtype"] == "int64" and meta["counts"] is None
        assert sum(meta["global_expert_counts"]) == sum(sizes) * 2
        assert (hist.mte2_bytes, hist.mte3_bytes, hist.vector_ops) == (rows * 4, 8 * 8, rows)
        assert gather.link_bytes == (ep - 1) * 8 * 8
        assert (quant.mte2_bytes, quant.mte3_bytes, quant.vector_ops) == (rows * HIDDEN * 2,
            rows * (HIDDEN + FP32), rows * (4 * HIDDEN + 2))
        fraction = (ep - 1) / ep
        assert send.link_bytes == pytest.approx(rows * HIDDEN * fraction)
        assert scale.link_bytes == pytest.approx(rows * FP32 * fraction)
        return_rows = (sum(sizes) * 2 - rows) / ep if tp == 1 else rows * fraction
        assert back.link_bytes == pytest.approx(return_rows * HIDDEN * 2)
        assert hist.deps == ((ops["topk"].op_id,) if rows else ("analysis_input",))
        assert gather.deps == (hist.op_id,)
        assert gather.op_id in pack.deps and quant.deps == (pack.op_id,)
        assert scale.deps == (quant.op_id, gather.op_id)
        assert send.deps == (quant.op_id, scale.op_id, gather.op_id)
        assert set(gate.deps) == {quant.op_id, scale.op_id, send.op_id}
        assert back.deps == (down.op_id,)
        assert "expert_up.quantize" not in ops and down.deps == (ops["expert_down.quantize"].op_id,)
    costs(raw, declared)

@pytest.mark.parametrize("change,message", (
    ({"payload_dtype": "fp16"}, "unsupported:payload_dtype"), ({"extra": 1}, "unknown_fields"),
    ({"activation_group_size": 16}, "requires_row_scale_width"), ({"moe_recipe": "mc2"}, "requires_alltoall_recipe"),
    ({"quantization": "w8a8_mxfp8"}, "requires_int8_activations")))
def test_separate_transport_requires_explicit_compatible_representation(change, message):
    key, value = next(iter(change.items()))
    raw = transport_case(1, 2) if key == "moe_recipe" else transport_case()
    if key == "activation_group_size":
        raw["model"]["quant"][key] = value
    elif key == "quantization":
        raw["model"]["quant"] = asdict(quant_preset(value))
    elif key == "moe_recipe":
        raw["runtime"][key] = value
        raw["runtime"].pop("routing")
    else:
        raw["runtime"]["kernels"]["moe_transport"].update(change)
    with pytest.raises(ModelingError, match=message):
        graphs(raw, phase="prefill")

def test_directed_quantized_matrix_is_reencoded_on_return_and_binding_is_idempotent():
    raw = parse_case(transport_case(3))
    declared = graphs(raw, (0, 1, 1))
    forward = ((0, 0, 0), (32, 0, 64), (64, 96, 0))
    rewritten = {}
    for rank, group in declared.items():
        ops = index(group)
        send, back = ops["moe_payload_dispatch"], ops["expert_combine"]
        sd = dict(send.metadata, peer_traffic={"kind": "matrix", "payload_bytes": forward})
        bd = {key: value for key, value in back.metadata if key != "peer_traffic"}
        rewritten[rank] = (replace(send, link_bytes=sum(forward[rank]), metadata=tuple(sd.items())),
                           replace(back, metadata=tuple(bd.items())))
    bound = bind_cohort_traffic(raw, rewritten)
    assert bind_cohort_traffic(raw, bound) == bound
    for rank, (send, back) in bound.items():
        assert dict(send.metadata)["peer_traffic"]["payload_bytes"][rank] == forward[rank]
        row = tuple(2 * forward[peer][rank] for peer in range(3))
        reverse = PeerTraffic.from_mapping(dict(back.metadata)["peer_traffic"], len(forward))
        assert tuple(reverse.payload(rank, peer) for peer in range(len(forward))) == row
        assert back.link_bytes == sum(row)

def test_quantized_transport_scope_preserves_draft_expert_traffic():
    fixture = ROOT / "tests/fixtures/verification/v4flash-communication-case.json"
    raw = json.loads(fixture.read_text())
    view = IterationPlan((RequestView("one", 4, 0, True, "prefill", draft_prepare=True),))
    for scopes in ((), ("body",)):
        selector = dict(SELECTOR, layer_scopes=scopes) if scopes else dict(SELECTOR)
        raw["runtime"]["kernels"]["moe_transport"] = selector
        parsed = parse_case(raw)
        ops = Frontend()(view, parsed.model, parsed.runtime, parsed.machine).ops
        quantized = [o for o in ops if o.recipe == "expert_quantized_dispatch"]
        assert len(quantized) == (43 if scopes else 44)
        assert all(dict(o.metadata)["payload_dtype"] == "int8" for o in quantized)
        if scopes:
            draft = [o for o in ops if o.recipe == "expert_dispatch" and o.phase == "draft_prefill"]
            assert len(draft) == 1 and draft[0].link_bytes == 4 * 6 * 4096 * 2 / 8 * 7 / 8

def test_tensor_padding_masks_empty_tail_ranks_from_source_realization():
    raw = case(16, 1, native=False, source=True, experts=256)
    raw["model"]["top_k"] = 8
    ops = index(graphs(raw, 12)[0])
    ledger = dict(ops["expert_up"].metadata)["routing_realization"]
    assert (ledger["valid_global_tokens"], ledger["allocated_global_tokens"]) == (12, 16)
    assert ledger["valid_tokens_per_rank"] == (1,) * 12 + (0,) * 4
    assert sum(ledger["global_expert_rows"]) == 12 * 8
    assert ops["router"].local_shape == (12, 256, HIDDEN)
    assert ops["mc2_tp_gather"].local_shape == (16, HIDDEN)
    assert ops["mc2_output_trim"].local_shape == (12, HIDDEN)

def test_native_protocol_delegates_selection_recipes_and_lowering():
    from types import SimpleNamespace
    from mindie_sim.prediction import mc2_graph
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.schema.operators import OpSpec
    for kind, quant, native_r, net_r in (
        ("arch22_v2_caller", "per_token_int8", "arch22_v2_caller_local", "arch22_v2_declared_peer_channel"),
        ("arch35_fullmesh", "none", "arch35_fullmesh_resource_phase", mc2_graph.NETWORK),
    ):
        decl = {"kind": kind, "dispatch_quant": quant, "max_inflight_per_rank": 2}
        rt = SimpleNamespace(kernels={"native_moe_protocol": decl}, moe_recipe="alltoall")
        reg = mc2_graph.recipes(rt.kernels)
        assert mc2_graph.select(rt.kernels) is decl and reg[net_r].max_inflight_per_rank == 2
        ops = tuple(OpSpec(n, "moe", "decode", 0, 0, (), (2, 32), (2, 32), (2, 32), (2, 32), deps=d, dtype="bf16")
                    for n, d in (("m.pack", ("in",)), ("m.dispatch", ("m.pack",)), ("m.combine", ("m.dispatch",)), ("m.unpack", ("m.combine",))))
        low = mc2_graph.lower(ops, runtime=rt, prefix="m", width=32, dtype="bf16", source_partitions=1, received_assignments=0, tokens=2, top_k=1, experts=2, ep=2)
        assert {o.recipe for o in low if o.source_boundary == "kernel"} == {native_r} and reg[native_r].prepare(low[0], Geometry(1, {"vector": 2}))[1].native is (kind == "arch35_fullmesh")
    with pytest.raises(ModelingError, match="native_moe_protocol_requires_supported_caller"):
        mc2_graph.select({"native_moe_protocol": {"kind": "unknown"}})
