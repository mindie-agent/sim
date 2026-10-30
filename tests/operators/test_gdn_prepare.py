"""Independent numerical, address and work oracles for GDN prepare.

The planner is checked against explicit per-task byte/FLOP enumeration and
against NumPy block inverses.  A structural contract pass is not an accuracy
claim and no measurement participates in these tests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import gdn_prepare as gdn
from mindie_sim.prediction.operators import gdn_registers
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.workload import IterationPlan, RequestView

ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "tests/fixtures/operator_cases/gdn_prepare.json"

BT = 64
K = 128
FP32 = 4
BF16 = 2
VCS_ROWS = 32 * 31 // 2
VCS_OPS = VCS_ROWS * (64 + 64 + 4 + 2)
REGISTER_CAPACITIES = (
    "reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_alu_lanes",
    "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes",
)


def control_case(split_copy: bool = False, units: int = 32, buffers=None) -> dict:
    """Same synthetic control geometry as tools/run_operator_cases.py."""
    rates = {"memory": 1e12, "vector": 1e12, "mte1": 1e12, "fixpipe": 1e12}
    rates.update({role: 1e12 for role in REGISTER_CAPACITIES})
    if split_copy:
        rates.update(read=1e12, write=1e12)
    return {
        "machine": {
            "name": "synthetic-contract-control", "num_devices": 1,
            "device": {"memory_bytes": 64 * 1024 ** 3, "compute_units": units,
                       "compute_peaks": {"bf16": 1e14, "fp16": 1e14, "fp32": 2.5e13,
                                         "fp8": 2e14, "int8": 2e14, "tf32": 5e13},
                       "rates": rates,
                       "resource_units": {
                           role: 2 * units for role in ("vector",) + REGISTER_CAPACITIES},
                       "service_modes": {
                           "ub_copy": {"resource": "vector", "rate": 1e12},
                           "reg_issue_slots": {"resource": "vector", "rate": 1e12}},
                       "buffers": dict(buffers or {}),
                       "buffer_units": {pool: 2 * units if pool == "ub" else units
                                        for pool in (buffers or {})}},
        },
        "model": {"name": "unlabeled-geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def geometry(units: int = 32) -> Geometry:
    return Geometry(units, {
        role: 2 * units for role in ("vector",) + REGISTER_CAPACITIES}, {})


def cases():
    return json.loads(CASE_FILE.read_text())["cases"]


def alternate_flags() -> dict:
    return {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False,
        "use_exp2": False,
        "output_a": True,
        "gate_dtype": "bf16",
    }


def native_flags() -> dict:
    """A5 device_op.py:1884-1903 and composite FP32 cast before Prepare."""
    return {
        **alternate_flags(),
        "use_beta_sigmoid_in_kernel": False,
        "use_exp2": True,
        "gate_dtype": "fp32",
    }


def native_frontend_case():
    case = parse_case(control_case(units=4))
    layer = replace(
        case.model.layer_plans[0], attention="gdn", linear_key_heads=2,
        linear_value_heads=6, linear_heads=6, linear_key_head_dim=K,
        linear_head_dim=K, conv_kernel=4, recurrent_state_dtype="fp32")
    return replace(
        case, model=replace(case.model, layer_plans=(layer,)),
        runtime=replace(
            case.runtime, kernels={"lowerings": {"gdn_prefill": "gdn_prefill_arch35"}},
            tensor_layouts={"linear_output_norm_gate": {"gate": "contiguous"}}))


# ---------------------------------------------------------------------------
# Explicit per-task work oracle (no aggregate shortcuts).
# ---------------------------------------------------------------------------

class Tally:
    def __init__(self):
        self.matrix = Counter()
        self.resources = Counter()

    def matrix_flops(self, dtype, amount):
        self.matrix[dtype] += float(amount)

    def add(self, role, amount):
        self.resources[role] += float(amount)


def source_cumsum_registers(values, scale=1.0):
    """Execute vf.h:205-365 shuffles and retain each Reg call and UB payload."""
    rows = len(values)
    aligned = 1
    while aligned < rows:
        aligned *= 2
    zero = np.zeros(64)
    register = np.zeros(64)
    register[:rows] = values
    events = []

    def emit(op, lanes=0, access=None):
        events.append({"op": op, "active_lanes": lanes, "access": access})

    emit("create_mask")
    emit("create_mask")
    emit("duplicate", 64)
    emit("load", 64, ("read", "x", 0, 256))
    for width in (2, 4, 8, 16):
        if aligned < width:
            break
        registers = [register]
        while len(registers) < width:
            half = len(registers)
            split = [None] * (2 * half)
            for index, value in enumerate(registers):
                joined = np.concatenate((value, zero))
                split[index], split[index + half] = joined[::2], joined[1::2]
                emit("deinterleave", 64)
            registers = split
        for index in range(width // 2, width):
            registers[index][:32] += registers[width // 2 - 1][:32]
            emit("add", 32)
        while len(registers) > 1:
            half = len(registers) // 2
            merged = []
            for index in range(half):
                joined = np.empty(128)
                joined[::2] = registers[index]
                joined[1::2] = registers[index + half]
                merged.append(joined[:64])
                emit("interleave", 64)
            registers = merged
        register = registers[0]
    if aligned >= 32:
        stored = register.copy()
        emit("store", 64, ("write", "x", 0, 256))
        emit("barrier")
        for start in range(0, aligned, 32):
            left = stored[start + 15]
            register[start + 16:start + 32] += left
            emit("broadcast_load", 64, ("read", "x", (start + 15) * FP32, FP32))
            emit("update_mask")
            if start + 32 != 64:
                emit("update_mask")
            emit("mask_xor")
            emit("add", 16)
    if aligned == 64:
        stored = register.copy()
        register[32:] += stored[31]
        emit("store", 64, ("write", "x", 0, 256))
        emit("barrier")
        emit("broadcast_load", 64, ("read", "x", 31 * FP32, FP32))
        emit("mask_xor")
        emit("add", 32)
    result = scale * register[:rows]
    emit("update_mask")
    emit("muls", rows)
    emit("store", rows, ("write", "x", 0, rows * FP32))
    return result, tuple(events)


def bf16_words(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_values(words):
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def source_l2norm_registers(words, eps=1e-6):
    """Execute source BF16 row pairs and retain typed calls plus data edges.

    Calls describe Reg APIs, not compiler-expanded machine instructions or
    dependency latency. Memory intervals are UB payloads in bytes.
    """
    rows = len(words)
    output = np.full((BT, K), 0x5a5a, dtype=np.uint16)
    rstd = np.full(BT, np.float32(-17.0))
    registers, events = {}, []

    def emit(name, op, value, inputs=(), *, lanes=64, access=None, row=None, group=None):
        registers[name] = value
        events.append({"id": name, "op": op, "inputs": tuple(inputs),
                       "active_lanes": lanes, "access": access, "row": row, "group": group})
        return value

    emit("all", "create_mask_all", None, lanes=0)
    emit("first", "create_mask_vl1", None, lanes=0)
    emit("one", "duplicate_scalar_f32", np.float32(1), ("first",), lanes=1)
    groups = [tuple(range(start, start + 2)) for start in range(0, rows - 1, 2)]
    if rows % 2:
        groups.append((rows - 1,))
    for group, active_rows in enumerate(groups):
        def issue(row, suffix, op, value, inputs=(), *, lanes=64, access=None):
            return emit(f"r{row}:{suffix}", op, value, inputs, lanes=lanes,
                        access=access, row=row, group=group)

        def reg(row, suffix):
            return registers[f"r{row}:{suffix}"]

        # The helper loads both 64-element halves before either Cast; the
        # paired caller invokes that helper for both rows before squaring.
        for row in active_rows:
            for half in range(2):
                begin = half * 64
                issue(row, f"x{half}", "load_bf16_unpack",
                      words[row, begin:begin + 64].copy(),
                      access=("read", "x", (row * K + begin) * BF16, 64 * BF16))
            for half in range(2):
                issue(row, f"f{half}", "cast_bf16_f32", bf16_values(reg(row, f"x{half}")),
                      (f"r{row}:x{half}", "all"))
        for row in active_rows:
            for half in range(2):
                value = reg(row, f"f{half}")
                issue(row, f"square{half}", "mul_f32", value * value,
                      (f"r{row}:f{half}", "all"))
        for row in active_rows:
            issue(row, "halves", "add_f32", reg(row, "square0") + reg(row, "square1"),
                  (f"r{row}:square0", f"r{row}:square1", "all"))
        for row in active_rows:
            issue(row, "sum", "reduce_sum_f32", np.sum(reg(row, "halves"), dtype=np.float32),
                  (f"r{row}:halves", "all"))
        for row in active_rows:
            issue(row, "epsilon", "adds_f32", reg(row, "sum") + np.float32(eps),
                  (f"r{row}:sum", "first"), lanes=1)
        for row in active_rows:
            issue(row, "sqrt", "sqrt_f32", np.sqrt(reg(row, "epsilon")),
                  (f"r{row}:epsilon", "first"), lanes=1)
        for row in active_rows:
            issue(row, "rstd", "div_f32", registers["one"] / reg(row, "sqrt"),
                  ("one", f"r{row}:sqrt", "first"), lanes=1)
        for row in active_rows:
            issue(row, "broadcast", "duplicate_lane_f32", np.full(64, reg(row, "rstd")),
                  (f"r{row}:rstd", "all"))
        for row in active_rows:
            rstd[row] = reg(row, "rstd")
            issue(row, "rstd_store", "store_f32_first", None,
                  (f"r{row}:rstd", "first"), lanes=1,
                  access=("write", "rstd", row * FP32, FP32))
        for row in active_rows:
            for half in range(2):
                issue(row, f"y{half}", "mul_f32", reg(row, f"f{half}") * reg(row, "broadcast"),
                      (f"r{row}:f{half}", f"r{row}:broadcast", "all"))
        for row in active_rows:
            for half in range(2):
                packed = issue(row, f"packed{half}", "cast_f32_bf16",
                               bf16_words(reg(row, f"y{half}")),
                               (f"r{row}:y{half}", "all"))
                begin = half * 64
                output[row, begin:begin + 64] = packed
                issue(row, f"y_store{half}", "store_bf16_pack", None,
                      (f"r{row}:packed{half}", "all"),
                      access=("write", "y", (row * K + begin) * BF16, 64 * BF16))
    return output, rstd, tuple(events)


SOURCE_OPCODES = {
    "create_mask": "CreateMask", "create_mask_all": "CreateMask",
    "create_mask_vl1": "CreateMask", "update_mask": "UpdateMask", "mask_xor": "XorMask",
    "duplicate": "Duplicate", "duplicate_scalar_f32": "Duplicate",
    "duplicate_lane_f32": "Duplicate", "load": "Load", "load_bf16_unpack": "Load",
    "broadcast_load": "LoadBRC", "store": "Store", "store_f32_first": "Store",
    "store_bf16_pack": "Store", "cast_bf16_f32": "CastIn", "cast_f32_bf16": "CastOut",
    "add": "Add", "add_f32": "Add", "mul_f32": "Mul", "muls": "Muls",
    "adds_f32": "Adds", "sqrt_f32": "Sqrt", "div_f32": "Div",
    "reduce_sum_f32": "ReduceSUM", "interleave": "Interleave",
    "deinterleave": "DeInterleave",
}


def source_instruction_counts(events):
    """Aggregate executed source calls; tuple fields are calls/lanes/read/write."""
    counts = {}
    for event in events:
        if event["op"] == "barrier":
            continue
        values = counts.setdefault(SOURCE_OPCODES[event["op"]], [0, 0, 0, 0])
        values[0] += 1
        values[1] += event["active_lanes"]
        if event["access"]:
            direction, _, _, size = event["access"]
            values[2 if direction == "read" else 3] += size
    return {opcode: tuple(values) for opcode, values in counts.items()}


def source_register_capacity(events):
    """Apply 950 Reg issue/write/UB-port rules to the independent event stream.

    The official single/dual issue table defines one single-issue call per
    cycle or two eligible calls, with two data Reg writes and at most one UB
    store. These are capacity demands; no event dependency gets a cycle count.
    """
    result = Counter()
    shuffles = {"interleave", "deinterleave"}
    reductions = {"reduce_sum_f32"}
    stores = {"store", "store_f32_first", "store_bf16_pack"}
    predicates = {"create_mask", "create_mask_all", "create_mask_vl1",
                  "update_mask", "mask_xor"}
    loads = {"load", "load_bf16_unpack", "broadcast_load"}
    for event in events:
        op = event["op"]
        if op == "barrier":
            continue
        result["reg_issue_slots"] += 2 if op in shuffles | reductions | stores else 1
        if op not in predicates | stores:
            result["reg_data_writes"] += 2 if op in shuffles else 1
        if event["access"]:
            result["reg_ub_accesses"] += 1
            if event["access"][0] == "write":
                result["reg_ub_stores"] += 1
        if op in shuffles:
            result["reg_shuffle_lanes"] += event["active_lanes"]
        elif op in reductions:
            result["reg_reduce_lanes"] += event["active_lanes"]
        elif op in {"sqrt_f32", "div_f32"}:
            result["reg_special_lanes"] += event["active_lanes"]
        elif op not in predicates | loads | stores:
            result["reg_alu_lanes"] += event["active_lanes"]
    return result


def source_copy_oracle(params):
    """Enumerate the operand loads and Fixpipe calls in each source pack."""
    heads = params["heads"]
    ratio = heads // params["key_heads"]
    vdim = params["dv"]
    pack_size = 3 if ratio == 3 else 4
    tasks = [
        (head, min(BT, length - start))
        for length in params["lengths"] for start in range(0, length, BT)
        for head in range(heads)
    ]
    l0_tiles, fix_tiles, ub_tiles = [], [], []
    for start in range(0, len(tasks), pack_size):
        pack = tasks[start:start + pack_size]
        # Stage4_PrefillI loads I into L0 once, then reuses it for all heads.
        l0_tiles.extend([(BT, BT, FP32)] * 2)
        for head, rows in pack:
            # Stage3 copies I_vcs, then packs two diagonal leaves. The
            # CANN UB->UB DataCopy implementation uses PIPE_V, not MTE3.
            ub_tiles.extend(((32, 64, FP32), (32, 32, FP32), (32, 32, FP32)))
            if head % ratio == 0:
                l0_tiles.extend([(BT, K, BF16)] * 2)
                dual = ratio > 2 or (ratio == 2 and len(pack) < 4)
                # DumpKktToUb uses two separate Fixpipe calls, not broadcast.
                fix_tiles.extend([(BT, BT, FP32)] * (2 if dual else 1))
            l0_tiles.extend([(BT, BT, FP32)] * 2)  # LeafLeft @ -L
            fix_tiles.append((BT, BT, FP32))       # Y -> GM
            l0_tiles.extend([(BT, BT, FP32)] * 4)  # I@Left and Y@Right
            if params.get("output_a", True):
                fix_tiles.append((rows, BT, BF16))
            if rows < BT or not params.get("output_a", True):
                fix_tiles.append((BT, BT, BF16))   # A -> L1
            # W and each 128-column U half reload both BF16 L0 operands.
            for _ in range(1 + vdim // K):
                l0_tiles.extend(((BT, BT, BF16), (BT, K, BF16)))
                fix_tiles.append((rows, K, BF16))
    return {
        "mte1": sum(rows * columns * width for rows, columns, width in l0_tiles),
        "fixpipe": sum(rows * columns * width for rows, columns, width in fix_tiles),
        "ub_copy": sum(rows * columns * width for rows, columns, width in ub_tiles),
    }


def oracle_work(params: dict, cores: int = 32) -> Tally:
    """Enumerate every (sequence, chunk, value-head) task literally."""
    lengths = [int(v) for v in params["lengths"]]
    hv_count = int(params["heads"])
    hk_count = int(params["key_heads"])
    gqa = hv_count // hk_count
    vdim = int(params["dv"])
    gate_bytes = 4 if params.get("gate_dtype", "bf16") == "fp32" else 2
    sigmoid = params.get("use_beta_sigmoid_in_kernel", True)
    exp2 = params.get("use_exp2", False)
    out = Tally()

    # Stage 0: once on each launched core (two AIVs each).
    for _ in range(cores):
        out.add("vector", 2 * (32 * 64 * 7) + 2 * (64 * 64) + 64 * 64 * 4)
        out.add("mte3", 8 * (64 * 64 * FP32) + 64 * 64 * FP32)

    for length in lengths:
        n_chunks = (length + BT - 1) // BT
        for local in range(n_chunks):
            start = local * BT
            rows = min(BT, length - start)
            tail = rows < BT
            _, cumsum_events = source_cumsum_registers(np.zeros(rows))
            _, _, norm_events = source_l2norm_registers(np.zeros((rows, K), dtype=np.uint16))
            cumsum_capacity = source_register_capacity(cumsum_events)
            norm_capacity = source_register_capacity(norm_events)
            for hv in range(hv_count):
                owns_hk = gqa <= 1 or (hv % gqa) == 0

                # Stage 1 gate/beta, every value head.
                vector = 0
                if tail:
                    vector += 2 * BT
                vector += rows if gate_bytes == 2 else 0
                vector += 8 * rows if sigmoid else rows if gate_bytes == 2 else 0
                reads = 2 * rows * gate_bytes
                writes = 2 * rows * FP32
                out.add("vector", vector)
                out.add("hbm", reads + writes)
                out.add("mte2", reads)
                out.add("mte3", writes)
                for role, amount in cumsum_capacity.items():
                    out.add(role, amount)

                if owns_hk:
                    # Stage 1 key-head normalization and k' L1 upload.
                    vector = 0
                    if tail:
                        vector += BT * K + 3 * BT * K
                    reads = 2 * rows * K * BF16
                    gm_writes = 2 * rows * K * BF16 + 2 * rows * FP32
                    out.add("vector", vector)
                    out.add("hbm", reads + gm_writes)
                    out.add("mte2", reads)
                    out.add("mte3", gm_writes + BT * K * BF16)
                    for role, amount in norm_capacity.items():
                        out.add(role, 2 * amount)

                    # Stage 2 KKT, once per key head.
                    out.matrix_flops("bf16", 2 * BT * BT * K)

                # Stage 3 WY leaves.
                out.add("vector", (7 + exp2) * BT * BT + BT * BT + VCS_OPS)
                out.add("mte3", BT * BT * FP32 + 2 * (BT // 2) ** 2 * FP32)

                # Stage 4a prefill I@I; 4b LeafLeft@(-L) and Y write/read.
                out.matrix_flops("fp32", 2 * BT ** 3)
                out.matrix_flops("fp32", 2 * BT ** 3)
                out.add("hbm", 2 * BT * BT * FP32)
                out.add("mte2", BT * BT * FP32)

                # Stage 5 two more 64-cubed products and A output path.
                out.matrix_flops("fp32", 2 * (2 * BT ** 3))
                a_write = rows * BT * BF16
                a_read = BT * BT * BF16 if rows == BT else 0
                out.add("hbm", a_write + a_read)
                out.add("mte2", a_read)

                # Stage 6 beta-scaled V and K_hat.
                vector = 0
                if tail:
                    vector += BT * vdim + BT * K
                # The caller passes chunkSize to the row scalers, so padded
                # zero rows still consume vector work.
                vector += 3 * BT * vdim + (5 + exp2) * BT * K
                reads = rows * vdim * BF16 + rows * K * BF16
                out.add("vector", vector)
                out.add("hbm", reads)
                out.add("mte2", reads)
                out.add("mte3", BT * vdim * BF16 + BT * K * BF16)

                # Stage 7 W and U (BF16 products, M-row stores).
                out.matrix_flops("bf16", 2 * BT * K * BT + 2 * BT * vdim * BT)
                out.add("hbm", rows * (K + vdim) * BF16)
                out.add("mte3", rows * (K + (vdim if vdim == 128 else 0)) * BF16)
    for resource, amount in source_copy_oracle(params).items():
        out.add(resource, amount)
    return out


def prepared_plan(params: dict):
    op = gdn.make_op(params)
    geo = geometry()
    prepared, plan = PlannedRecipe(gdn.build).prepare(op, geo)
    return op, prepared, plan


# ---------------------------------------------------------------------------
# Work and matrix oracles.
# ---------------------------------------------------------------------------

def test_per_task_work_oracle_matches_plan_for_every_case():
    for case in cases():
        op, prepared, plan = prepared_plan(case["params"])
        oracle = oracle_work(case["params"])
        assert plan.work.matrix == dict(oracle.matrix)
        assert plan.work.resources == dict(oracle.resources)
        assert prepared.flops == sum(oracle.matrix.values())
        assert dict(prepared.matrix_flops) == dict(oracle.matrix)
        assert prepared.vector_ops == oracle.resources["vector"]
        assert prepared.hbm_bytes == oracle.resources["hbm"]
        assert prepared.mte2_bytes == oracle.resources["mte2"]
        assert prepared.mte3_bytes == oracle.resources["mte3"]


@pytest.mark.parametrize("heads,key_heads,vdim,lengths,output_a", [
    (2, 1, 128, [17], True),       # Partial GQA2: two KKT Fixpipe calls.
    (4, 2, 128, [17], True),       # Full GQA2: each KKT stays on its owner AIV.
    (3, 1, 256, [64, 17], True),   # V256: three W/U MMAD operand pairs.
    (4, 1, 128, [64, 17], False),
    (4, 4, 256, [64], True),
])
def test_l0_operand_reuse_and_fixpipe_destinations_follow_source_calls(
        heads, key_heads, vdim, lengths, output_a):
    value = dict(native_flags(), heads=heads, key_heads=key_heads, dv=vdim,
                 lengths=lengths, dk=K, chunk=BT, dtype="bf16")
    value["output_a"] = output_a
    op = gdn.make_op(value)
    plan = gdn.build(op, geometry())
    for resource, expected in source_copy_oracle(value).items():
        assert plan.work.resources[resource] == expected
    report = verify_operator(control_case(split_copy=True), op, gdn.build)
    assert report["status"] == "passed", report


def layout_totals(params: dict, *, cores: int = 32) -> dict:
    """Closed-form totals from token/head addresses, independent of pack walk."""
    lengths = [int(v) for v in params["lengths"]]
    hv_count = int(params["heads"])
    hk_count = int(params["key_heads"])
    vdim = int(params["dv"])
    tokens = sum(lengths)
    chunks = sum((length + BT - 1) // BT for length in lengths)
    full_chunks = sum(length // BT for length in lengths)
    tails = chunks - full_chunks
    total_tasks = chunks * hv_count

    reads = (
        hv_count * tokens * 2 * 2                 # g and beta, bf16
        + hk_count * tokens * K * BF16             # k, once per key head
        + hk_count * tokens * K * BF16             # q, once per key head
        + chunks * hv_count * BT * BT * FP32       # Stage 4 Y read-back
        + full_chunks * hv_count * BT * BT * BF16  # Stage 5 full A read-back
        + hv_count * tokens * vdim * BF16          # Stage 6 v
        + hv_count * tokens * K * BF16             # Stage 6 k_hat
    )
    gm_writes = (
        hv_count * tokens * FP32 * 2               # g_out and beta_out
        + hk_count * tokens * K * BF16 * 2         # k_hat and q_hat
        + hk_count * tokens * FP32 * 2             # k_rstd and q_rstd
        + chunks * hv_count * BT * BT * FP32       # Stage 4 Y write
        + hv_count * tokens * BT * BF16            # Stage 5 A write (all rows)
        + hv_count * tokens * K * BF16             # Stage 7 W
        + hv_count * tokens * vdim * BF16          # Stage 7 U
    )
    onchip_writes = (
        cores * (8 * BT * BT * FP32 + BT * BT * FP32)  # Stage 0 leaf/I uploads
        + chunks * hk_count * BT * K * BF16        # Stage 1 k' full 64-row UB->L1
        + chunks * hv_count * (BT * BT * FP32 + 2 * (BT // 2) ** 2 * FP32)
        + chunks * hv_count * (BT * vdim * BF16 + BT * K * BF16)
    )
    fix_gm_writes = (
        chunks * hv_count * BT * BT * FP32
        + hv_count * tokens * BT * BF16
        + (hv_count * tokens * vdim * BF16 if vdim > 128 else 0)
    )
    fp32 = chunks * hv_count * 4 * (2 * BT ** 3)
    bf16 = (
        chunks * hk_count * (2 * BT * BT * K)
        + chunks * hv_count * (2 * BT * K * BT + 2 * BT * vdim * BT)
    )
    vector = (
        cores * (2 * (32 * 64 * 7) + 2 * (64 * 64) + 64 * 64 * 4)
        + tails * hv_count * (2 * BT)
        + hv_count * tokens
        + 8 * hv_count * tokens
        + tails * hk_count * (BT * K + 3 * BT * K)
        + chunks * hv_count * (7 * BT * BT + BT * BT + VCS_OPS)
        + tails * hv_count * (BT * vdim + BT * K)
        + chunks * hv_count * (3 * BT * vdim + 5 * BT * K)
    )
    return {
        "tokens": tokens, "chunks": chunks, "full_chunks": full_chunks,
        "tasks": total_tasks, "mte2": reads,
        "mte3": gm_writes - fix_gm_writes + onchip_writes,
        "hbm": reads + gm_writes, "onchip": onchip_writes,
        "fix_gm": fix_gm_writes,
        "fp32": fp32, "bf16": bf16, "vector": vector,
    }


def test_token_address_layout_and_stage_totals_are_conserved():
    for case in cases():
        params = case["params"]
        op, prepared, plan = prepared_plan(params)
        expected = layout_totals(params)
        assert plan.work.resources["mte2"] == expected["mte2"]
        assert plan.work.resources["mte3"] == expected["mte3"]
        assert plan.work.resources["hbm"] == expected["hbm"]
        assert plan.work.matrix.get("fp32", 0) == expected["fp32"]
        assert plan.work.matrix.get("bf16", 0) == expected["bf16"]
        assert plan.work.resources["vector"] == expected["vector"]
        assert prepared.hbm_bytes == expected["hbm"]
        assert prepared.vector_ops == expected["vector"]


def test_four_fp32_products_and_key_head_kk_time():
    for case in cases():
        params = case["params"]
        op, prepared, plan = prepared_plan(params)
        lengths = [int(v) for v in params["lengths"]]
        hv = int(params["heads"])
        hk = int(params["key_heads"])
        chunks = sum((length + BT - 1) // BT for length in lengths)
        one_product = 2 * BT ** 3
        assert plan.work.matrix["fp32"] == chunks * hv * 4 * one_product
        # KKT is once per key head, not once per value head.
        kkt = chunks * hk * (2 * BT * BT * K)
        wu = chunks * hv * (2 * BT * K * BT + 2 * BT * int(params["dv"]) * BT)
        assert plan.work.matrix["bf16"] == kkt + wu
        assert hk < hv  # cases exercise grouped key heads
        # A triangular-solve shortcut such as n^3/3 would not be a multiple of
        # the four dense 64-cubed products used by the source.
        assert plan.work.matrix["fp32"] % (4 * one_product) == 0


def test_owner_normalization_is_per_key_head_and_not_per_value_head():
    params = dict(alternate_flags(), lengths=[64], heads=48, key_heads=16,
                  dk=128, dv=128, chunk=64, dtype="bf16")
    with_norm = gdn.make_op(params)
    without_norm = gdn.make_op({**params, "use_qk_l2norm_in_kernel": False})
    p_norm = gdn.build(with_norm, geometry())
    p_plain = gdn.build(without_norm, geometry())
    dhbm = p_norm.work.resources["hbm"] - p_plain.work.resources["hbm"]
    dvec = p_norm.work.resources["vector"] - p_plain.work.resources["vector"]
    owners = 16
    # q read + k_hat/q_hat + two rstd vectors, once per key head.
    assert dhbm == owners * (64 * K * BF16 + 2 * 64 * K * BF16 + 2 * 64 * FP32)
    # Both k and q normalize once per key head. Their typed capacities are
    # separate from the remaining aggregate vector work and from GM traffic.
    assert dvec == 0
    _, _, events = source_l2norm_registers(np.zeros((BT, K), dtype=np.uint16))
    for role, amount in source_register_capacity(events).items():
        assert p_norm.work.resources[role] - p_plain.work.resources.get(role, 0) == \
            owners * 2 * amount


def test_beta_scaling_feeds_w_u_and_tail_pads():
    params = dict(alternate_flags(), lengths=[65], heads=48, key_heads=16,
                  dk=128, dv=128, chunk=64, dtype="bf16")
    case = layout_totals(params)
    _, _, plan = prepared_plan(params)
    onchip = (plan.work.resources["mte3"]
              - (plan.work.resources["hbm"] - plan.work.resources["mte2"]
                 - case["fix_gm"]))
    assert onchip == case["onchip"]
    # A full chunk reads A back; a 1-row tail writes A but keeps the L1 fill
    # on-chip, so the read-back is only the full chunk.
    full = 1
    tail = 1
    assert case["onchip"] == (
        32 * (8 * BT * BT * FP32 + BT * BT * FP32)
        + (full + tail) * 48 * (BT * BT * FP32 + 2 * (BT // 2) ** 2 * FP32)
        + (full + tail) * 48 * (BT * 128 * BF16 + BT * K * BF16)
        + (full + tail) * 16 * BT * K * BF16
    )

    # Removing A output changes only the GM A write/read path.  Full chunks
    # write+read A; the 1-row tail writes A but fills L1 directly from L0C.
    with_a_op, _, with_a = prepared_plan(params)
    without_a_op, _, without_a = prepared_plan({**params, "output_a": False})
    assert dict(with_a_op.metadata)["output_a"] is True
    assert dict(without_a_op.metadata)["output_a"] is False
    dhbm = with_a.work.resources["hbm"] - without_a.work.resources["hbm"]
    dfix = with_a.work.resources["fixpipe"] - without_a.work.resources["fixpipe"]
    assert dhbm == 48 * ((BT * BT * BF16) * 2 + tail * BT * BF16)
    assert dfix == 48 * tail * BT * BF16
    assert with_a.work.resources["mte3"] == without_a.work.resources["mte3"]


def test_alternate_source_flags_and_v256_are_bindable():
    base = dict(alternate_flags(), lengths=[65], heads=48, key_heads=16,
                dk=128, dv=128, chunk=64, dtype="bf16")
    base_op, _, base_plan = prepared_plan(base)

    # exp2 adds one multiply per gate element and one per k element in Stage 6.
    exp2 = gdn.make_op({**base, "use_exp2": True})
    exp2_plan = gdn.build(exp2, geometry())
    assert exp2_plan.work.resources["vector"] - base_plan.work.resources["vector"] == (
        2 * 48 * (BT * BT + BT * K))
    assert verify_operator(control_case(), exp2, gdn.build)["status"] == "passed"

    # Fused softplus replaces the bf16 cast and reads a_log/dt scalars.
    gate = gdn.make_op({**base, "use_gate_in_kernel": True, "has_dt_bias": True})
    gate_plan = gdn.build(gate, geometry())
    assert gate_plan.work.resources["vector"] - base_plan.work.resources["vector"] == (
        13 * 65 * 48 + 2 * 2 * 48)
    assert gate_plan.work.resources["mte2"] - base_plan.work.resources["mte2"] == 4 * 2 * 48
    assert verify_operator(control_case(), gate, gdn.build)["status"] == "passed"

    # A float32 gate copy removes the bf16->fp32 cast but doubles gate bytes.
    fp32 = gdn.make_op({**base, "gate_dtype": "fp32"})
    fp32_plan = gdn.build(fp32, geometry())
    assert fp32_plan.work.resources["vector"] == base_plan.work.resources["vector"] - 65 * 48
    assert fp32_plan.work.resources["mte2"] - base_plan.work.resources["mte2"] == 4 * 65 * 48
    assert verify_operator(control_case(), fp32, gdn.build)["status"] == "passed"

    # V=256 executes a second 128-column U product and a wider Stage 6.
    v256 = gdn.make_op({**base, "dv": 256})
    v256_plan = gdn.build(v256, geometry())
    assert v256_plan.work.matrix["bf16"] - base_plan.work.matrix["bf16"] == (
        2 * 48 * (2 * BT * 128 * BT))
    assert v256_plan.work.resources["hbm"] - base_plan.work.resources["hbm"] == (
        2 * 48 * 65 * 128 * BF16)
    assert verify_operator(control_case(), v256, gdn.build)["status"] == "passed"


# ---------------------------------------------------------------------------
# Numerical oracles: normalization, cumsum, gate, block inverse and beta scale.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows, expected", [
    (1, (0, 0, 1, 0, 1, 1, 0, 0)),
    (17, (16, 26, 1, 1, 2, 3, 1, 1)),
    (32, (16, 26, 1, 1, 2, 3, 1, 1)),
    (33, (18, 26, 1, 3, 3, 4, 3, 2)),
    (64, (18, 26, 1, 3, 3, 4, 3, 2)),
])
def test_cumsum_register_shuffles_match_prefix_sum_and_source_calls(rows, expected):
    values = np.arange(rows, dtype=np.float64) ** 2 - 0.25 * np.arange(rows)
    result, events = source_cumsum_registers(values, scale=1 / math.log(2))
    calls = Counter(event["op"] for event in events)
    np.testing.assert_allclose(result, np.cumsum(values) / math.log(2), rtol=1e-14)
    add, shuffle, load, broadcast, store, masks, xor, barriers = expected
    assert calls == Counter({
        "add": add, "interleave": shuffle, "deinterleave": shuffle,
        "load": load, "broadcast_load": broadcast, "store": store,
        "update_mask": masks, "mask_xor": xor, "barrier": barriers,
        "create_mask": 2, "duplicate": 1, "muls": 1,
    })


def test_cumsum_inventory_and_capacity_follow_executed_source_for_every_tail():
    for rows in range(1, BT + 1):
        values = np.arange(rows, dtype=np.float64) ** 2 - 0.25 * np.arange(rows)
        result, events = source_cumsum_registers(values)
        np.testing.assert_allclose(result, np.cumsum(values), rtol=1e-14)
        inventory = gdn_registers.cumsum(rows)
        actual = {
            name: (count.calls, count.active_lanes, count.read_bytes, count.write_bytes)
            for name, count in inventory.counts.items()
        }
        assert actual == source_instruction_counts(events), rows
        assert inventory.barriers == sum(event["op"] == "barrier" for event in events), rows
        assert inventory.capacity_work(gdn_registers.CLASSES).resources == \
            dict(source_register_capacity(events)), rows
        # The masked final write carries only real rows; intermediate stores
        # retain the whole register before BRC reads of a single FP32 scalar.
        stores = [event for event in events if event["op"] == "store"]
        assert stores[-1]["access"][3] == rows * FP32
        assert all(event["access"][3] == 256 for event in stores[:-1])
        assert all(event["access"][3] == FP32 for event in events
                   if event["op"] == "broadcast_load")


def test_l2norm_inventory_and_capacity_follow_bf16_pair_and_tail_source():
    for rows in range(1, BT + 1):
        values = np.arange(rows * K, dtype=np.float32).reshape(rows, K) / 128
        _, _, events = source_l2norm_registers(bf16_words(values))
        inventory = gdn_registers.l2norm(rows, "bf16")
        actual = {
            name: (count.calls, count.active_lanes, count.read_bytes, count.write_bytes)
            for name, count in inventory.counts.items()
        }
        assert actual == source_instruction_counts(events), rows
        assert inventory.barriers == 0
        assert inventory.capacity_work(gdn_registers.CLASSES).resources == \
            dict(source_register_capacity(events)), rows
        # Each VL1 operation still issues a call; active-lane totals are not
        # a claim that the masked Sqrt/Div dependency latency is one cycle.
        for opcode in ("Adds", "Sqrt", "Div"):
            assert inventory.counts[opcode].calls == rows
            assert inventory.counts[opcode].active_lanes == rows


@pytest.mark.parametrize("rows", (1, 2, 3, 17, 64))
def test_l2norm_bf16_row_pairs_and_odd_tail_match_scalar_math_and_ub_intervals(rows):
    channels = np.arange(K, dtype=np.float32)
    raw = (np.arange(rows)[:, None] + 1) * ((channels % 17) - 8) / 16
    raw += (channels + 1) / 512
    if rows > 1:
        raw[1] = 0
    words = bf16_words(raw)
    output, rstd, events = source_l2norm_registers(words)
    source = bf16_values(words).astype(np.float64)
    expected_rstd = np.array([
        1 / math.sqrt(math.fsum(float(value) ** 2 for value in row) + float(np.float32(1e-6)))
        for row in source
    ])
    np.testing.assert_allclose(rstd[:rows], expected_rstd, rtol=2e-6)
    np.testing.assert_allclose(bf16_values(output[:rows]), source * expected_rstd[:, None],
                               rtol=2**-8 + 2e-6, atol=1e-8)
    assert np.all(output[rows:] == 0x5a5a)
    assert np.all(rstd[rows:] == -17.0)

    counts = Counter(event["op"] for event in events)
    per_row = {
        "load_bf16_unpack": 2, "cast_bf16_f32": 2, "mul_f32": 4,
        "add_f32": 1, "reduce_sum_f32": 1, "adds_f32": 1,
        "sqrt_f32": 1, "div_f32": 1, "duplicate_lane_f32": 1,
        "store_f32_first": 1, "cast_f32_bf16": 2, "store_bf16_pack": 2,
    }
    expected_calls = Counter({name: count * rows for name, count in per_row.items()})
    expected_calls.update(create_mask_all=1, create_mask_vl1=1, duplicate_scalar_f32=1)
    assert counts == expected_calls
    intervals = {}
    for event in events:
        if event["access"] is not None:
            direction, tensor, offset, size = event["access"]
            intervals.setdefault((direction, tensor), []).extend(range(offset, offset + size))
    assert intervals == {
        ("read", "x"): list(range(rows * K * BF16)),
        ("write", "rstd"): list(range(rows * FP32)),
        ("write", "y"): list(range(rows * K * BF16)),
    }
    for event in events:
        if event["op"] in {"sqrt_f32", "div_f32", "adds_f32", "store_f32_first"}:
            assert event["active_lanes"] == 1


def test_l2norm_paired_rows_have_independent_data_paths_and_separate_rstd_store():
    _, _, events = source_l2norm_registers(bf16_words(np.arange(3 * K).reshape(3, K) / 128))
    by_id = {event["id"]: event for event in events}
    position = {event["id"]: index for index, event in enumerate(events)}

    def ancestors(name):
        result, pending = set(), list(by_id[name]["inputs"])
        while pending:
            parent = pending.pop()
            if parent not in result:
                result.add(parent)
                pending.extend(by_id[parent]["inputs"])
        return result

    for row in (0, 1):
        chain = ancestors(f"r{row}:rstd")
        assert {f"r{row}:{part}" for part in (
            "x0", "x1", "f0", "f1", "square0", "square1", "halves", "sum", "epsilon", "sqrt"
        )} <= chain
        assert "one" in chain
        assert not any(name.startswith(f"r{1 - row}:") for name in chain)
        assert f"r{row}:broadcast" not in ancestors(f"r{row}:rstd_store")
        for half in range(2):
            output_chain = ancestors(f"r{row}:y_store{half}")
            assert f"r{row}:f{half}" in output_chain
            assert f"r{row}:broadcast" in output_chain
            assert f"r{row}:rstd_store" not in output_chain
    assert position["r1:f1"] < position["r0:square0"]
    assert position["r1:square1"] < position["r0:halves"]
    assert position["r0:sqrt"] < position["r1:sqrt"] < position["r0:rstd"]
    assert "r0:sqrt" not in ancestors("r1:sqrt")
    assert {event["row"] for event in events if event["group"] == 0} == {0, 1}
    assert {event["row"] for event in events if event["group"] == 1} == {2}
    assert position["r1:y_store1"] < position["r2:x0"]


def _causal_gate(khat, beta_eff, gcum):
    rows = khat.shape[0]
    gate = np.zeros((rows, rows), dtype=np.float64)
    for i in range(rows):
        for j in range(i):
            delta = np.clip(gcum[i] - gcum[j], -50.0, 50.0)
            gate[i, j] = beta_eff[i] * math.exp(delta) * float(khat[i] @ khat[j])
    return gate


def _block_product_inverse(gate):
    half = gate.shape[0] // 2
    eye = np.eye(gate.shape[0])
    right = np.linalg.inv(np.eye(half) + gate[:half, :half])
    left = np.linalg.inv(np.eye(half) + gate[half:, half:])
    left_block = np.zeros_like(gate)
    left_block[half:, half:] = left
    right_block = np.zeros_like(gate)
    right_block[:half, :half] = right
    y = eye + left_block @ (-gate)
    return y, left_block + y @ right_block


def test_numpy_l2norm_cumsum_gate_and_four_product_block_inverse():
    rng = np.random.default_rng(20261028)
    raw_k = rng.normal(size=(BT, K)).astype(np.float64)
    g = rng.normal(size=BT).astype(np.float64) * 0.2
    beta = rng.normal(size=BT).astype(np.float64)

    rstd = 1.0 / np.sqrt(np.sum(raw_k * raw_k, axis=1) + 1e-6)
    khat = raw_k * rstd[:, None]
    assert np.allclose(np.sum(khat * khat, axis=1), 1.0, atol=2e-5)

    gcum = np.cumsum(g)
    assert np.allclose(np.diff(gcum), g[1:])
    beta_eff = 1.0 / (1.0 + np.exp(-beta))
    gate = _causal_gate(khat, beta_eff, gcum)
    assert np.allclose(np.triu(gate), 0.0)

    y, a_num = _block_product_inverse(gate)
    a_ref = np.linalg.inv(np.eye(BT) + gate)
    assert np.allclose(a_num, a_ref, atol=2e-10)
    # The prefill I@I, LeafLeft@(-L), I@LeafLeft and Y@LeafRight products are
    # all dense BT x BT x BT products; count them explicitly.
    assert gate.shape == (BT, BT)
    assert y.shape == (BT, BT)

    v = rng.normal(size=(BT, 128)).astype(np.float64)
    kbg = khat * (beta_eff * np.exp(gcum))[:, None]
    vb = v * beta_eff[:, None]
    w_ref = np.linalg.solve(np.eye(BT) + gate, kbg)
    u_ref = np.linalg.solve(np.eye(BT) + gate, vb)
    assert np.allclose(a_num @ kbg, w_ref, atol=2e-10)
    assert np.allclose(a_num @ vb, u_ref, atol=2e-10)
    # Beta scaling is real: removing beta changes W and U.
    assert not np.allclose(w_ref, a_ref @ khat)
    assert not np.allclose(u_ref, a_ref @ v)


def test_tail_rows_pad_to_identity_and_leave_the_valid_block_unchanged():
    rng = np.random.default_rng(7)
    valid = 5
    raw = np.zeros((BT, K))
    raw[:valid] = rng.normal(size=(valid, K))
    g = np.zeros(BT)
    g[:valid] = rng.normal(size=valid) * 0.1
    beta = np.zeros(BT)
    beta[:valid] = rng.normal(size=valid)
    rstd = 1.0 / np.sqrt(np.sum(raw * raw, axis=1) + 1e-6)
    khat = raw * rstd[:, None]
    beta_eff = np.zeros(BT)
    beta_eff[:valid] = 1.0 / (1.0 + np.exp(-beta[:valid]))
    gate = _causal_gate(khat, beta_eff, np.cumsum(g))
    _, a_padded = _block_product_inverse(gate)

    small = np.zeros((valid, K))
    small[:] = raw[:valid]
    small_rstd = 1.0 / np.sqrt(np.sum(small * small, axis=1) + 1e-6)
    small_hat = small * small_rstd[:, None]
    small_gate = _causal_gate(small_hat, beta_eff[:valid], np.cumsum(g[:valid]))
    a_small = np.linalg.inv(np.eye(valid) + small_gate)
    assert np.allclose(a_padded[:valid, :valid], a_small, atol=2e-10)
    assert np.allclose(a_padded[valid:, valid:], np.eye(BT - valid))


# ---------------------------------------------------------------------------
# Varlen chunk/address oracle.
# ---------------------------------------------------------------------------

def test_varlen_chunk_table_and_token_offsets_match_cu_seqlens():
    lengths = [63, 1, 65]
    chunks = gdn.PackedChunks.from_lengths(lengths)
    expected = [
        (0, 0, 0, 63),
        (1, 0, 63, 1),
        (2, 0, 64, 64),
        (2, 1, 128, 1),
    ]
    assert list(chunks.all_ranges()) == expected
    assert chunks.total == 4 and chunks.padded_tokens == 256

    # cu_seqlens and the canonical chunk_indices built by the host wrapper.
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    indices = []
    for seq, length in enumerate(lengths):
        for local in range((length + BT - 1) // BT):
            indices.append(seq)
            indices.append(local)
    assert cu == [0, 63, 64, 129]
    assert indices == [0, 0, 1, 0, 2, 0, 2, 1]

    total = cu[-1]
    hk, hv, gqa = 8, 32, 4
    seen_q = set()
    seen_v = set()
    qhat_rows = 0
    seq_ends = np.cumsum([0] + lengths).tolist()
    for seq, local, token_start, rows in chunks.all_ranges():
        assert token_start + rows <= seq_ends[seq + 1]
        for hk_idx in range(hk):
            base = ((0 * hk + hk_idx) * total + token_start) * K
            for offset in range(base, base + rows * K):
                assert offset not in seen_q
                seen_q.add(offset)
            qhat_rows += rows
        for hv_idx in range(hv):
            base = ((0 * hv + hv_idx) * total + token_start) * 128
            for offset in range(base, base + rows * 128):
                assert offset not in seen_v
                seen_v.add(offset)
    assert qhat_rows == hk * total
    assert len(seen_q) == hk * total * K
    assert len(seen_v) == hv * total * 128


# ---------------------------------------------------------------------------
# Pack ownership and execution contract.
# ---------------------------------------------------------------------------

def test_tasks_per_pack_and_gqa2_middle_swap():
    for gqa in (1, 2, 3, 4):
        params = dict(alternate_flags(), lengths=[64], heads=4 * gqa,
                      key_heads=4, dk=128, dv=128, chunk=64, dtype="bf16")
        op = gdn.make_op(params)
        meta = dict(op.metadata)
        assert meta["gqa_ratio"] == gqa
        assert meta["tasks_per_pack"] == (3 if gqa == 3 else 4)
        if gqa == 2:
            assert [gdn._pack_work_id(0, 4, t, 2) for t in range(4)] == [0, 2, 1, 3]
        else:
            assert [gdn._pack_work_id(0, 4 if gqa != 3 else 3, t, gqa)
                    for t in range(4 if gqa != 3 else 3)] == list(
                        range(4 if gqa != 3 else 3))


@pytest.mark.parametrize("lengths", ([64], [63, 1, 65]))
def test_native_fp32_gate_no_sigmoid_still_publishes_beta_eff(lengths):
    value = dict(native_flags(), lengths=lengths, heads=48, key_heads=16,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    op, _, plan = prepared_plan(value)
    expected = oracle_work(value)
    assert plan.work.resources == dict(expected.resources)
    assert plan.work.matrix == dict(expected.matrix)
    sigmoid = gdn.build(gdn.make_op({
        **value, "use_beta_sigmoid_in_kernel": True,
    }), geometry())
    # This explicit kernel declaration requests beta_eff for both branches.
    # The no-sigmoid public composite instead omits that optional output.
    assert sigmoid.work.resources["hbm"] == plan.work.resources["hbm"]
    assert sigmoid.work.resources["mte3"] == plan.work.resources["mte3"]
    assert sigmoid.work.resources["vector"] - plan.work.resources["vector"] == \
        8 * sum(lengths) * 48
    report = verify_operator(control_case(split_copy=True), op, gdn.build)
    assert report["status"] == "passed", report
    selected = report["result"]["operator"]["metadata"]["dispatch"]
    assert selected["gate_dtype"] == "fp32" and selected["use_exp2"]
    assert not selected["use_beta_sigmoid_in_kernel"]


def test_native_prefill_exports_both_distinct_normalized_key_head_tensors():
    # The composite allocates distinct q_hat and k_hat outputs, then exports
    # each BNSD tensor as BSND (native API:635-640 and 700-714).
    case = native_frontend_case()
    views = (RequestView("r0", 17, 0, False, "prefill"),
             RequestView("r1", 65, 23, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, mode="prefill")
    exports = {op.recipe: op for op in ops
               if op.recipe in {"q_hat_export", "k_hat_export"}}
    assert set(exports) == {"q_hat_export", "k_hat_export"}
    for op in exports.values():
        assert op.mte2_bytes == op.mte3_bytes == (17 + 65) * 2 * K * BF16
    assert exports["q_hat_export"].op_id != exports["k_hat_export"].op_id


@pytest.mark.parametrize("prefill_lengths", ((65,), (17, 65)))
def test_mixed_batch_keeps_native_prefill_and_separate_decode_recurrence(prefill_lengths):
    # Exact-image gdn.py:555-601 handles decode first;604-653 still calls
    # FLA for the prefill suffix. Mixed mode cannot erase that dispatch.
    case = native_frontend_case()
    views = (RequestView("decode", 1, 127, True, "decode"),) + tuple(
        RequestView(f"prefill{index}", length, 23 if index else 0, False, "prefill")
        for index, length in enumerate(prefill_lengths))
    work = Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)
    native = [op for op in work.ops
              if op.family in {"gdn_prepare", "gdn_state", "gdn_output"}]
    assert {op.family for op in native} == {"gdn_prepare", "gdn_state", "gdn_output"}
    assert len(native) == 3
    for op in native:
        assert dict(op.metadata)["lengths"] == prefill_lengths
    scans = [op for op in work.ops if op.family == "recurrent_vector"]
    assert len(scans) == 1
    assert dict(scans[0].metadata)["sequence_tokens"] == (1,)
    for recipe in ("linear_qkvz", "linear_beta_alpha", "linear_output_norm_gate", "linear_output"):
        assert sum(op.recipe == recipe for op in work.ops) == 1


def test_native_output_copy_writes_only_actual_tokens_of_padded_caller_buffer():
    # gdn.py:416-418 slices the actual input;736-737 writes only that prefix
    # into the outer zero-filled buffer. Padding is not an O-kernel output.
    case = native_frontend_case()
    views = (RequestView("r0", 17, 0, False, "prefill"),
             RequestView("r1", 65, 23, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, tokens=128, mode="prefill")
    output = next(op for op in ops if op.family == "gdn_output")
    assert output.local_shape == (1, 82, 6, K)
    zero = next(op for op in ops if op.recipe == "core_output_zero")
    assert zero.mte3_bytes == 128 * 6 * K * BF16
    copies = [op for op in ops
              if op.recipe == "core_output_copy"]
    assert sum(op.mte2_bytes for op in copies) == 82 * 6 * K * BF16
    assert sum(op.mte3_bytes for op in copies) == 82 * 6 * K * BF16


@pytest.mark.parametrize("decode_count", (0, 2))
def test_native_prefill_views_keep_full_batch_qkv_storage_and_gate_output_ports(decode_count):
    # gdn.py:532-533 rearranges all actual tokens before :614-618 slices the
    # prefill suffix. NumPy views independently supply the resulting addresses.
    case = native_frontend_case()
    views = tuple(RequestView(f"d{i}", 1, 127 + i, True, "decode")
                  for i in range(decode_count)) + (
        RequestView("p0", 17, 0, False, "prefill"),
        RequestView("p1", 65, 23, False, "prefill"))
    actual, heads, key_heads = 82 + decode_count, 6, 2
    mode = "mixed" if decode_count else "prefill"
    ops, _ = Frontend().body(case.model, case.runtime, views, tokens=128, mode=mode)
    q_size, k_size, v_size = actual * key_heads * K, actual * key_heads * K, actual * heads * K
    fused = np.arange(q_size + k_size + v_size, dtype=np.int64)
    bsnd = [
        fused[:q_size].reshape(1, actual, key_heads, K),
        fused[q_size:q_size + k_size].reshape(1, actual, key_heads, K),
        fused[q_size + k_size:].reshape(1, actual, heads, K),
    ]

    def source_addresses(op):
        layout = dict(op.metadata)["source_layout"]
        indices = np.indices(layout["shape"], dtype=np.int64)
        return layout["offset"] + sum(axis * stride for axis, stride in
                                      zip(indices, layout["strides"]))

    for name, array in zip(("q", "k", "v"), bsnd):
        copied = next(op for op in ops if op.recipe == name + "_head_first")
        expected = array[:, decode_count:].transpose(0, 2, 1, 3)
        np.testing.assert_array_equal(source_addresses(copied), expected)
        assert dict(copied.metadata)["source_layout"]["storage_shape"] == (fused.size,)
    concat = next(op for op in ops if op.recipe == "qkv_concatenate")
    assert concat.mte2_bytes == concat.mte3_bytes == fused.size * BF16

    gated = next(op for op in ops if op.family == "gdn_gating")
    assert gated.local_shape == (actual, heads)
    gate_addresses = np.arange(actual * heads, dtype=np.int64).reshape(1, actual, heads)
    decay = next(op for op in ops if op.recipe == "g_head_first")
    beta = next(op for op in ops if op.recipe == "beta_fp32")
    np.testing.assert_array_equal(source_addresses(decay),
                                  gate_addresses[:, decode_count:].transpose(0, 2, 1))
    np.testing.assert_array_equal(source_addresses(beta), gate_addresses[:, decode_count:])
    assert dict(decay.metadata)["source_output"] != dict(beta.metadata)["source_output"]
    assert beta.mte2_bytes == 82 * heads * BF16
    assert beta.mte3_bytes == 82 * heads * FP32


def test_mixed_outputs_concatenate_before_copying_actual_caller_prefix():
    # gdn.py:701-705 materializes cat(decode, prefill), then :737 copies the
    # concatenation into the caller's padded output. They are two transfers.
    case = native_frontend_case()
    views = (RequestView("d0", 1, 127, True, "decode"),
             RequestView("d1", 1, 255, True, "decode"),
             RequestView("p0", 17, 0, False, "prefill"),
             RequestView("p1", 65, 23, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, tokens=128, mode="mixed")
    output = next(op for op in ops if op.family == "gdn_output")
    scans = [op for op in ops if op.family == "recurrent_vector"]
    assert sum(sum(dict(op.metadata)["sequence_tokens"]) for op in scans) == 2
    merged = [op for op in ops
              if dict(op.metadata).get("materialization_reason") == "concatenate"
              and output.op_id in op.deps]
    assert len(merged) == 1
    assert merged[0].mte2_bytes == merged[0].mte3_bytes == 84 * 6 * K * BF16
    copied = [op for op in ops
              if op.recipe == "core_output_copy"]
    assert len(copied) == 1
    assert merged[0].op_id in copied[0].deps
    assert copied[0].mte2_bytes == copied[0].mte3_bytes == 84 * 6 * K * BF16
    destination = dict(copied[0].metadata)["target_layout"]
    assert tuple(destination["shape"]) == (84, 6, K)
    assert tuple(destination["storage_shape"]) == (128, 6, K)
    assert destination["offset"] == 0


@pytest.mark.parametrize("tokens", (1, 2))
def test_split_reshape_and_contiguous_copy_only_when_source_has_a_storage_gap(tokens):
    # Image qwen_gdn_linear_attn.py:818-822 flattens each split and then
    # concatenates. At T=1 flatten is a view; at T=2 it crosses a row gap.
    case = native_frontend_case()
    key_width, value_width = 2 * K, 6 * K
    width = 2 * key_width + value_width
    mixed = np.arange(tokens * width, dtype=np.uint16).reshape(tokens, width)
    q, k, v = np.split(mixed, (key_width, 2 * key_width), axis=1)
    expected_split_bytes = sum(
        2 * value.nbytes for value in (q, k, v)
        if not np.shares_memory(value, value.reshape(-1)))
    ba = np.arange(tokens * 2 * 6, dtype=np.uint16).reshape(tokens, 2 * 6)
    expected_gate_bytes = sum(
        2 * value.nbytes for value in np.split(ba, 2, axis=1)
        if not value.flags.c_contiguous)
    views = (RequestView("r", tokens, 0, False, "prefill"),)
    ops, _ = Frontend().body(case.model, case.runtime, views, mode="prefill")

    def copied_from(shape):
        return sum(op.hbm_bytes for op in ops if op.family == "tensor_copy"
                   and tuple(dict(op.metadata).get("source_layout", {}).get(
                       "storage_shape", ())) == shape)

    assert copied_from(mixed.shape) == expected_split_bytes
    assert copied_from(ba.shape) == expected_gate_bytes
    concat = next(op for op in ops if op.recipe == "qkv_concatenate")
    assert concat.mte2_bytes == concat.mte3_bytes == mixed.nbytes


def test_structural_contract_for_all_cases_and_copy_modes():
    for case in cases():
        op = gdn.make_op(case["params"])
        for split in (False, True):
            result = verify_operator(control_case(split_copy=split), op, gdn.build)
            assert result["status"] == "passed", (case["id"], split, result)
            assert result["finite_completion"] and result["buffers_released"]
            assert result["work_mismatches"] == {}
            assert result["result"]["remaining_buffers"] == {}
            assert result["result"]["live_tasks"] == 0
            assert result["result"]["peak_buffers"] == {}


def test_plan_is_lazy_in_token_count_and_bounded_memory():
    params = dict(native_flags(), lengths=[10 ** 9], heads=48, key_heads=16,
                  dk=128, dv=128, chunk=64, dtype="bf16")
    tracemalloc.start()
    start = time.perf_counter()
    op = gdn.make_op(params)
    plan = gdn.build(op, geometry())
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Only one sequence and a handful of pack metadata objects exist; no
    # per-token or per-pack list was materialized.
    assert elapsed < 2.0
    assert peak < 32 * 1024 ** 2
    assert op.padded_shape[2] == math.ceil(10 ** 9 / BT) * BT
    assert math.isfinite(plan.work.resources["hbm"])


def execute(value, case, trace_limit=0):
    op = gdn.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(gdn.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes, trace_limit=trace_limit)
    prepared, recipe = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    ready = execution.engine.submit("input_ready")
    complete = recipe.lower(context).start(execution.engine, "operator", ready)
    execution.engine.advance(until=(complete,))
    return execution.engine, complete.time


def phase_event(engine, pack, name, phase):
    suffixes = (f":tile{pack}:{name}:{phase}",
                f":tile{pack}:{name}:program:{phase}")
    return next(row for row in engine.trace if row["task"].endswith(suffixes))


def test_k_ready_releases_kkt_before_q_and_gate_finish():
    # prepare.h:337-355 publishes K-ready before K_hat/rstd, Q normalization
    # and gate work. A slow Q vector path must not delay an independent KKT.
    value = dict(native_flags(), lengths=[64], heads=1, key_heads=1,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"].update(bf16=1e8, fp32=1e30)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e6, reg_alu_lanes=1e6, read=1e30, write=1e30,
        mte1=1e30, fixpipe=1e30)
    engine, _ = execute(value, case, trace_limit=10000)
    ready = phase_event(engine, 0, "k_ready_0", "store")
    kkt = phase_event(engine, 0, "s2_0", "compute")
    q = phase_event(engine, 0, "s1q_0", "compute")
    gate = phase_event(engine, 0, "s1gate_0", "compute")
    assert kkt["start_ms"] >= ready["end_ms"]
    assert max(kkt["start_ms"], q["start_ms"]) < min(kkt["end_ms"], q["end_ms"])
    assert gate["start_ms"] >= q["end_ms"]
    assert kkt["end_ms"] < gate["end_ms"]


def test_next_pack_k_preprocessing_overlaps_stage4_before_l1_slot_reuse():
    # prepare.h:330-338 waits Stage4 only at K's L1 write. Full packs may
    # read/normalize into the released S1 UB while prior Stage4 still runs.
    # The common.h:252 notification is PIPE_FIX: Y's MTE2 reload can continue.
    value = dict(native_flags(), lengths=[128], heads=4, key_heads=4,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"].update(bf16=1e30, fp32=1e6)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e10, read=1e30, write=1e30, mte1=1e30, fixpipe=1e30)
    engine, _ = execute(value, case, trace_limit=10000)
    previous_s6 = phase_event(engine, 0, "s6_2", "store")
    previous_y = phase_event(engine, 0, "y_dump_0", "publish")
    next_read = phase_event(engine, 1, "k_ready_0", "load")
    next_norm = phase_event(engine, 1, "k_ready_0", "compute")
    next_upload = phase_event(engine, 1, "k_ready_0", "store")
    assert next_read["start_ms"] >= previous_s6["end_ms"]
    assert next_norm["end_ms"] < previous_y["end_ms"]
    assert next_upload["start_ms"] >= previous_y["end_ms"]


def test_q_and_gate_reads_overlap_k_hat_export_without_overtaking_mte3():
    # prepare.h:341-355 issues K_hat/rstd stores before Q's input read.
    # MTE2 and PIPE_V may run while those stores drain; later Q/g stores
    # retain their issue order on this AIV's single MTE3 stream.
    value = dict(native_flags(), lengths=[64], heads=1, key_heads=1,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"].update(bf16=1e30, fp32=1e30)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e30, read=1e30, write=1e6, mte1=1e30, fixpipe=1e30)
    engine, _ = execute(value, case, trace_limit=10000)
    ready = phase_event(engine, 0, "k_ready_0", "store")
    exported = phase_event(engine, 0, "s1k_export_0", "store")
    q = phase_event(engine, 0, "s1q_0", "compute")
    q_store = phase_event(engine, 0, "s1q_0", "store")
    gate_read = phase_event(engine, 0, "s1gate_0", "load")
    gate_store = phase_event(engine, 0, "s1gate_0", "store")
    assert q["start_ms"] >= ready["end_ms"]
    assert q["end_ms"] < exported["end_ms"]
    assert gate_read["end_ms"] < exported["end_ms"]
    assert q_store["start_ms"] >= exported["end_ms"]
    assert gate_store["start_ms"] >= q_store["end_ms"]


def test_stage3_gate_for_all_local_heads_precedes_kkt_consumption():
    # prepare.h:800-805 executes all local PrepareGate calls before entering
    # the first WaitCubeKktDone. A long KKT therefore cannot block either gate.
    value = dict(native_flags(), lengths=[64], heads=4, key_heads=4,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"].update(bf16=1e5, fp32=1e30)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e8, read=1e30, write=1e30, mte1=1e30, fixpipe=1e30)
    engine, _ = execute(value, case, trace_limit=10000)
    first_kkt = phase_event(engine, 0, "s2_0", "compute")
    for lane in (0, 1):
        first_gate = phase_event(engine, 0, f"s3gate_{lane}", "compute")
        last_gate = phase_event(engine, 0, f"s3identity_{lane + 2}", "store")
        first_apply = phase_event(engine, 0, f"s3mul_{lane}", "compute")
        last_s1 = phase_event(engine, 0, f"s1gate_{lane + 2}", "store")
        assert first_gate["start_ms"] >= last_s1["end_ms"]
        assert last_gate["end_ms"] < first_kkt["end_ms"]
        assert first_apply["start_ms"] >= last_gate["end_ms"]
        assert first_apply["start_ms"] >= phase_event(
            engine, 0, f"s2_{lane}", "publish")["end_ms"]


def test_stage3_neg_l_upload_overlaps_vcs_before_leaf_publication():
    # prepare.h:509-516 uploads -L after packing its diagonal leaves, then
    # runs VCS on PIPE_V while that independent UB->L1 MTE3 copy proceeds.
    value = dict(native_flags(), lengths=[64], heads=1, key_heads=1,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"].update(bf16=1e30, fp32=1e30)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e6, read=1e30, write=1e6, mte1=1e30, fixpipe=1e30)
    engine, _ = execute(value, case, trace_limit=10000)
    packed = phase_event(engine, 0, "s3pack_0", "store")
    neg_l = phase_event(engine, 0, "s3neg_0", "store")
    vcs = phase_event(engine, 0, "s3vcs_0", "compute")
    leaves = phase_event(engine, 0, "s3leaf_0", "store")
    assert neg_l["start_ms"] >= packed["end_ms"]
    assert vcs["start_ms"] >= packed["end_ms"]
    assert max(neg_l["start_ms"], vcs["start_ms"]) < min(
        neg_l["end_ms"], vcs["end_ms"])
    assert leaves["start_ms"] >= max(neg_l["end_ms"], vcs["end_ms"])
    assert neg_l["demand"]["mte3@r0:s0"] == BT * BT * FP32
    assert leaves["demand"]["mte3@r0:s0"] == 2 * 32 * 32 * FP32


@pytest.mark.parametrize("vdim", (128, 256))
def test_stage7_u_destination_and_deferred_w_follow_actual_ub_capacity(vdim):
    # Source Stage7_AicOne retains W plus U128, but U256 goes directly to GM.
    # DrainPackWuToGm runs after the following pack's local S3 heads; only
    # the final pack is drained by the explicit terminal S7 handshake.
    value = dict(native_flags(), lengths=[81], heads=4, key_heads=4,
                 dk=K, dv=vdim, chunk=BT, dtype="bf16")
    engine, _ = execute(value, control_case(units=1), trace_limit=10000)

    def event(suffix):
        return next(row for row in engine.trace if row["task"].endswith(suffix))

    for pack, rows in enumerate((64, 17)):
        for head in range(4):
            drain = event(
                f":tile{pack + 1}:drain_{'prev' if pack == 0 else 'last'}"
                f"{head}:program:store")
            retained_columns = K + (vdim if vdim == 128 else 0)
            assert drain["demand"]["memory@r0:s0"] == rows * retained_columns * BF16
            w = event(f":tile{pack}:s7_{head}:program:part0:publish")
            assert w["demand"]["fixpipe@r0:s0"] == rows * K * BF16
            assert w["demand"].get("memory@r0:s0", 0) == 0
            if vdim == 256:
                direct_u = [
                    event(f":tile{pack}:s7_{head}:program:part{part}:publish")
                    for part in (1, 2)
                ]
                for half in direct_u:
                    assert half["demand"]["memory@r0:s0"] == rows * 128 * BF16
                    assert half["demand"]["fixpipe@r0:s0"] == rows * 128 * BF16
                    assert half["end_ms"] <= event(f":tile{pack}:cube_done")["end_ms"]
                reused = event(f":tile{pack}:s7_{head}:program:part2:l0")
                assert reused["start_ms"] >= direct_u[0]["end_ms"]
            else:
                u = event(f":tile{pack}:s7_{head}:program:part1:publish")
                assert u["demand"]["fixpipe@r0:s0"] == rows * vdim * BF16
                assert u["demand"].get("memory@r0:s0", 0) == 0
            if pack == 0:
                for local_head in range(head % 2, 4, 2):
                    s3 = phase_event(engine, 1, f"s3leaf_{local_head}", "store")
                    assert drain["start_ms"] >= s3["end_ms"]
                s6 = event(f":tile1:s6_{head % 2}:program:load")
                assert s6["start_ms"] >= drain["end_ms"]
            else:
                assert drain["start_ms"] >= event(":tile1:cube_done")["end_ms"]


def test_stage5_second_operands_overlap_first_mmad_before_accumulating():
    # Stage5_Compute uses separate l0Af/l0Bf and l0AfP/l0BfP operand banks,
    # while its two FP32 MMADs accumulate in one L0C tile.
    value = dict(native_flags(), lengths=[64], heads=1, key_heads=1,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    case = control_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["fp32"] = 1e7
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e30, read=1e30, write=1e30, mte1=1e6)
    engine, _ = execute(value, case, trace_limit=10000)

    def event(suffix):
        return next(row for row in engine.trace if row["task"].endswith(suffix))

    first_loaded = event(":s5mm_0:program:left:l0")
    first_mmad = event(":s5mm_0:program:left:compute")
    second_loaded = event(":s5mm_0:program:right:l0")
    second_mmad = event(":s5mm_0:program:right:compute")
    assert second_loaded["start_ms"] >= first_loaded["end_ms"]
    assert max(second_loaded["start_ms"], first_mmad["start_ms"]) \
        < min(second_loaded["end_ms"], first_mmad["end_ms"])
    assert second_mmad["start_ms"] >= first_mmad["end_ms"]


def test_execution_window_is_constant_when_sequence_length_quadruples():
    base = dict(native_flags(), heads=12, key_heads=4,
                dk=K, dv=128, chunk=BT, dtype="bf16")
    summaries = [
        execute({**base, "lengths": [length]}, control_case(units=4))[0].summary()
        for length in (256, 1024)
    ]
    short, long = summaries
    assert long["peak_live_tasks"] == short["peak_live_tasks"]
    assert long["completed_tasks"] > 3 * short["completed_tasks"]
    assert long["live_tasks"] == short["live_tasks"] == 0


def test_each_aiv_ub_and_each_aic_buffer_is_resident_in_its_own_pool_unit():
    sizes = {"ub": 236 * 1024, "l1": 496 * 1024,
             "l0a": 64 * 1024, "l0b": 64 * 1024, "l0c": 256 * 1024}
    value = dict(native_flags(), lengths=[128], heads=12, key_heads=4,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    engine, buffered_time = execute(value, control_case(units=4, buffers=sizes))
    _, no_pool_time = execute(value, control_case(units=4))
    assert buffered_time == no_pool_time
    assert engine.summary()["peak_allocated_per_unit"] == {
        f"{pool}@r0:s0:unit{unit}": size
        for pool, size in sizes.items()
        for unit in range(8 if pool == "ub" else 4)
    }
    assert not engine.allocations and not engine.unit_allocated
    assert all(size == 0 for size in engine.allocated.values())


def test_single_pack_keeps_its_own_core_rate_when_idle_cores_are_added():
    value = dict(native_flags(), lengths=[64], heads=1, key_heads=1,
                 dk=K, dv=128, chunk=BT, dtype="bf16")
    durations = []
    for units in (1, 32):
        case = control_case(units=units)
        device = case["machine"]["device"]
        device["compute_peaks"] = {"bf16": units * 1e9, "fp32": units * 0.5e9}
        device["rates"] = {"memory": 1e30, "vector": 2 * units * 1e9,
                           "mte1": units * 1e30, "fixpipe": units * 1e30}
        device["rates"].update({role: 2 * units * 1e9 for role in REGISTER_CAPACITIES})
        device["service_modes"]["ub_copy"]["rate"] = 2 * units * 1e30
        device["service_modes"]["reg_issue_slots"]["rate"] = 2 * units * 1e9
        durations.append(execute(value, case)[1])
    # Every launched AIV runs Stage0, but idle AICs cannot accelerate the
    # one owner pack; total setup work grows while per-core duration stays put.
    assert durations[1] == pytest.approx(durations[0], rel=1e-12)
