"""Independent oracles for the chunked GDN H-state operator.

The work/address oracle enumerates every ``(sequence, value head, chunk)``
task and derives bytes and FLOPs from explicit tensor intervals. It shares no
closed form with ``gdn_state.build``. The numerical oracle evaluates the chunk
algebra with a per-element outer-product accumulation.
"""
from dataclasses import replace

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import ExecutionError
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import gdn_state
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

K = 128
V = 128
CHUNK = 64


def params(lengths=(64,), heads=48, key_heads=16, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": key_heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
    }
    value.update(overrides)
    return value


def hardware_case(*, units=32, split_copy=False, buffers=None):
    rates = {"memory": 1e12, "vector": 1e12, "mte1": 1e12, "fixpipe": 1e12}
    if split_copy:
        rates.update(read=1e12, write=1e12)
    return {
        "machine": {
            "name": "oracle-control", "num_devices": 1,
            "device": {
                "memory_bytes": 64 * 1024**3, "compute_units": units,
                "compute_peaks": {"bf16": 1e14, "fp32": 2.5e13},
                "rates": rates, "resource_units": {"vector": 2 * units},
                **({"buffers": dict(buffers)} if buffers else {}),
                "buffer_units": {pool: 2 * units if pool == "ub" else units
                                 for pool in (buffers or {})},
            },
        },
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


# ---------------------------------------------------------------------------
# Independent ownership and per-(head, chunk) work/address oracle.
# ---------------------------------------------------------------------------

def brute_force_owner(lengths, heads, cores):
    """Assign each flattened head task to a core with the contiguous ceiling."""
    total = len(lengths) * heads
    per_core = (total + cores - 1) // cores
    return [min(index // per_core, cores - 1) for index in range(total)]


def oracle_units(lengths, heads, cores):
    """Reconstruct the kernel's serial units from a brute-force owner list."""
    owner = brute_force_owner(lengths, heads, cores)
    units = []
    for core in range(cores):
        tasks = [index for index, assigned in enumerate(owner) if assigned == core]
        cursor = 0
        while cursor < len(tasks):
            flat = tasks[cursor]
            sequence, hv_begin = divmod(flat, heads)
            take = 1
            while cursor + take < len(tasks) and take < 4:
                nxt = tasks[cursor + take]
                nxt_sequence, nxt_hv = divmod(nxt, heads)
                if nxt_sequence != sequence or nxt_hv != hv_begin + take:
                    break
                take += 1
            units.append((core, sequence, hv_begin, tuple(
                hv_begin + offset for offset in range(take))))
            cursor += take
    return units


def oracle_work(value, cores):
    """Enumerate every head/chunk task and derive the physical ledger."""
    lengths = tuple(int(item) for item in value["lengths"])
    heads, key_heads = int(value["heads"]), int(value["key_heads"])
    use_initial = bool(value.get("use_initial_state", True))
    final = bool(value.get("output_final_state", True))
    gate_bf16 = str(value.get("gate_dtype", "bf16")) in {"bf16", "bfloat16"}
    exp2 = bool(value.get("use_exp2", False))
    group = heads // key_heads
    gate_bytes = 2 if gate_bf16 else 4
    chunks_per_sequence = [(length + CHUNK - 1) // CHUNK for length in lengths]
    total_chunks = sum(chunks_per_sequence)
    chunk_prefix = [sum(chunks_per_sequence[:sequence])
                    for sequence in range(len(lengths))]
    token_prefix = [sum(lengths[:sequence]) for sequence in range(len(lengths))]

    matrix, mte1, fixpipe, mte2, mte3, vector = 0, 0, 0, 0, 0, 0
    local_mte2 = 0
    w_ranges, u_ranges, vnew_ranges = [], [], []
    h_reads, h_writes = [], []
    right_writes, right_reads = [], []
    initial_ranges, final_ranges = [], []

    for core, sequence, hv_begin, hvs in oracle_units(lengths, heads, cores):
        length = lengths[sequence]
        chunks = chunks_per_sequence[sequence]
        kh_count = len({hv // group for hv in hvs})
        for chunk in range(chunks):
            valid = min(CHUNK, length - chunk * CHUNK)
            rows = (valid + 15) // 16 * 16
            first, last = chunk == 0, chunk == chunks - 1
            stage0 = use_initial or not first
            stage2 = final or not last
            token_begin = token_prefix[sequence] + chunk * CHUNK
            for hv in hvs:
                h_base = (hv * total_chunks + chunk_prefix[sequence] + chunk) * K * V
                if first and use_initial:
                    mte2 += K * V * 4
                    mte3 += K * V * 2
                    vector += K * V
                    initial_ranges.append((sequence, hv, "h0"))
                    h_writes.append((h_base, K * V))
                if stage0:
                    mte2 += 2 * valid * K + 2 * K * V
                    if valid < CHUNK:
                        local_mte2 += CHUNK * K * 2
                    matrix += 2 * rows * K * V
                    # ComputeStage0Head loads padded W and full H into L0,
                    # then publishes only valid P rows as FP32 to its AIV.
                    for tile_rows, tile_columns in ((rows, K), (K, V)):
                        mte1 += tile_rows * tile_columns * 2
                    fixpipe += valid * V * 4
                    w_ranges.append(((hv * sum(lengths) + token_begin) * K, valid * K))
                    h_reads.append((h_base, K * V))
                # Stage1: U read, V_new write, delta and optional right multiply.
                mte2 += 2 * valid * V
                mte3 += 2 * valid * V
                vector += 3 * valid * V
                u_ranges.append(((hv * sum(lengths) + token_begin) * V, valid * V))
                vnew_ranges.append(((hv * sum(lengths) + token_begin) * V, valid * V))
                if stage2:
                    mte2 += gate_bytes * valid
                    mte3 += 2 * valid * V
                    vector += valid * V
                    per_element = 2 + (1 if gate_bf16 else 0) + (1 if exp2 else 0)
                    vector += valid * per_element + per_element
                    slot = hv - hv_begin
                    right_base = (core * 4 + slot) * CHUNK * V
                    right_writes.append((right_base, valid * V))
                    right_reads.append((right_base, valid * V))
                    # Both physical layouts load two BF16 operands; swapping
                    # their roles changes addresses, not transferred bytes.
                    for tile_rows, tile_columns in ((K, rows), (rows, V)):
                        mte1 += tile_rows * tile_columns * 2
                    fixpipe += K * V * 4
                    if valid < CHUNK:
                        local_mte2 += CHUNK * V * 2
                if first and not use_initial:
                    mte3 += K * V * 2
                    vector += K * V + valid * V
                    h_writes.append((h_base, K * V))
            if stage2:
                mte2 += kh_count * 2 * valid * K + len(hvs) * 2 * valid * V
                if valid < CHUNK:
                    local_mte2 += kh_count * CHUNK * K * 2
                matrix += len(hvs) * 2 * K * rows * V
                if first and use_initial:
                    mte2 += len(hvs) * K * V * 4
                    for hv in hvs:
                        initial_ranges.append((sequence, hv, "resident"))
                if not last:
                    mte3 += len(hvs) * K * V * 2
                    vector += len(hvs) * 3 * K * V
                    if first and not use_initial:
                        vector += len(hvs) * K * V
                    for hv in hvs:
                        h_writes.append(((
                            hv * total_chunks + chunk_prefix[sequence] + chunk + 1
                        ) * K * V, K * V))
                elif final:
                    mte3 += len(hvs) * K * V * 4
                    vector += len(hvs) * 2 * K * V
                    if first and not use_initial:
                        vector += len(hvs) * K * V
                    for hv in hvs:
                        final_ranges.append((sequence, hv))
    return {
        "matrix": matrix, "mte1": mte1, "fixpipe": fixpipe,
        "mte2": mte2 + local_mte2, "mte3": mte3, "hbm": mte2 + mte3,
        "local_mte2": local_mte2,
        "vector": vector, "w_ranges": w_ranges, "u_ranges": u_ranges,
        "vnew_ranges": vnew_ranges, "h_reads": h_reads, "h_writes": h_writes,
        "right_writes": right_writes, "right_reads": right_reads,
        "initial_ranges": initial_ranges, "final_ranges": final_ranges,
    }


def plan_ledger(value, cores=32):
    op = gdn_state.make_op(value)
    return gdn_state.build(op, Geometry(cores, {"vector": 2 * cores}, {})), op


def assert_work_matches(plan, oracle):
    expected_matrix = ({"bf16": pytest.approx(oracle["matrix"])}
                       if oracle["matrix"] else {})
    assert plan.work.matrix == expected_matrix
    assert plan.work.resources["mte2"] == pytest.approx(oracle["mte2"])
    assert plan.work.resources["mte3"] == pytest.approx(oracle["mte3"])
    assert plan.work.resources["hbm"] == pytest.approx(oracle["hbm"])
    assert plan.work.resources["mte2"] + plan.work.resources["mte3"] \
        - oracle["local_mte2"] == pytest.approx(plan.work.resources["hbm"])
    assert plan.work.resources["vector"] == pytest.approx(oracle["vector"])
    for resource in ("mte1", "fixpipe"):
        assert plan.work.resources.get(resource, 0) == pytest.approx(oracle[resource])


# ---------------------------------------------------------------------------
# Contiguous ownership and the 48-head / 32-core counterexample.
# ---------------------------------------------------------------------------

def test_contiguous_ceil_ownership_and_24_active_groups():
    groups = gdn_state.head_ownership([64], 48, 32)
    assert len(groups) == 24
    assert [(g.core, g.begin, g.end, len(g.units)) for g in groups] == [
        (core, 2 * core, 2 * core + 2, 1) for core in range(24)]
    assert [unit.heads for group in groups for unit in group.units] == [2] * 24
    # Cores 24..31 own no head task and never appear in the active plan.
    assert [group.core for group in groups] == list(range(24))

    for lengths, heads, cores in (
        ([63, 1, 65], 32, 32),
        ([64], 48, 16),
        ([64], 48, 48),
        ([7], 5, 3),
        ([7], 7, 4),
        ([3, 4, 5], 9, 7),
    ):
        groups = gdn_state.head_ownership(lengths, heads, cores)
        assert [(g.core, u.sequence, u.hv_begin,
                 tuple(range(u.hv_begin, u.hv_begin + u.heads)))
                for g in groups for u in g.units] == oracle_units(lengths, heads, cores)
        covered = [u.sequence * heads + hv for g in groups for u in g.units
                   for hv in range(u.hv_begin, u.hv_begin + u.heads)]
        assert covered == list(range(len(lengths) * heads))
        for group in groups:
            assert group.end - group.begin == sum(u.heads for u in group.units)
            assert all(u.heads <= 4 for u in group.units)
            # A unit may cross a sequence boundary only between units, never
            # inside one unit.
            assert all(u.sequence >= group.units[0].sequence for u in group.units)

    assert len(gdn_state.head_ownership([64], 48, 48)) == 48
    assert len(gdn_state.head_ownership([64], 48, 16)) == 16
    assert all(g.units[0].heads == 3 for g in gdn_state.head_ownership([64], 48, 16))


def test_ownership_respects_sequence_boundaries_for_uneven_lengths():
    groups = gdn_state.head_ownership([63, 1, 65], 32, 32)
    assert len(groups) == 32
    units = [unit for group in groups for unit in group.units]
    assert any(unit.heads == 1 for unit in units), \
        "a sequence-boundary cut must create a one-head unit"
    assert sum(unit.heads for unit in units) == 96
    for unit in units:
        assert 0 <= unit.sequence < 3
        assert unit.hv_begin + unit.heads <= 32
        assert unit.hv_begin // 32 == (unit.hv_begin + unit.heads - 1) // 32


def test_core_count_changes_only_kg_duplication_not_matrix_or_math():
    value = params()
    work16 = plan_ledger(value, 16)[0].work
    work32 = plan_ledger(value, 32)[0].work
    work48 = plan_ledger(value, 48)[0].work
    assert len(gdn_state.head_ownership([64], 48, 16)) == 16
    assert len(gdn_state.head_ownership([64], 48, 32)) == 24
    assert len(gdn_state.head_ownership([64], 48, 48)) == 48
    assert work16.matrix == work32.matrix == work48.matrix == {"bf16": 48 * 4_194_304}
    assert (work16.resources["vector"] == work32.resources["vector"]
            == work48.resources["vector"])
    assert (work16.resources["mte3"] == work32.resources["mte3"]
            == work48.resources["mte3"])
    # One chunk, so each unit loads one kg copy per distinct key head. A finer
    # contiguous split duplicates the shared key-head load across units.
    kg_bytes = 2 * K * 64
    assert work32.resources["mte2"] - work16.resources["mte2"] == 16 * kg_bytes
    assert work48.resources["mte2"] - work32.resources["mte2"] == 16 * kg_bytes


# ---------------------------------------------------------------------------
# Ledger and address coverage.
# ---------------------------------------------------------------------------

CASES = [
    ((64,), 48, 16),
    ((65,), 48, 16),
    ((128,), 48, 16),
    ((63, 1, 65), 32, 8),
    ((193,), 48, 16),
]


@pytest.mark.parametrize("lengths,heads,key_heads", CASES)
def test_enumerated_head_chunk_ledger_matches_plan(lengths, heads, key_heads):
    value = params(lengths=lengths, heads=heads, key_heads=key_heads)
    plan, _ = plan_ledger(value, 32)
    assert_work_matches(plan, oracle_work(value, 32))


@pytest.mark.parametrize("value,cores", [
    (params(lengths=(1,), heads=1, key_heads=1), 1),
    (params(lengths=(129,), heads=7, key_heads=7), 4),
    (params(lengths=(63, 1, 65), heads=32, key_heads=8), 8),
    (params(use_initial_state=False), 32),
    (params(output_final_state=False), 32),
    (params(use_initial_state=False, output_final_state=False), 32),
    (params(lengths=(65,), use_initial_state=False, output_final_state=False), 32),
    (params(lengths=(193,), heads=32, key_heads=8, gate_dtype="fp32",
            use_exp2=True), 32),
])
def test_additional_geometries_and_flag_combinations_match_oracle(value, cores):
    plan, op = plan_ledger(value, cores)
    assert_work_matches(plan, oracle_work(value, cores))
    report = verify_operator(hardware_case(units=cores), op, gdn_state.build)
    assert report["status"] == "passed", report


def test_specific_48_head_32_core_counterexample_numbers():
    plan, _ = plan_ledger(params(), 32)
    oracle = oracle_work(params(), 32)
    assert plan.work.matrix == {"bf16": 201_326_592}
    assert plan.work.resources == {
        "hbm": 17_045_504, "mte2": 10_754_048, "mte3": 6_291_456,
        "vector": 3_941_520, "mte1": 3_932_160, "fixpipe": 4_718_592,
    }
    assert oracle["matrix"] == 201_326_592
    # 16 units own one key head and 8 units own two; the duplicated kg read is
    # 8 * 64 * 128 * 2 = 131072 bytes on top of the uniform 16 * 704768 base.
    assert 16 * 704_768 + 8 * 721_152 == plan.work.resources["hbm"]
    # H is written once per (head, chunk) even for a single-chunk sequence.
    assert sum(length for _, length in oracle["h_writes"]) == 48 * K * V
    assert sorted(oracle["h_writes"]) == sorted(oracle["h_reads"])
    assert len(oracle["initial_ranges"]) == 48 * 2
    assert len(oracle["final_ranges"]) == 48


@pytest.mark.parametrize("rows", (1, 17, 63))
def test_tail_l1_zero_fill_uses_mte2_without_inventing_gm_reads(rows):
    # ClearL1 initializes the entire W/right/Kg slot, then PIPE_MTE2 fences
    # it before the valid-row GM copy. Four heads share one Kg slot here.
    value = params(lengths=(rows,), heads=4, key_heads=1, state_v_first=True)
    plan, _ = plan_ledger(value, 1)
    cleared_slots = [(CHUNK, K)] * 4 + [(CHUNK, V)] * 4 + [(CHUNK, K)]
    local_bytes = sum(r * c * 2 for r, c in cleared_slots)
    assert plan.work.resources["mte2"] + plan.work.resources["mte3"] \
        - plan.work.resources["hbm"] == local_bytes
    assert_work_matches(plan, oracle_work(value, 1))


def test_interval_coverage_is_disjoint_and_sparse_for_gm_tensors():
    value = params(lengths=(193,), heads=8, key_heads=4)
    oracle = oracle_work(value, 8)
    # Stage0 reads every input token/head exactly once; V_new is written once.
    assert _disjoint(oracle["w_ranges"]) == 8 * 193 * K
    assert _disjoint(oracle["u_ranges"]) == 8 * 193 * V
    assert _disjoint(oracle["vnew_ranges"]) == 8 * 193 * V
    # H: every (value head, chunk) snapshot is written once and read once.
    assert _disjoint(oracle["h_writes"]) == 8 * 4 * K * V
    assert sorted(oracle["h_writes"]) == sorted(oracle["h_reads"])
    # S-1 and the resident Stage3 initialization both read initial_state.
    assert len(oracle["initial_ranges"]) == 8 * 2
    assert len(oracle["final_ranges"]) == 8
    # Right workspace is per (core, round head) and is reused across chunks:
    # read intervals equal write intervals for every (slot, chunk) task.
    assert sorted(oracle["right_writes"]) == sorted(oracle["right_reads"])
    for begin, length in oracle["right_writes"]:
        assert begin % (CHUNK * V) == 0
        assert 0 < length <= CHUNK * V


def _disjoint(intervals):
    ordered = sorted(intervals)
    total = 0
    previous_end = -1
    for begin, length in ordered:
        if length == 0:
            continue
        assert begin >= previous_end, (begin, previous_end)
        previous_end = begin + length
        total += length
    return total


def test_state_flags_change_only_the_modeled_initial_final_and_chunk_work():
    for length in (64, 65, 193):
        default = oracle_work(params(lengths=(length,)), 32)
        no_initial = oracle_work(
            params(lengths=(length,), use_initial_state=False), 32)
        no_final = oracle_work(
            params(lengths=(length,), output_final_state=False), 32)
        plan_default, _ = plan_ledger(params(lengths=(length,)), 32)
        plan_no_initial, _ = plan_ledger(
            params(lengths=(length,), use_initial_state=False), 32)
        plan_no_final, _ = plan_ledger(
            params(lengths=(length,), output_final_state=False), 32)
        assert plan_default.work.resources["mte2"] > plan_no_initial.work.resources["mte2"]
        assert default["mte3"] > no_final["mte3"]
        assert no_initial["initial_ranges"] == []
        assert no_final["final_ranges"] == []
        assert_work_matches(plan_default, default)
        assert_work_matches(plan_no_initial, no_initial)
        assert_work_matches(plan_no_final, no_final)


def test_gate_width_and_exp2_are_explicit_vector_work():
    base = plan_ledger(params(), 32)[0].work
    exp2 = plan_ledger(params(use_exp2=True), 32)[0].work
    fp32 = plan_ledger(params(gate_dtype="fp32"), 32)[0].work
    fp32_exp2 = plan_ledger(params(gate_dtype="fp32", use_exp2=True), 32)[0].work
    gate_tasks = 48  # 48 heads, one chunk, final state keeps the last right
    gate_tokens = 64
    assert exp2.resources["vector"] - base.resources["vector"] == \
        gate_tasks * (gate_tokens + 1)
    assert base.resources["vector"] - fp32.resources["vector"] == \
        gate_tasks * (gate_tokens + 1)
    assert fp32_exp2.resources["vector"] - fp32.resources["vector"] == \
        gate_tasks * (gate_tokens + 1)
    assert fp32_exp2.resources["vector"] == base.resources["vector"]
    assert fp32.resources["mte2"] - base.resources["mte2"] == 2 * gate_tasks * gate_tokens
    assert fp32.resources["hbm"] - base.resources["hbm"] == 2 * gate_tasks * gate_tokens


# ---------------------------------------------------------------------------
# Public execution path, buffers and identity invariance.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lengths,heads,key_heads", CASES)
def test_public_isolated_path_conserves_declared_work(lengths, heads, key_heads):
    value = params(lengths=lengths, heads=heads, key_heads=key_heads)
    op = gdn_state.make_op(value)
    plan = gdn_state.build(op, Geometry(32, {"vector": 64}, {}))
    for split in (False, True):
        report = verify_operator(hardware_case(split_copy=split), op, gdn_state.build)
        assert report["status"] == "passed", report
        assert report["finite_completion"]
        assert report["buffers_released"]
        assert report["work_mismatches"] == {}
        row = report["result"]
        assert row["duration_ms"] > 0
        assert row["operator"]["matrix_flops"] == {"bf16": plan.work.matrix["bf16"]}
        assert row["operator"]["hbm_bytes"] == plan.work.resources["hbm"]
        assert row["operator"]["mte2_bytes"] == plan.work.resources["mte2"]
        assert row["operator"]["mte3_bytes"] == plan.work.resources["mte3"]
        assert row["operator"]["vector_ops"] == plan.work.resources["vector"]
        assert row["serviced"]["matrix@r0:s0"] == pytest.approx(
            1000 * plan.work.matrix["bf16"] / 1e14, rel=1e-12)
        assert row["serviced"]["memory@r0:s0"] == pytest.approx(
            plan.work.resources["hbm"], rel=1e-12)
        if split:
            assert row["serviced"]["mte2@r0:s0"] == pytest.approx(
                plan.work.resources["mte2"], rel=1e-12)
            assert row["serviced"]["mte3@r0:s0"] == pytest.approx(
                plan.work.resources["mte3"], rel=1e-12)


def test_identity_fields_do_not_change_physical_plan_or_execution():
    value = params(lengths=(65,))
    op = gdn_state.make_op(value)
    plan = gdn_state.build(op, Geometry(32, {"vector": 64}, {}))
    renamed = replace(op, op_id="identity-control", layer=97, requests=("other",))
    renamed_plan = gdn_state.build(renamed, Geometry(32, {"vector": 64}, {}))
    assert renamed_plan.work == plan.work
    first = verify_operator(hardware_case(), op, gdn_state.build)["result"]
    second = verify_operator(hardware_case(), renamed, gdn_state.build)["result"]
    assert first["duration_ms"] == second["duration_ms"]
    assert first["serviced"] == second["serviced"]


def test_persistent_state_buffers_are_allocated_once_and_released():
    value = params(lengths=(65,), state_v_first=True, use_exp2=True, gate_dtype="fp32")
    op = gdn_state.make_op(value)
    sizes = {"ub": 226 * 1024, "l1": 320 * 1024,
             "l0a": 32 * 1024, "l0b": 64 * 1024, "l0c": 128 * 1024}
    case = hardware_case(buffers=sizes)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(gdn_state.build)})
    row = operator_costs(
        case, (op,), recipes=recipes,
    )["operators"][0]
    assert row["peak_buffers_per_unit"] == {
        f"{pool}@r0:s0:unit{unit}": size
        for pool, size in sizes.items()
        for unit in range(48 if pool == "ub" else 24)
    }
    unbuffered = operator_costs(hardware_case(), (op,), recipes=recipes)["operators"][0]
    assert row["duration_ms"] == unbuffered["duration_ms"]
    assert row["peak_buffers"] == {f"{pool}@r0:s0": size for pool, size in sizes.items()}
    assert not any(row["remaining_buffers"].values())
    # A too-small UB pool must fail visibly instead of silently dropping the
    # persistent state allocation.
    small = hardware_case(buffers={"ub": sizes["ub"] - 1})
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        operator_costs(
            small, (op,),
            recipes=RecipeBook(recipes={op.recipe: PlannedRecipe(gdn_state.build)}),
        )


def execute(value, case, trace_limit=0):
    op = gdn_state.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(gdn_state.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes, trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    assert result.completion.time > 0
    return execution.engine


def phase_event(engine, chunk, name, phase):
    suffixes = (f":tile{chunk}:{name}:program:{phase}",
                f":tile{chunk}:{name}:{phase}")
    return next(row for row in engine.trace if row["task"].endswith(suffixes))


def test_live_state_stays_constant_while_completed_chunk_work_grows():
    base = params(heads=1, key_heads=1, state_v_first=True, gate_dtype="fp32", use_exp2=True)
    short, long = [
        execute({**base, "lengths": [length]}, hardware_case(units=1)).summary()
        for length in (64 * 8, 64 * 32)
    ]
    assert short["live_tasks"] == long["live_tasks"] == 0
    assert short["peak_live_tasks"] == long["peak_live_tasks"]
    assert long["completed_tasks"] > 3 * short["completed_tasks"]


def test_one_head_recurrence_cannot_borrow_idle_aics_or_aivs():
    durations = []
    for units in (1, 32):
        case = hardware_case(units=units)
        device = case["machine"]["device"]
        device["compute_peaks"]["bf16"] = units * 1e9
        device["rates"] = {"memory": 1e30, "vector": 2 * units * 1e9,
                           "mte1": units * 1e30, "fixpipe": units * 1e30}
        engine = execute(params(
            lengths=(128,), heads=1, key_heads=1, state_v_first=True,
            gate_dtype="fp32", use_exp2=True), case)
        durations.append(engine.now)
    assert durations[1] == pytest.approx(durations[0], rel=1e-12)


def test_four_head_unit_publishes_right_only_after_both_local_heads_finish():
    # arch35 vec.h RunStage1 queues both RIGHT_READY flags after its two
    # local-head MTE3 stores. Right shares the high half of the P/D data slot.
    case = hardware_case(units=1)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e15
    case["machine"]["device"]["rates"].update(memory=1e30, vector=1e6)
    engine = execute(params(
        heads=4, key_heads=1, state_v_first=True, gate_dtype="fp32", use_exp2=True),
        case, trace_limit=10000)
    for aiv in (0, 1):
        second_delta = phase_event(engine, 0, f"delta{aiv + 2}", "store")
        first_right = phase_event(engine, 0, f"right{aiv}", "load")
        assert first_right["start_ms"] >= second_delta["end_ms"]


@pytest.mark.parametrize("heads", (1, 4))
def test_initial_state_banks_are_drained_before_main_loop_reuses_ub(heads):
    # RunSMinusOne drains every local bank before H_READY and before entering
    # ProcessSingleHeadPipeline/RunStage1, whose U/g banks alias the same UB.
    case = hardware_case(units=1, split_copy=True)
    case["machine"]["device"]["rates"].update({
        "memory": 1e30, "vector": 1e9, "read": 1e6, "write": 1e6,
    })
    engine = execute(params(
        heads=heads, key_heads=1, state_v_first=True,
        gate_dtype="fp32", use_exp2=True), case, trace_limit=10000)
    for aiv in range(min(heads, 2)):
        initialized = max(
            phase_event(engine, 0, f"pre{head}", "store")["end_ms"]
            for head in range(aiv, heads, 2))
        for head in range(aiv, heads, 2):
            assert phase_event(engine, 0, f"input{head}", "load")["start_ms"] >= initialized
            assert phase_event(engine, 0, f"h{head}", "load")["start_ms"] >= initialized


def test_p_slot_release_follows_vector_consumption_not_right_gm_writeback():
    # P occupies the lower 32 KiB; right occupies the upper half of the data
    # slot. PIPE_V releases P before MTE3 drains V_new and right.
    case = hardware_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e15
    case["machine"]["device"]["rates"].update({
        "memory": 1e30, "vector": 1e8, "read": 1e30, "write": 1e5,
    })
    engine = execute(params(
        heads=4, key_heads=1, state_v_first=True,
        gate_dtype="fp32", use_exp2=True), case, trace_limit=10000)
    for aiv in (0, 1):
        consumed = phase_event(engine, 0, f"delta{aiv}", "compute")
        drain = phase_event(engine, 0, f"delta{aiv}", "store")
        produced = phase_event(engine, 0, f"p{aiv + 2}", "compute")
        published = next(
            row for row in engine.trace
            if row["task"].endswith(f":tile0:p_publish{aiv + 2}"))
        # Separate L0C banks let both results finish while their previous P
        # consumers are active; each publication still waits its own P_FREE.
        assert produced["end_ms"] < consumed["end_ms"]
        assert published["start_ms"] >= consumed["end_ms"]
        assert published["end_ms"] < drain["end_ms"]


def test_d_slot_release_waits_h_alias_store_but_not_independent_final_state():
    # Nonfinal H is written from the D slot itself, so D_FREE is on MTE3.
    # Final state is written from a separate resident FP32 bank, so D_FREE is
    # on PIPE_V and the following head's D can arrive during that GM write.
    case = hardware_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e15
    case["machine"]["device"]["rates"].update({
        "memory": 1e30, "vector": 1e8, "read": 1e30, "write": 1e5,
    })
    engine = execute(params(
        lengths=(128,), heads=4, key_heads=1, state_v_first=True,
        gate_dtype="fp32", use_exp2=True), case, trace_limit=10000)
    for chunk in (0, 1):
        for aiv in (0, 1):
            consumed = phase_event(engine, chunk, f"state{aiv}", "compute")
            drain = phase_event(engine, chunk, f"state{aiv}", "store")
            published = next(
                row for row in engine.trace
                if row["task"].endswith(f":tile{chunk}:d_publish{aiv + 2}"))
            assert published["start_ms"] >= consumed["end_ms"]
            if chunk == 0:
                assert published["start_ms"] >= drain["end_ms"]
            else:
                assert published["end_ms"] < drain["end_ms"]


def test_single_head_lookahead_orders_w_k_copies_and_overlaps_current_d():
    # Single-head cube source issues W then K on MTE2, both for startup and
    # lookahead. Issuing the next pair after right_N must not delay MMAD D_N
    # until that lookahead pair finishes.
    case = hardware_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e7
    case["machine"]["device"]["rates"].update({
        "memory": 1e30, "vector": 1e8, "read": 1e6, "write": 1e8,
    })
    engine = execute(params(
        lengths=(192,), heads=1, key_heads=1, state_v_first=True,
        gate_dtype="fp32", use_exp2=True), case, trace_limit=10000)
    for chunk in range(3):
        w = phase_event(engine, chunk, "w0", "load")
        k = phase_event(engine, chunk, "k0", "load")
        assert k["start_ms"] >= w["end_ms"]
        if chunk:
            right = phase_event(engine, chunk - 1, "right0", "load")
            d = phase_event(engine, chunk - 1, "d0", "compute")
            assert w["start_ms"] >= right["end_ms"]
            assert max(w["start_ms"], d["start_ms"]) < min(k["end_ms"], d["end_ms"])
            previous_input = phase_event(engine, chunk - 1, "input0", "load")
            lookahead_input = phase_event(engine, chunk, "input0", "load")
            previous_p = phase_event(engine, chunk - 1, "p0", "compute")
            assert lookahead_input["start_ms"] >= previous_input["end_ms"]
            assert lookahead_input["end_ms"] < previous_p["end_ms"]


def test_ping_pong_l0_inputs_wait_for_same_bank_mmad_and_allow_other_bank_load():
    # Both P and D use two physical L0A/L0B banks. M_MTE1 releases a bank
    # after its MMAD; the other bank can be loaded while that MMAD runs.
    case = hardware_case(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e7
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e30, read=1e30, write=1e30)
    engine = execute(params(
        heads=4, key_heads=1, state_v_first=True,
        gate_dtype="fp32", use_exp2=True), case, trace_limit=10000)
    for stage in ("p", "d"):
        first = phase_event(engine, 0, f"{stage}0", "compute")
        lookahead = phase_event(engine, 0, f"{stage}1", "l0")
        assert lookahead["start_ms"] < first["end_ms"]
        for head in (2, 3):
            consumed = phase_event(engine, 0, f"{stage}{head - 2}", "compute")
            reused = phase_event(engine, 0, f"{stage}{head}", "l0")
            assert reused["start_ms"] >= consumed["end_ms"]


# ---------------------------------------------------------------------------
# Numerical algebra oracle.
# ---------------------------------------------------------------------------

def test_chunk_map_matches_per_element_outer_product_accumulation():
    rng = np.random.default_rng(20261028)
    chunk, k, v = 5, 7, 9
    state = rng.standard_normal((k, v))
    w = rng.standard_normal((chunk, k))
    u = rng.standard_normal((chunk, v))
    key = rng.standard_normal((chunk, k))
    gate = -0.1 - 0.2 * rng.random(chunk)

    # Matrix form used by the kernel: P=W@H; delta=U-P; right=decay*delta;
    # D=K^T@right; H_next=exp(g_last)*H+D.
    p = w @ state
    delta = u - p
    decay = np.exp(gate[-1] - gate)
    right = delta * decay[:, None]
    d_fast = key.T @ right
    next_fast = np.exp(gate[-1]) * state + d_fast

    # Independent per-element accumulation: no matrix products in the state
    # update, and the P product is also written as an explicit inner sum.
    next_slow = np.exp(gate[-1]) * state.copy()
    for row in range(chunk):
        z = float(np.exp(gate[-1] - gate[row]))
        for kk in range(k):
            for vv in range(v):
                p_slow = sum(float(w[row, j]) * float(state[j, vv]) for j in range(k))
                next_slow[kk, vv] += z * float(key[row, kk]) * (
                    float(u[row, vv]) - p_slow)
    assert np.allclose(next_fast, next_slow, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("value_first", (False, True))
def test_physical_state_addresses_and_update_operands_preserve_asymmetric_state(value_first):
    # Source FwdHStateOffset uses [sequence,head,V,K] in the captured branch.
    # Square axes alone cannot distinguish it from [K,V]; values are asymmetric.
    sequence, head, heads = 1, 2, 3
    logical = np.arange(K * V, dtype=np.float64).reshape(K, V) / 4096
    physical = np.ascontiguousarray(logical.T if value_first else logical)
    offsets = np.array([
        [gdn_state.state_offset(sequence, head, key, value, heads, value_first)
         for value in range(V)] for key in range(K)
    ])
    base = (sequence * heads + head) * K * V
    assert sorted((offsets - base).ravel()) == list(range(K * V))
    decoded = physical.ravel()[offsets - base]
    np.testing.assert_array_equal(decoded, logical)
    assert offsets[0, 1] != offsets[1, 0]
    if value_first:
        assert offsets[0, 1] - base == K
        assert offsets[1, 0] - base == 1

    rng = np.random.default_rng(710)
    rows = 7
    w = rng.standard_normal((rows, K)) / 128
    u = rng.standard_normal((rows, V))
    key = rng.standard_normal((rows, K)) / 128
    gates = -np.arange(1, rows + 1) / 16
    op = gdn_state.make_op(params(
        lengths=(rows,), state_v_first=value_first, gate_dtype="fp32", use_exp2=True))
    meta = dict(op.metadata)
    key_stride, value_stride = meta["h_semantic_strides"]
    # Column-major H in ComputeStage0Head must interpret the physical [V,K].
    indexed = physical.ravel()[np.arange(K)[:, None] * key_stride
                               + np.arange(V)[None, :] * value_stride]
    p = w @ indexed
    np.testing.assert_allclose(p, w @ logical)
    assert not np.allclose(p, w @ logical.T)
    right = (u - p) * np.exp2(gates[-1] - gates)[:, None]
    operands = {"right_transposed": right.T, "kg": key,
                "kg_transposed": key.T, "right": right}
    left, right_name = meta["update_operands"]
    update = operands[left] @ operands[right_name]
    next_physical = np.exp2(gates[-1]) * physical + update
    expected = np.exp2(gates[-1]) * logical
    for row in range(rows):
        expected += np.outer(key[row], (u[row] - p[row])) * np.exp2(gates[-1] - gates[row])
    np.testing.assert_allclose(next_physical.T if value_first else next_physical,
                               expected, rtol=1e-12, atol=1e-12)
    plan, _ = plan_ledger(params(
        lengths=(rows,), state_v_first=value_first, gate_dtype="fp32", use_exp2=True))
    assert_work_matches(plan, oracle_work(params(
        lengths=(rows,), state_v_first=value_first, gate_dtype="fp32", use_exp2=True), 32))
