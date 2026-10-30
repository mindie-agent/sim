"""Independent source-expression, address, precision and dependency controls.

The helper replay follows pinned triton_attention_helpers.py:418-440. It records
typed tensor operands, not a guessed compiled instruction sequence. No device
runtime is imported. Synthetic service rates never establish hardware accuracy.
"""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators.attention_online import SERVICE_ROLES, build, make_op
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError

BASE = dict(query=1, context=0, heads=1, key_heads=1, head_dim=16,
            dtype="bf16", causal=True)
CASES = (
    BASE,
    dict(BASE, query=1, context=128, heads=8, key_heads=2, head_dim=64),
    dict(BASE, query=4, context=1024, heads=8, key_heads=2, head_dim=64),
    dict(BASE, query=64, context=64, heads=8, key_heads=2, head_dim=64),
    dict(BASE, query=17, context=129, heads=12, key_heads=3, head_dim=64),
    dict(BASE, query=65, context=257, heads=8, head_dim=64),
    dict(BASE, query=33, context=65, heads=4, head_dim=256,
         block=128, compute_capability=(10, 0)),
)


class Operand:
    def __init__(self, shape, ledger):
        self.shape, self.ledger = tuple(shape), ledger

    def __getitem__(self, index):
        return Operand(np.empty(self.shape)[index].shape, self.ledger)

    def binary(self, other, role):
        shape = np.broadcast_shapes(
            self.shape, other.shape if isinstance(other, Operand) else ())
        self.ledger[role] += int(np.prod(shape))
        return Operand(shape, self.ledger)

    def __sub__(self, other):
        return self.binary(other, "fp32_sub")

    def __mul__(self, other):
        return self.binary(other, "fp32_mul")

    def __add__(self, other):
        return self.binary(other, "fp32_add")

    def __truediv__(self, other):
        return self.binary(other, "fp32_div")

    def __gt__(self, other):
        return self.binary(other, "fp32_compare")

    def __le__(self, other):
        return self.binary(other, "int32_compare")

    def __lt__(self, other):
        return self.binary(other, "int32_compare")

    def __and__(self, other):
        return self.binary(other, "predicate_and")

    def cast(self, dtype):
        self.ledger["cvt_fp32_" + dtype] += int(np.prod(self.shape))
        return Operand(self.shape, self.ledger)


class SourceOperations:
    def maximum(self, lhs, rhs):
        return lhs.binary(rhs, "fp32_max")

    def where(self, condition, lhs, rhs):
        return lhs.binary(rhs, "fp32_select")

    def exp(self, value):
        value.ledger["fp32_exp"] += int(np.prod(value.shape))
        return Operand(value.shape, value.ledger)

    def reduce(self, value, axis, role):
        value.ledger[role] += int(np.prod(value.shape))
        return Operand(tuple(n for i, n in enumerate(value.shape) if i != axis),
                       value.ledger)

    def max(self, value, axis):
        return self.reduce(value, axis, "source_tl_reduce_max_elements")

    def sum(self, value, axis):
        return self.reduce(value, axis, "source_tl_reduce_sum_elements")


def source_softmax(tl, S, M, L):
    # Verbatim operations from helpers418-440, with tl supplied by the oracle.
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


def source_oracle(p):
    """Enumerate source addresses and replay expressions; no closed-form sums."""
    q, c, h, k, d = (p[name] for name in
                     ("query", "context", "heads", "key_heads", "head_dim"))
    ratio = h // k
    cc = p.get("compute_capability", (8, 9))
    tuned = cc[0] == 10 and d == 256 and q > 1 and ratio <= 16
    bm = 32 if tuned else (16 if ratio <= 16 else 1 << (ratio - 1).bit_length())
    bq = bm // ratio
    tile = 128 if tuned else p.get("block", 32)
    dp = 1 << (d - 1).bit_length()
    ledger, memory, matrices = Counter(), Counter(), Counter()
    tl = SourceOperations()
    q_addresses = []
    for head in range(k):
        for block in range(q // bq + 1):
            # find_seq_idx + resolver load expressions, even on early return.
            for address in (("cu", 0), ("cu", 0), ("cu", 0), ("cu", 1), ("len", 0)):
                memory["metadata"] += 4
            if block * bq >= q:
                continue
            rows = [(block * bq + m // ratio, head * ratio + m % ratio)
                    for m in range(bm)]
            for token, query_head in rows:
                for dim in range(d):
                    if token < q and query_head < h:
                        q_addresses.append((token, query_head, dim))
                        memory["q"] += 2
                        memory["output"] += 2
            max_prefix = min(c + block * bq + (bm - 1) // ratio + 1, q + c)
            if not p["causal"]:
                max_prefix = q + c
            M, L, acc = Operand((bm,), ledger), Operand((bm,), ledger), Operand((bm, dp), ledger)
            for first in range(0, max_prefix, tile):
                for key in range(first, first + tile):
                    memory["page"] += 4
                    if key < max_prefix:
                        memory["kv"] += 2 * d * 2
                matrices[p["dtype"]] += 2 * bm * tile * dp
                S = Operand((bm, tile), ledger) * 0.25
                query_pos, seq_offset = Operand((bm, 1), ledger), Operand((tile,), ledger)
                seq_mask = (seq_offset[None, :] <= query_pos if p["causal"]
                            else seq_offset[None, :] < (q + c))
                row_mask = Operand((bm, 1), ledger) & Operand((bm, 1), ledger)
                S = tl.where(row_mask & seq_mask, S, float("-inf"))
                M, L, P, alpha = source_softmax(tl, S, M, L)
                acc = acc * alpha[:, None]
                P.cast(p["dtype"])
                matrices[p["dtype"]] += 2 * bm * tile * dp
            (acc / L[:, None]).cast(p["dtype"])
    reads = sum(memory[name] for name in ("metadata", "q", "page", "kv"))
    ledger.update(hbm=reads + memory["output"], mte2=reads, mte3=memory["output"])
    return dict(ledger), dict(matrices), memory, q_addresses, (bm, bq, tile)


def hardware(*, units=1, memory=1e30, matrix_per_unit=1e30,
             mode_rates=None, mode_resources=None, buffers=None):
    mode_rates, mode_resources = mode_rates or {}, mode_resources or {}
    rates = {"memory": memory, "vector": units * 1e12}
    modes = {}
    for role in SERVICE_ROLES:
        resource = mode_resources.get(role, "vector")
        rates.setdefault(resource, units * 1e12)
        modes[role] = {"resource": resource, "rate": units * mode_rates.get(role, 1e30)}
    device = {
        "memory_bytes": 1 << 30, "compute_units": units,
        "compute_peaks": {dtype: units * matrix_per_unit for dtype in ("bf16", "fp16")},
        "rates": rates, "service_modes": modes,
        "resource_units": {role: units for role in rates},
    }
    if buffers:
        device["buffers"] = buffers
        device["buffer_units"] = dict.fromkeys(buffers, units)
    return {
        "machine": {"name": "synthetic-source-control", "num_devices": 1, "device": device},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 16,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 16,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def run(params, machine=None, *, ready=0, trace_limit=10000):
    op = make_op(params)
    case = parse_case(machine or hardware())
    execution = Execution(case, CostModel(case),
                          recipes=RecipeBook({op.recipe: PlannedRecipe(build)}),
                          trace_limit=trace_limit)
    prepared, plan = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    event = execution.engine.submit("ready", release=ready)
    end = plan.lower(context).start(execution.engine, "attention", event)
    before = {}
    if ready:
        execution.engine.advance(horizon=ready)
        before = {k: v for k, v in execution.engine.serviced.items() if v}
    execution.engine.advance(until=(end,))
    return execution.engine, end, before, prepared, plan


@pytest.mark.parametrize("params", CASES)
def test_source_expression_inventory_and_memory_addresses(params):
    expected, matrices, _, _, _ = source_oracle(params)
    op = make_op(params)
    plan = build(op, Geometry(32))
    assert plan.work.resources == expected
    assert plan.work.matrix == matrices
    assert op.vector_ops == 0
    assert dict(op.metadata)["resource_work"] == {
        role: count for role, count in expected.items() if role not in ("hbm", "mte2", "mte3")}


@pytest.mark.parametrize("ratio", (1, 3, 7, 16, 17))
@pytest.mark.parametrize("causal", (False, True))
def test_nondivisible_gqa_tail_and_capability_family(ratio, causal):
    for cc in ((8, 9), (10, 0), (10, 3), (12, 0)):
        for query in (1, 3, 16, 17):
            p = dict(BASE, query=query, context=17, heads=2 * ratio, key_heads=2,
                     head_dim=256, compute_capability=cc, causal=causal)
            resources, matrices, memory, addresses, shape = source_oracle(p)
            op = make_op(p)
            plan = build(op, Geometry(2))
            metadata = dict(op.metadata)
            assert plan.work.resources == resources
            assert plan.work.matrix == matrices
            assert (metadata["block_m"], metadata["block_q"], metadata["tile_size"]) == shape
            assert metadata["q_read_bytes"] == memory["q"] == len(addresses) * 2
            assert metadata["query_row_slot_overlap"] == len(addresses) // 256 - query * 2 * ratio


def test_source_softmax_running_max_clamp_and_noncausal_broadcast():
    counts = Counter()
    source_softmax(SourceOperations(), Operand((16, 32), counts),
                   Operand((16,), counts), Operand((16,), counts))
    # The pinned source has distinct maximum, comparison and select operations.
    assert counts["fp32_max"] == counts["fp32_compare"] == counts["fp32_select"] == 16
    assert sum(counts.values()) == 2160
    causal = build(make_op(dict(BASE, context=31)), Geometry(1)).work.resources
    full = build(make_op(dict(BASE, context=31, causal=False)), Geometry(1)).work.resources
    assert causal["int32_compare"] == 512
    assert full["int32_compare"] == 32
    assert {k: v for k, v in causal.items() if k != "int32_compare"} == {
        k: v for k, v in full.items() if k != "int32_compare"}


def test_exp_service_rate_changes_duration_without_changing_source_work():
    fast = hardware(mode_rates={"fp32_exp": 1e6})
    slow = hardware(mode_rates={"fp32_exp": 1e3})
    a, end_a, _, op_a, plan_a = run(BASE, fast)
    b, end_b, _, op_b, plan_b = run(BASE, slow)
    assert plan_a.work == plan_b.work
    assert op_a.flops == op_b.flops
    assert end_b.time > 100 * end_a.time
    assert b.serviced["vector@r0:s0"] > 100 * a.serviced["vector@r0:s0"]


def test_missing_typed_rate_does_not_fall_back_to_generic_vector():
    machine = hardware()
    del machine["machine"]["device"]["service_modes"]["fp32_exp"]
    # This is the pricing boundary, not defensive parser coverage.
    with pytest.raises(ModelingError, match="physical_resource_rate_missing:fp32_exp"):
        run(BASE, machine)


def rows_by_suffix(engine, suffix):
    return [row for row in engine.trace if row["task"].endswith(suffix)]


def test_alpha_rescale_pv_are_independent_of_slow_l_sum_but_next_tile_waits():
    p = dict(BASE, context=32)
    machine = hardware(mode_rates={"source_tl_reduce_sum_elements": 1e3},
                       mode_resources={"source_tl_reduce_sum_elements": "reduction"})
    engine, end, _, _, _ = run(p, machine)
    tile0_sum = next(row for row in rows_by_suffix(engine, ":p_sum") if ":tile0:" in row["task"])
    tile0_pv = next(row for row in rows_by_suffix(engine, ":pv") if ":tile0:" in row["task"])
    tile0_l = next(row for row in rows_by_suffix(engine, ":l_update") if ":tile0:" in row["task"])
    tile1_load = next(row for row in rows_by_suffix(engine, ":page_table") if ":tile1:" in row["task"])
    assert tile0_pv["end_ms"] < tile0_sum["end_ms"]
    assert tile0_l["start_ms"] >= tile0_sum["end_ms"]
    assert tile1_load["start_ms"] >= tile0_l["end_ms"]
    assert rows_by_suffix(engine, ":output_divide")[0]["start_ms"] >= rows_by_suffix(
        engine, ":l_update")[-1]["end_ms"]
    assert end.done


def test_output_cast_and_store_wait_for_division_and_delayed_input():
    machine = hardware(mode_rates={"fp32_div": 1e3})
    engine, end, before, _, _ = run(BASE, machine, ready=1.25)
    division = rows_by_suffix(engine, ":output_divide")[0]
    cast = rows_by_suffix(engine, ":output_cast")[0]
    store = rows_by_suffix(engine, ":output")[0]
    assert cast["start_ms"] >= division["end_ms"]
    assert store["start_ms"] >= cast["end_ms"]
    assert before == {}
    assert all(row["start_ms"] >= 1.25 for row in engine.trace if row["serviced"])
    assert end.time >= 1.25 + 256


def test_single_cta_ownership_parallel_ctas_and_shared_hbm():
    durations = []
    for units in (1, 8, 32):
        _, end, _, _, _ = run(BASE, hardware(units=units, matrix_per_unit=1e6))
        durations.append(end.time)
    assert durations == pytest.approx([32.768] * 3)
    for units, wanted in ((1, 65.536), (2, 32.768)):
        engine, end, _, _, _ = run(dict(BASE, heads=2, key_heads=2),
                                  hardware(units=units, matrix_per_unit=1e6))
        assert end.time == pytest.approx(wanted)
        if units == 2:
            qk = rows_by_suffix(engine, ":qk")
            assert {tuple(row["placement"]["matrix@r0:s0"]) for row in qk} == {(0,), (1,)}
            assert max(row["start_ms"] for row in qk) < min(row["end_ms"] for row in qk)
    for units in (1, 8, 32):
        _, end, _, _, _ = run(BASE, hardware(units=units, memory=1e6))
        assert end.time == pytest.approx(0.276)


def test_guarded_metadata_and_masked_payload_have_distinct_boundaries():
    engine, _, _, op, _ = run(dict(BASE, query=16), hardware(units=2))
    loads = rows_by_suffix(engine, ":metadata")
    assert len(loads) == 2
    assert [row["serviced"]["memory@r0:s0"] for row in loads] == [20, 20]
    assert len(rows_by_suffix(engine, ":q_load")) == 1
    assert dict(op.metadata)["early_return_metadata_bytes"] == 20
    p = dict(BASE, context=32, head_dim=17)
    engine, _, _, op, _ = run(p)
    pages = rows_by_suffix(engine, ":page_table")
    k = rows_by_suffix(engine, ":k_load")
    v = rows_by_suffix(engine, ":v_load")
    assert [row["serviced"]["memory@r0:s0"] for row in pages] == [128, 128]
    assert [row["serviced"]["memory@r0:s0"] for row in k] == [1088, 34]
    assert [row["serviced"]["memory@r0:s0"] for row in v] == [1088, 34]
    assert op.flops == 4 * 16 * 32 * 32 * 2


def test_declared_register_state_is_owned_and_bounded_by_active_ctas():
    machine = hardware(units=2, buffers={"register": 1 << 20, "ub": 1, "l1": 1})
    engine, _, _, op, _ = run(dict(BASE, heads=2, key_heads=2), machine)
    peak = engine.summary()["peak_allocated_per_unit"]
    assert peak == {"register@r0:s0:unit0": 1664, "register@r0:s0:unit1": 1664}
    assert not any(engine.summary()["live_allocated"].values())
    short, _, _, _, _ = run(dict(BASE, query=256), machine, trace_limit=0)
    long, _, _, _, _ = run(dict(BASE, query=512), machine, trace_limit=0)
    assert long.summary()["completed_tasks"] > short.summary()["completed_tasks"]
    assert long.summary()["peak_live_tasks"] == short.summary()["peak_live_tasks"]
    assert long.summary()["live_tasks"] == short.summary()["live_tasks"] == 0


def test_undeclared_register_state_never_uses_ub_l1_or_shared_storage():
    plain, baseline, _, _, _ = run(BASE, hardware(matrix_per_unit=1e6))
    other, end, _, _, _ = run(
        BASE, hardware(matrix_per_unit=1e6, buffers={"ub": 1, "l1": 1, "shared_memory": 1}))
    assert end.time == baseline.time
    assert other.summary()["peak_allocated_per_unit"] == {}
    assert plain.summary()["peak_allocated_per_unit"] == {}
    assert not any(other.summary()["live_allocated"].values())


@pytest.mark.parametrize("params", CASES)
def test_isolated_public_contract_identity_delay_and_typed_work(params):
    op = make_op(params)
    machine = hardware(units=4, memory=1e9, matrix_per_unit=1e9,
                       mode_rates=dict.fromkeys(SERVICE_ROLES, 1e9))
    immediate = verify_operator(machine, op, build)
    renamed = replace(op, op_id="different-name", layer=19, requests=("other",))
    delayed = verify_operator(machine, renamed, build, input_ready_ms=2)
    assert immediate["status"] == delayed["status"] == "passed"
    assert immediate["work_mismatches"] == delayed["work_mismatches"] == {}
    assert delayed["input_dependency_obeyed"]
    assert immediate["result"]["duration_ms"] == pytest.approx(delayed["result"]["duration_ms"])


def cast_storage(values, dtype):
    values = np.asarray(values, np.float32)
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1))
            & np.uint32(0xffff0000)).view(np.float32)


@pytest.mark.parametrize("dtype,expected,uncast", (
    ("bf16", 0.11962890625, 0.119140625),
    ("fp16", 0.1192626953125, 0.11920166015625),
))
def test_source_p_conversion_changes_actual_stored_output(dtype, expected, uncast):
    S = np.array([[0, -2]], np.float32)
    _, L, P, _ = source_softmax(np, S, np.array([-np.inf], np.float32),
                                np.ones(1, np.float32))
    V = np.array([[0], [1]], np.float32)
    result = cast_storage(cast_storage(P, dtype) @ V / L[:, None], dtype)
    without_p_cast = cast_storage(P @ V / L[:, None], dtype)
    assert result.item() == expected
    assert without_p_cast.item() == uncast
    assert result.item() != without_p_cast.item()


@pytest.mark.parametrize("dtype", ("bf16", "fp16"))
def test_padded_rows_remain_finite_and_online_recurrence_handles_max_change(dtype):
    # FP32 helper state, actual P storage conversion, three different tile maxima.
    values = cast_storage(np.array([[1, -2], [3, 5], [-4, 2], [2, -1]], np.float32), dtype)
    scores = np.array([[0, -2, 3, -1], [-np.inf] * 4], np.float32)
    M, L = np.full(2, -np.inf, np.float32), np.ones(2, np.float32)
    acc = np.zeros((2, 2), np.float32)
    for first, last in ((0, 2), (2, 3), (3, 4)):
        M, L, P, alpha = source_softmax(np, scores[:, first:last], M, L)
        acc = acc * alpha[:, None] + cast_storage(P, dtype) @ values[first:last]
    assert np.all(np.isfinite(acc))
    assert np.all(np.isfinite(M))
    assert L[1] == 0
    # The padded row is masked at the output store; its 0/0 is not a payload.
    out = cast_storage(acc[0] / L[0], dtype)
    dense_p = np.exp(scores[0] - np.max(scores[0]))
    ideal = dense_p @ values / np.sum(dense_p)
    np.testing.assert_allclose(out, ideal, rtol=0.006, atol=0.004)
