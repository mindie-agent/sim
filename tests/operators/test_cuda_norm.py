"""Independent checks for the source-derived CUDA normalization planner.

The tests never execute CUDA.  They reconstruct the launch rules from the
source text, interpret the pinned CCCL warp/block reduction, execute the two
source kernels on a small CPU model, and compare the independently counted
arithmetic, shared-temporary traffic and global byte streams with the plan and
with the shared engine's actual service.  Structural contract checks are
labelled separately from numerical-oracle checks.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import cuda_norm
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.schema.common import ModelingError

CASE_FILE = Path(__file__).parents[2] / "tests/fixtures/operator_cases/cuda_norm.json"
WARP = 32
STEPS = 5
TEMPLATE_WARPS = 32


def bf16(values):
    """Round float32 to bfloat16 (nearest-even), retaining float32 storage."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).copy()
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def quantize(values, dtype):
    array = np.asarray(values, dtype=np.float32)
    if dtype == "bf16":
        return bf16(array)
    if dtype == "fp16":
        return np.asarray(array, dtype=np.float16).astype(np.float32)
    if dtype == "fp32":
        return array
    raise AssertionError(dtype)


def cub_reduce_counts(block):
    """Independently interpret the pinned CCCL v3.0.0 warp-reduction tree.

    ``block_reduce_warp_reductions.cuh`` splits the template block into
    32-thread logical warps; the last runtime warp passes its remainder as
    ``num_valid``.  ``warp_reduce_shfl.cuh`` iterates five steps and executes
    the add only when the shuffled source lane stays inside the segment.
    After a barrier, only thread 0 combines successor lane-0 aggregates.
    """
    full, partial = divmod(block, WARP)
    warps = full + (1 if partial else 0)
    warp_adds = 0
    for warp in range(warps):
        valid = WARP if warp < full else partial
        for step in range(STEPS):
            warp_adds += max(0, valid - (1 << step))
    cross = sum(1 for warp in range(1, TEMPLATE_WARPS) if warp * WARP < block)
    return {"active_warps": warps, "full_warps": full,
            "partial_warp_lanes": partial, "warp_lane_adds": warp_adds,
            "cross_warp_adds": cross, "warp_shuffle_steps": STEPS * warps}


def independent_dispatch(rows, width, dtype, kernel, has_weight,
                         aligned, batch_invariant, input_stride):
    size = {"bf16": 2, "fp16": 2, "fp32": 4}[dtype]
    max_block = 1024 if rows < 256 else 256
    if kernel == "rms_norm":
        vector = math.gcd(16 // size, width)
        block = min(width // vector, max_block)
        vectorized = vector > 1
        path = ("scalar_width_one" if vector == 1 else
                "unspecified" if not aligned else
                "fully_vectorized_rows"
                if input_stride * size % (vector * size) == 0 else
                "row_prefix_vector_tail")
    else:
        block = min(width, max_block)
        vectorized = (aligned and not batch_invariant and width % 8 == 0
                      and input_stride % 8 == 0)
        vector = 8 if vectorized else 0
        path = "fused_width8" if vectorized else "fused_generic_width0"
    return {"rows": rows, "width": width, "dtype": dtype, "size": size,
            "grid": rows, "block": block, "max_block": max_block,
            "vector_width": vector, "vectorized": vectorized,
            "vector_path": path, "has_weight": has_weight,
            "aligned": aligned, "batch_invariant": batch_invariant,
            "input_stride": input_stride,
            **cub_reduce_counts(block)}


class Counters:
    """Performed CPU work, counted where the source arithmetic executes."""

    def __init__(self):
        self.load_ops = 0
        self.store_ops = 0
        self.warp_ops = 0
        self.cross_ops = 0
        self.normalize_ops = 0
        self.hbm_read = 0
        self.hbm_write = 0
        self.load_hbm = 0
        self.store_hbm = 0
        self.warp_shuffles = 0
        self.shared_writes = 0
        self.shared_reads = 0
        self.barriers = 0

    @property
    def vector(self):
        return (self.load_ops + self.store_ops + self.warp_ops
                + self.cross_ops + self.normalize_ops)

    @property
    def hbm(self):
        return self.hbm_read + self.hbm_write


def _cvt_is_real(dtype):
    return dtype != "fp32"


def _warp_reduce(partials, block, counters):
    """Execute the pinned warp shuffle tree and thread-0 successor chain."""
    info = cub_reduce_counts(block)
    aggregates = []
    for warp in range(info["active_warps"]):
        valid = WARP if warp < info["full_warps"] else info["partial_warp_lanes"]
        lanes = np.zeros(WARP, dtype=np.float32)
        lanes[:valid] = partials[warp * WARP: warp * WARP + valid]
        for step in range(STEPS):
            counters.warp_shuffles += 1
            offset = 1 << step
            updated = lanes.copy()
            for lane in range(WARP):
                if lane + offset < valid:
                    updated[lane] = np.float32(lanes[lane] + lanes[lane + offset])
                    counters.warp_ops += 1
            lanes = updated
        aggregates.append(lanes[0])
    counters.shared_writes += info["active_warps"]
    counters.barriers += 1
    total = aggregates[0]
    for warp in range(1, info["active_warps"]):
        total = np.float32(total + aggregates[warp])
        counters.cross_ops += 1
        counters.shared_reads += 1
    return total


def simulate_source(*, kernel, rows, width, dtype, has_weight=True, aligned=True,
                    batch_invariant=False, input_stride=None, seed=20261028,
                    epsilon=1e-5):
    """Small independent execution of the two CUDA kernels, including CUB."""
    stride = width if input_stride is None else int(input_stride)
    info = independent_dispatch(rows, width, dtype, kernel, has_weight,
                                aligned, batch_invariant, stride)
    block, vec = info["block"], info["vector_width"]
    rng = np.random.default_rng(seed)
    x = quantize(rng.normal(size=(rows, stride)), dtype)[:, :width].copy()
    residual = quantize(rng.normal(size=(rows, width)), dtype)
    weight = quantize(rng.normal(size=(width,)), dtype)
    out = np.empty((rows, width), dtype=np.float32)
    state = np.empty((rows, width), dtype=np.float32)
    counters = Counters()
    size = info["size"]
    fused = kernel == "fused_add_rms_norm"

    for row in range(rows):
        partials = np.zeros(block, dtype=np.float32)

        # The first pass writes the fused residual in low precision and
        # accumulates each thread's float32 sum of squares.
        if fused:
            z = np.empty(width, dtype=np.float32)
            for index in range(width):
                counters.hbm_read += 2 * size
                counters.load_hbm += 2 * size
                z[index] = quantize(
                    np.float32(x[row, index]) + np.float32(residual[row, index]),
                    dtype)
                counters.load_ops += 1  # residual add (packed per lane)
                counters.hbm_write += size
                counters.load_hbm += size
            state[row] = z
        else:
            z = x[row]

        if fused and vec == 8:
            # _f16Vec::sum_squares converts pairs and accumulates pair sums.
            for chunk in range(width // vec):
                thread = chunk % block
                acc = np.float32(0)
                start = chunk * vec
                for pair in range(0, vec, 2):
                    a = np.float32(z[start + pair])
                    b = np.float32(z[start + pair + 1])
                    if _cvt_is_real(dtype):
                        counters.load_ops += 2
                    counters.load_ops += 2  # two squares
                    pair_sum = np.float32(np.float32(a * a) + np.float32(b * b))
                    counters.load_ops += 1
                    acc = np.float32(acc + pair_sum)
                    counters.load_ops += 1
                partials[thread] = np.float32(partials[thread] + acc)
                counters.load_ops += 1  # outer variance += sum_squares()
        else:
            chunk_width = vec if vec > 1 else 1
            chunks = width // chunk_width
            for chunk in range(chunks):
                thread = chunk % block
                acc = partials[thread]
                start = chunk * chunk_width
                for offset in range(chunk_width):
                    value = np.float32(z[start + offset])
                    if _cvt_is_real(dtype):
                        counters.load_ops += 1
                    counters.load_ops += 1  # square
                    acc = np.float32(acc + np.float32(value * value))
                    counters.load_ops += 1
                partials[thread] = acc
            if not fused:
                counters.hbm_read += width * size
                counters.load_hbm += width * size

        total = _warp_reduce(partials, block, counters)
        counters.normalize_ops += 4  # width->float, variance/width, +epsilon, rsqrtf
        counters.barriers += 1
        scale = np.float32(1.0) / np.sqrt(
            np.float32(np.float32(total / np.float32(width)) + np.float32(epsilon)),
            dtype=np.float32)

        # Second pass reads the first-pass value (input for plain; residual
        # for fused), applies the optional weight and writes the output.
        chunk_width = vec if vec > 1 else 1
        second_chunks = width // chunk_width
        for chunk in range(second_chunks):
            start = chunk * chunk_width
            for offset in range(chunk_width):
                index = start + offset
                counters.hbm_read += size
                counters.store_hbm += size
                value = np.float32(z[index])
                if _cvt_is_real(dtype):
                    counters.store_ops += 1
                if has_weight:
                    counters.hbm_read += size
                    counters.store_hbm += size
                    weight_value = np.float32(weight[index])
                    if _cvt_is_real(dtype):
                        counters.store_ops += 1
                else:
                    weight_value = None
                scaled = np.float32(value * scale)
                counters.store_ops += 1
                if weight_value is not None:
                    scaled = np.float32(scaled * weight_value)
                    counters.store_ops += 1
                if _cvt_is_real(dtype):
                    counters.store_ops += 1
                out[row, index] = quantize(scaled, dtype)
                counters.hbm_write += size
                counters.store_hbm += size

    return {"out": out, "state": state, "counters": counters, "dispatch": info}


def reference_rms_norm(x, weight, residual, epsilon, dtype):
    """Independent float64 reference with dtype-rounded inputs/states."""
    x = np.asarray(x, dtype=np.float32)
    if residual is None:
        z = x
    else:
        z = quantize(np.float32(x) + np.float32(residual), dtype)
    variance = np.mean(np.square(z.astype(np.float64)), axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(variance + np.float64(epsilon))
    scaled = z.astype(np.float64) * scale
    if weight is not None:
        scaled = scaled * np.asarray(weight, dtype=np.float64)
    return scaled, z


def hardware(*, units=4, hbm=1e12, vector=1e12, shared_rate=1e12,
             shared=65536, mode_rates=None, pool_name="shared"):
    rates = {"hbm": hbm, "vector": vector, "shared_memory": shared_rate}
    modes = {role: {"resource": physical, "rate": rates[physical]}
             for role, physical in cuda_norm.SERVICE_ROLES.items()}
    for role, rate in (mode_rates or {}).items():
        modes[role]["rate"] = rate
    return {
        "machine": {"name": "synthetic", "num_devices": 1, "device": {
            "memory_bytes": 2**30, "compute_units": units,
            "compute_peaks": {"bf16": 1e14, "fp16": 1e14, "fp32": 2.5e13},
            "rates": rates, "resource_units": {"vector": units, "shared_memory": units},
            "service_modes": modes, "buffers": {} if shared is None else {pool_name: shared},
            "buffer_units": {} if shared is None else {pool_name: units},
        }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def execution(op, *, ready_ms=0, horizon=None, **device):
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    case = parse_case(hardware(**device))
    book = RecipeBook(recipes={op.recipe: PlannedRecipe(cuda_norm.build)})
    run = Execution(case, CostModel(case), recipes=book, trace_limit=30000)
    physical, plan = run.prepare(op)
    context = run._context(physical, 0, "none")
    done = plan.lower(context).start(
        run.engine, "norm", run.engine.submit("ready", release=ready_ms))
    if horizon is not None:
        run.engine.advance(horizon=horizon)
    else:
        run.engine.advance(until=(done,))
    return plan, run.engine, done


def public_cost(params, **device):
    op = cuda_norm.make_op(params)
    case = parse_case(hardware(**device))
    book = RecipeBook(recipes={op.recipe: PlannedRecipe(cuda_norm.build)})
    return operator_costs(case, (op,), recipes=book)["operators"][0]


def arithmetic_lane_count(work):
    return sum(value for role, value in work.items()
               if role not in {"hbm", "shared_read_bytes", "shared_write_bytes",
                               "warp_shuffle_calls"})


@pytest.mark.parametrize("rows", (1, 255, 256, 257))
def test_host_dispatch_and_cub_actual_lane_predicates(rows):
    for width in (1, 8, 40, 513, 5120, 16384):
        for dtype in ("bf16", "fp16", "fp32"):
            for kernel in cuda_norm.KERNELS:
                for aligned, invariant, stride in (
                    (True, False, width), (False, False, width),
                    (True, True, width), (True, False, width + 1),
                ):
                    expected = independent_dispatch(
                        rows, width, dtype, kernel, True, aligned, invariant, stride)
                    actual = cuda_norm.source_dispatch(
                        rows=rows, width=width, dtype=dtype, kernel=kernel,
                        aligned=aligned, batch_invariant=invariant, input_stride=stride)
                    assert (actual.grid, actual.block, actual.vector_width) == (
                        expected["grid"], expected["block"], expected["vector_width"])
                    # Interpret every actual source PTX predicate, including inactive tails.
                    adds = sum(lane + offset < min(32, actual.block - base)
                               for base in range(0, actual.block, 32)
                               for lane in range(min(32, actual.block - base))
                               for offset in (1, 2, 4, 8, 16))
                    assert actual.warp_lane_adds == adds
                    assert actual.cross_warp_adds == len(range(32, actual.block, 32))


@pytest.mark.parametrize("dtype", ("bf16", "fp16", "fp32"))
@pytest.mark.parametrize("width", (8, 40, 16384))
@pytest.mark.parametrize("has_weight", (False, True))
def test_source_vector_local_sum_then_outer_variance_accumulation(dtype, width, has_weight):
    for invariant in (False, True):
        kwargs = dict(kernel="fused_add_rms_norm", rows=1, width=width,
                      dtype=dtype, batch_invariant=invariant, has_weight=has_weight)
        simulated = simulate_source(**kwargs)
        work = cuda_norm.build(cuda_norm.make_op(kwargs), Geometry(1)).work.resources
        assert arithmetic_lane_count(work) == simulated["counters"].vector
        assert work["hbm"] == simulated["counters"].hbm
        dispatch = simulated["dispatch"]
        residual_adds = width if dtype == "fp32" else 0
        expected_adds = width + residual_adds + dispatch["warp_lane_adds"] + \
            dispatch["cross_warp_adds"] + 1 + (0 if invariant else width // 8)
        assert work["fp32_add"] == expected_adds
        if dtype != "fp32":
            assert work[dtype + "_add"] == width
            assert work["cvt_" + dtype + "_fp32"] == (2 + has_weight) * width
            assert work["cvt_fp32_" + dtype] == width


@pytest.mark.parametrize("params", (
    dict(kernel="rms_norm", rows=2, width=40, dtype="bf16"),
    dict(kernel="rms_norm", rows=2, width=513, dtype="fp32", has_weight=False),
    dict(kernel="fused_add_rms_norm", rows=2, width=40, dtype="fp16"),
    dict(kernel="fused_add_rms_norm", rows=2, width=513, dtype="bf16", aligned=False),
    dict(kernel="fused_add_rms_norm", rows=1, width=16384, dtype="fp32"),
))
def test_cpu_source_values_match_dtype_rounded_float64_reference(params):
    simulated = simulate_source(**params, seed=11)
    rng = np.random.default_rng(11)
    rows, width, dtype = params["rows"], params["width"], params["dtype"]
    x = quantize(rng.normal(size=(rows, width)), dtype)
    residual = quantize(rng.normal(size=(rows, width)), dtype)
    weight = quantize(rng.normal(size=(width,)), dtype)
    expected, state = reference_rms_norm(
        x, weight if params.get("has_weight", True) else None,
        residual if params["kernel"].startswith("fused") else None, 1e-5, dtype)
    rtol, atol = {"bf16": (3e-2, 3e-3), "fp16": (3e-3, 1e-3),
                  "fp32": (2e-5, 1e-6)}[dtype]
    np.testing.assert_allclose(simulated["out"], quantize(expected, dtype),
                               rtol=rtol, atol=atol)
    if params["kernel"].startswith("fused"):
        np.testing.assert_array_equal(simulated["state"], state)


def test_scalar_dependency_counterexample_and_distinct_typed_rates():
    params = dict(rows=1, width=1, dtype="fp32", has_weight=False)
    # Suppress the newly explicit shuffle/shared services only in this synthetic
    # original counterexample, preserving exactly its 12B + six arithmetic ops.
    modes = {"warp_shuffle_calls": 1e30, "cvt_int32_fp32": 1e30}
    row = public_cost(params, units=1, hbm=1e6, vector=1e6,
                      shared_rate=1e30, mode_rates=modes)
    assert row["duration_ms"] == pytest.approx(.018)
    slow = public_cost(params, units=1, hbm=1e6, vector=1e6,
                       shared_rate=1e30,
                       mode_rates={**modes, "fp32_rsqrt": 1e5})
    assert slow["duration_ms"] - row["duration_ms"] == pytest.approx(.009)
    assert slow["service_delta"] == {}


def test_warp_loops_preserve_read_compute_store_without_row_barrier():
    op = cuda_norm.make_op(dict(rows=1, width=513, dtype="fp32",
                               kernel="fused_add_rms_norm", aligned=False))
    _, engine, _ = execution(op, units=1, hbm=1e6, vector=1e6, shared_rate=1e6)
    work = [row for row in engine.trace if row["demand"]]
    reads = [row for row in work if row["task"].endswith(":read")]
    for read in reads:
        base = read["task"].rsplit(":", 1)[0]
        computes = sorted((row for row in work if row["task"].startswith(base + ":compute")),
                          key=lambda row: row["task"])
        write = next(row for row in work if row["task"] == base + ":write")
        assert read["end_ms"] <= computes[0]["start_ms"]
        assert computes[-1]["end_ms"] <= write["start_ms"]
    # Partial warp16 can reduce its one value before any full warp finishes32 values.
    early = next(row for row in work if
                 ":first_pass:program:group16:shuffle0" in row["task"])
    full_end = max(row["end_ms"] for row in work if
                   ":first_pass:program:group0:elements:" in row["task"])
    assert early["start_ms"] < full_end
    # Thread0's successor reads/adds form a chain after every warp's shared store.
    stores = [row for row in work if row["task"].endswith(":aggregate_store")]
    previous = max(row["end_ms"] for row in stores)
    for successor in range(1, 17):
        read = next(row for row in work if row["task"].endswith(
            f":aggregate_read{successor}"))
        add = next(row for row in work if row["task"].endswith(
            f":aggregate_add{successor}"))
        assert read["start_ms"] >= previous
        assert add["start_ms"] >= read["end_ms"]
        previous = add["end_ms"]


@pytest.mark.parametrize("pool_name", ("smem", "shared"))
def test_shared_cta_lifetime_capacity_ownership_and_missing_pool_assumption(pool_name):
    from mindie_sim.prediction.engine import ExecutionError
    op = cuda_norm.make_op(dict(rows=2, width=1, dtype="fp32", has_weight=False))
    plan, engine, done = execution(op, units=2, hbm=1e6, vector=2e6,
                                   shared=136, ready_ms=5, horizon=4.9, pool_name=pool_name)
    assert not done.done and not any(engine.serviced.values())
    assert not any(engine.allocated.values())
    engine.advance(horizon=5.001)
    assert sum(engine.allocated.values()) == 272
    assert {a.units for a in engine.allocations.values()} == {(0,), (1,)}
    engine.advance(until=(done,))
    assert engine.summary()["live_allocated"] == {pool_name + "@r0:s0": 0}
    assert engine.summary()["peak_allocated_per_unit"] == {
        pool_name + "@r0:s0:unit0": 136, pool_name + "@r0:s0:unit1": 136}
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        execution(op, units=2, shared=135, pool_name=pool_name)
    _, unlimited, _ = execution(op, units=2, shared=None)
    assert unlimited.summary()["peak_allocated"] == {}
    assert unlimited.serviced["shared_memory@r0:s0"] > 0


def test_shared_hbm_and_single_cta_fixed_per_sm_supply():
    params = dict(rows=1, width=8, dtype="bf16")
    compute_times = []
    for units in (1, 4, 32):
        memory = public_cost(params, units=units, hbm=1e9, vector=1e30,
                             shared_rate=1e30)
        assert memory["duration_ms"] == pytest.approx(64 / 1e6, rel=1e-6)
        compute = public_cost(params, units=units, hbm=1e30, vector=units*1e6,
                              shared_rate=1e30)
        compute_times.append(compute["duration_ms"])
        assert compute["service_delta"] == {}
    assert compute_times == pytest.approx([.073] * 3)


def test_missing_typed_rate_is_visible_and_no_generic_vector_fallback():
    op = cuda_norm.make_op(dict(rows=1, width=8, dtype="bf16"))
    plan = cuda_norm.build(op, Geometry(1))
    context = RecipeContext(op, {}, {"hbm": "memory", "vector": "vector"},
                            {"memory": 1e6, "vector": 1e6}, 1)
    engine = Engine([Resource("memory", 1e6), Resource("vector", 1e6)])
    with pytest.raises(ModelingError, match="physical_resource_rate_missing"):
        done = plan.program(context).start(engine, "missing", engine.submit("ready"))
        engine.advance(until=(done,))


def test_typed_ops_and_real_stride_alignment_boundary():
    base = cuda_norm.make_op(dict(rows=6, width=8, dtype="bf16"))
    typed = replace(base, local_shape=(2, 3, 8), semantic_shape=(2, 3, 8),
                    padded_shape=(2, 3, 8), launch_shape=(), metadata=())
    plain, _, _ = execution(base)
    rank3, _, _ = execution(typed)
    assert rank3.work == plain.work
    for params in (dict(rows=2, input_stride=9), dict(rows=1, aligned=False)):
        op = cuda_norm.make_op(dict(width=8, dtype="bf16", **params))
        with pytest.raises(ModelingError, match="requires_aligned_vector_rows"):
            cuda_norm.build(op, Geometry(1))
    # Row0 never addresses a second row, so its unused stride cannot misalign it.
    one = cuda_norm.make_op(dict(rows=1, width=8, dtype="bf16", input_stride=9))
    cuda_norm.build(one, Geometry(1))
    fused = cuda_norm.make_op(dict(rows=2, width=8, dtype="bf16", input_stride=9,
                                  kernel="fused_add_rms_norm"))
    _, engine, _ = execution(fused)
    assert engine.serviced["memory@r0:s0"] == 2*8*2*6


def test_source_counts_are_lazy_for_a_billion_rows():
    params = dict(rows=10**9, width=5120, dtype="bf16")
    op = cuda_norm.make_op(params)
    plan = cuda_norm.build(op, Geometry(32))
    rates = {"hbm": 1e9, "vector": 1e9, "shared_memory": 1e9}
    context = RecipeContext(op, {}, dict(zip(rates, rates)), rates, 32,
                            service_modes={role: (physical, 1e12)
                                           for role, physical in cuda_norm.SERVICE_ROLES.items()})
    program = plan.program(context)
    assert sum(stream.count for stream in program.programs) == 10**9
    assert len(program.programs) == 32
    assert plan.work.resources["hbm"] == 10**9 * 5120 * 2 * 4


@pytest.mark.parametrize("case_row", json.loads(CASE_FILE.read_text())["cases"],
                         ids=lambda row: row["id"])
def test_fixed_geometry_work_conservation_through_real_execution(case_row):
    op = cuda_norm.make_op(case_row["params"])
    result = verify_operator(parse_case(hardware()), op, cuda_norm.build,
                             input_ready_ms=3)
    assert result["status"] == "passed"
    assert result["work_mismatches"] == {}
    assert result["result"]["remaining_buffers"] == {"shared@r0:s0": 0}
    assert result["result"]["service_before_input"] == {}
