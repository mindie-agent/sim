"""Finite analytical mapping selection for the hierarchical matrix program.

[Role] Screen declared tile candidates and rank their resource lower bounds.
[Invariants] Selection uses shape, rates, ownership and capacities, never timings
  or model labels. The selected mapping executes on the ordinary causal Engine.
  Its selection bound is neither the execution duration nor an optimality claim.
[Timeouts & Locks] O(candidates * cores * cores) arithmetic, no I/O or locks.
[Side-Effect Policy] Emits bounded mapping diagnostics through the caller only.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .matrix import HierarchicalMatmul
from .matrix_epilogue import MatrixEpilogue
from .ownership import read_role


DEFAULT_CANDIDATES = ((128, 128, 64), (128, 128, 128), (128, 256, 64), (128, 256, 128),
                      (256, 128, 64), (256, 128, 128), (256, 256, 64), (256, 256, 128))
NARROW_CANDIDATES = ((16, 16, 64), (32, 16, 64), (64, 16, 64),
                     (16, 32, 64), (32, 32, 64), (64, 32, 64))
# FP32 projections may expose smaller independent M/N blocks. Ordinary low
# precision projections retain their declared 128-element base mapping: a
# feasible block is not evidence that the deployed kernel dispatches it.
UNDERFILLED_CANDIDATES = tuple((m, n, k) for m in (16, 32, 64)
                              for n in (16, 32, 64) for k in (64, 128))


def owner_tiles(m, n, tm, tn, cores, groups=1):
    """Return count, summed M extents, N extents and output areas per owner."""
    nm, nn = math.ceil(m / tm), math.ceil(n / tn)
    if not nm:
        return ()
    dm, dn = nm * tm - m, nn * tn - n
    last_m_start = (nm - 1) * nn
    last_n_counts = [0] * cores
    period = cores // math.gcd(nn, cores)
    for row in range(min(nm, period)):
        last_n_counts[(row * nn + nn - 1) % cores] += (nm - 1 - row) // period + 1
    result = []
    for core in range(min(cores, nm * nn)):
        count = (nm * nn - 1 - core) // cores + 1
        before = max(0, (last_m_start - 1 - core) // cores + 1)
        last_m = count - before
        last_n = last_n_counts[core]
        corner = int((nm * nn - 1) % cores == core)
        result.append((count, count * tm - last_m * dm, count * tn - last_n * dn,
                       count * tm * tn - last_m * dm * tn - last_n * dn * tm
                       + corner * dm * dn))
    if groups == 1:
        return tuple(result)
    totals = [[0, 0, 0, 0] for _ in range(cores)]
    period = cores // math.gcd(nm * nn, cores)
    for group in range(min(groups, period)):
        repeats = (groups - 1 - group) // period + 1
        for core, values in enumerate(result):
            target = totals[(core + group * nm * nn) % cores]
            for axis, value in enumerate(values):
                target[axis] += repeats * value
    return tuple(tuple(row) for row in totals if row[0])


def mapping_bound(context, recipe):
    """Resource-only lower bound with private copies and actual owner tails."""
    total_m, n, k = context.op.local_shape[-3:]
    if not total_m:
        return 0.0
    names, meta = context.resource_names, dict(context.op.metadata)
    groups = int(meta.get("matrix_groups", 1))
    m = total_m // groups
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    owners = owner_tiles(m, n, recipe.tile_m, recipe.tile_n, context.core_count, groups)
    private = [inputs * a / total_m + weight * b / (n * groups)
               for _, a, b, _ in owners]
    scales = [epilogue.row_scale_bytes * a / total_m
              + epilogue.column_scale_bytes * b / (n * groups)
              for _, a, b, _ in owners]
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    left = owner_tiles(m, math.ceil(n / recipe.compute_n), recipe.tile_m,
                       recipe.tile_n // recipe.compute_n, context.core_count, groups)
    right = owner_tiles(math.ceil(m / recipe.compute_m), n,
                        recipe.tile_m // recipe.compute_m, recipe.tile_n, context.core_count, groups)
    inner = [a[3] * max(inputs / total_m, k * width) + b[3] * max(weight / (n * groups), k * width)
             for a, b in zip(left, right)]
    if recipe.reuse_inputs and recipe.reuse_weights:
        hbm = (inputs + weight + epilogue.row_scale_bytes
               + epilogue.column_scale_bytes + context.op.mte3_bytes)
    else:
        hbm = (((inputs + epilogue.row_scale_bytes) *
                (1 if recipe.reuse_inputs else math.ceil(n / recipe.tile_n)))
               + ((weight + epilogue.column_scale_bytes) *
                  (1 if recipe.reuse_weights else math.ceil(m / recipe.tile_m)))
               + context.op.mte3_bytes)
    bound = hbm / context.rates[names["hbm"]]
    demands = {}
    for role, values in (
        (read_role(names), private),
        (read_role(names, aiv=0), scales),
        ("mte1", inner),
        ("mte3", [context.op.mte3_bytes * area / (total_m * n) for _, _, _, area in owners]),
        ("cube", [context.work.get(names["cube"], 0) * area / (total_m * n)
                  for _, _, _, area in owners]),
        ("vector", [epilogue.operations_per_output * area for _, _, _, area in owners]),
    ):
        if role not in names:
            continue
        # Ownership can include several pipes or share one. Use the same
        # physical unit groups as program execution, including uneven groups.
        units = context.resource_units.get(role, context.core_count)
        resource = names[role]
        for core, value in enumerate(values):
            selected = context.owned(core, context.core_count, (role,))[resource]
            for unit in selected:
                key = resource, unit
                # Both read phases may share one pipe. Sum its service before
                # taking a bound, preserving contention across logical roles.
                demands[key] = demands.get(key, 0) + value / len(selected) * units / context.rates[resource]
    bound = max(bound, max(demands.values(), default=0))
    return bound


def footprint(context, recipe):
    total_m, n, k = context.op.local_shape[-3:]
    meta = dict(context.op.metadata)
    groups = int(meta.get("matrix_groups", 1))
    m = total_m // groups
    a, b, kd = min(m, recipe.tile_m), min(n, recipe.tile_n), min(k, recipe.tile_k)
    ai, bi = min(a, recipe.compute_m), min(b, recipe.compute_n)
    inputs, weight = MatrixEpilogue.from_op(context.op).operands(context.op)
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    output = dtype_bytes(meta.get("output_format", "bf16"))
    left = inputs * a / total_m * kd / k if total_m else 0
    right = weight * b / (n * groups) * kd / k
    return {"l1": recipe.slots * (left + right), "l0c": a * b * max(4, output),
            "l0a": recipe.compute_slots * max(left * ai / a if a else 0, ai * kd * width),
            "l0b": recipe.compute_slots * max(right * bi / b, bi * kd * width)}


_SELECT_CACHE = {}


def _tuples(items):
    return None if items is None else tuple(tuple(x) for x in items)


def _matmul_recipe_key(recipe):
    return (
        type(recipe), _tuples(recipe.candidates), recipe.compute_m, recipe.compute_n,
        recipe.slots, recipe.compute_slots, recipe.reuse_inputs, recipe.reuse_weights,
        _tuples(recipe.narrow_candidates), _tuples(recipe.underfilled_candidates),
        getattr(recipe, "compact_output_tiles", None),
    )


def _op_matrix_key(op):
    meta = dict(op.metadata)
    return (
        op.local_shape, op.dtype, op.mte2_bytes, op.mte3_bytes, op.vector_ops,
        meta.get("matrix_groups", 1), meta.get("compute_format"), meta.get("output_format"),
        meta.get("packed_weight_bytes", 0), meta.get("row_scale_bytes", 0),
        meta.get("column_scale_bytes", 0), meta.get("epilogue_ops_per_element", 0),
        MatrixEpilogue.from_op(op),
    )


@dataclass(frozen=True)
class MappedMatmul:
    candidates: tuple | None = None
    compute_m: int = 128
    compute_n: int = 128
    slots: int = 2
    compute_slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True
    narrow_candidates: tuple | None = None
    underfilled_candidates: tuple | None = None

    def select(self, context):
        cache_key = (
            _matmul_recipe_key(self), _op_matrix_key(context.op), context.core_count,
            tuple(sorted(context.resource_names.items())),
            tuple(sorted(context.rates.items())),
            tuple(sorted(context.work.items())),
            tuple(sorted(context.resource_units.items())),
            tuple(sorted(context.pool_capacities.items())),
        )
        if cache_key in _SELECT_CACHE:
            recipe, record = _SELECT_CACHE[cache_key]
            return recipe, deepcopy(record)
        records = []
        selected = None
        meta = dict(context.op.metadata)
        groups = int(meta.get("matrix_groups", 1))
        if groups < 1 or context.op.local_shape[-3] % groups:
            raise ModelingError("matrix_groups_require_equal_integer_rows")
        compute_format = meta.get("compute_format", context.op.dtype or "bf16")
        automatic_subtiles = self.candidates is None and compute_format == "fp32"
        narrow = (NARROW_CANDIDATES if automatic_subtiles else ()
                  ) if self.narrow_candidates is None else self.narrow_candidates
        narrow = narrow if context.op.local_shape[-2] <= 32 else ()
        m = context.op.local_shape[-3] // groups
        n = context.op.local_shape[-2]
        default_grid = math.ceil(m / 128) * math.ceil(n / 128) * groups
        underfilled = (UNDERFILLED_CANDIDATES if automatic_subtiles else ()
                       ) if self.underfilled_candidates is None else self.underfilled_candidates
        underfilled = underfilled if default_grid < context.core_count else ()
        candidates = [(tile, False) for tile in
                      (DEFAULT_CANDIDATES if self.candidates is None else self.candidates)]
        candidates.extend((tile, True) for tile in narrow)
        candidates.extend((tile, True) for tile in underfilled if tile not in narrow)
        for (tm, tn, tk), is_narrow in candidates:
            cm = min(self.compute_m, tm) if is_narrow else self.compute_m
            cn = min(self.compute_n, tn) if is_narrow else self.compute_n
            recipe = HierarchicalMatmul(
                tm, tn, tk, cm, cn,
                self.slots, self.compute_slots, self.reuse_inputs, self.reuse_weights)
            if min(tm, tn, tk, self.compute_m, self.compute_n,
                   self.slots, self.compute_slots) < 1:
                raise ModelingError("matrix_tile_dimensions_and_slots_must_be_positive")
            if tm % cm or tn % cn:
                raise ModelingError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
            buffers = footprint(context, recipe)
            exceeded = {pool: size for pool, size in buffers.items()
                        if pool in context.pool_capacities and size > context.pool_capacities[pool]}
            bound = None if exceeded else mapping_bound(context, recipe)
            waves = math.ceil(math.ceil(m / tm) * math.ceil(n / tn) * groups / context.core_count)
            records.append({"tile": [tm, tn, tk], "footprint_bytes_per_owner": buffers,
                            "exceeded_pools": exceeded, "resource_bound_ms": bound,
                            "maximum_owner_output_tiles": waves})
            if bound is not None:
                # Equal resource bounds can hide another serial output-tile
                # drain/prologue on an owner. Avoid those extra waves before
                # minimizing residency; smaller buffers alone are not faster.
                score = (bound, waves, -tk, sum(buffers.values()), tm, tn)
                if selected is None or score < selected[0]:
                    selected = (score, recipe)
        if selected is None:
            raise ModelingError(f"matrix_no_feasible_declared_mapping:"
                                f"shape={context.op.local_shape}:groups={groups}:dtype={context.op.dtype}")
        res = selected[1], {
            "basis": "declared_candidates_resource_lower_bound",
            "shape": list(context.op.local_shape[-3:]),
            "groups": groups,
            "default_128_grid_tiles": default_grid,
            "physical_compute_owners": context.core_count,
            "subtile_policy": ("fp32_default_or_explicit" if automatic_subtiles
                               else "explicit_only"),
            "underfilled_grid_candidates": bool(underfilled),
            "selected": asdict(selected[1]), "candidates": records,
            "capacity_screening": "declared_pools_only",
            "declared_pools": dict(context.pool_capacities),
            "read_role_bindings": {
                "matrix_operands": read_role(context.resource_names),
                "output_scales": read_role(context.resource_names, aiv=0),
            },
            "unmodeled_transfer_roles": [role for role in (
                                        "mte1", read_role(context.resource_names),
                                        "mte3")
                                        if role not in context.resource_names],
            "representation_policy": "packed_payload_and_scales_with_compute_width_bound",
            "conversion_latency": "not_inferred_from_transfer_bytes",
        }
        if len(_SELECT_CACHE) >= 1024:
            _SELECT_CACHE.pop(next(iter(_SELECT_CACHE)))
        _SELECT_CACHE[cache_key] = res
        return res[0], deepcopy(res[1])

    def lower(self, context):
        recipe, record = self.select(context)
        if context.record_mapping is not None:
            context.record_mapping(record)
        return recipe.lower(context)
