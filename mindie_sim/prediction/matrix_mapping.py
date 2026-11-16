"""Finite analytical mapping selection for the hierarchical matrix program.

[Role] Derive finite matrix geometry/owner tails, screen declared tile
  candidates, and expose one isolated matrix-operator entry.
[Invariants] Selection uses shape, rates, ownership and capacities, never timings
  or model labels. The selected mapping executes on the ordinary causal Engine.
  Its selection bound is neither the execution duration nor an optimality claim.
  Logical/local/storage/issued extents remain separate values.
[Timeouts & Locks] O(candidates * cores) arithmetic plus bounded explicit
  owner enumeration, no I/O or locks.
[Side-Effect Policy] Emits bounded mapping diagnostics through the caller only;
  the public entry runs only the caller's virtual Engine.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from itertools import count as _count
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .matrix import HierarchicalMatmul
from .matrix_epilogue import MatrixEpilogue
from .ownership import read_role


ZERO = 0
ONE = 1
FIRST = 0
LAST = -1
SECOND_LAST = -2
LAST_THREE = -3
REFILL_M = 128
MIN_NARROW_N = 32
MAX_SPLIT_K = 32
A5_L2_MTE2_RATE = 1e13
A5_BF16_PEAK_FLOOR = 4e14

DEFAULT_CANDIDATES = ((128, 128, 64), (128, 128, 128), (128, 256, 64), (128, 256, 128),
                      (256, 128, 64), (256, 128, 128), (256, 256, 64), (256, 256, 128))
NARROW_CANDIDATES = ((16, 16, 64), (32, 16, 64), (64, 16, 64),
                     (16, 32, 64), (32, 32, 64), (64, 32, 64))
# FP32 projections may expose smaller independent M/N blocks. Ordinary low
# precision projections retain their declared 128-element base mapping: a
# feasible block is not evidence that the deployed kernel dispatches it.
UNDERFILLED_CANDIDATES = tuple((m, n, k) for m in (16, 32, 64)
                              for n in (16, 32, 64) for k in (64, 128))


def mte_owner_unit_count(resource_units):
    """Hardware unit count that pins one matrix output owner.

    Native ``owned(core, core_count, read_role)`` partitions ``mte2_aic``.
    A supply that never declares those units has no per-owner split.
    """
    units = dict(resource_units or ())
    for name in ("mte2_aic", "mte2", "compute_units"):
        value = int(units.get(name, 0) or 0)
        if value > 0:
            return value
    return 0


def gemm_fractal_operand_bytes(tokens, n_tiles, n_pad, k_pad, act_bytes, weight_bytes,
                               *, fractal=16):
    """Summed operand bytes of the piecewise GEMM copy, before partials.

    Activation is copied once per N tile at the 16-row fractal height.
    Weights move once at the padded N.  Split-K cancels: each split carries
    ``K / splits``.  This is the family's summed copy, not the slowest owner.
    """
    tokens = int(tokens)
    if tokens <= 0 or int(n_tiles) <= 0 or int(k_pad) <= 0:
        return Fraction(0)
    rows = (tokens + int(fractal) - 1) // int(fractal) * int(fractal)
    act = Fraction(act_bytes)
    weight = Fraction(weight_bytes)
    return (Fraction(int(n_tiles)) * rows * Fraction(int(k_pad)) * act
            + Fraction(int(k_pad)) * Fraction(int(n_pad)) * weight)


def critical_output_owner_rate_fraction(
    m, n, k, tile_m, tile_n, units, *,
    act_bytes, weight_bytes, charged_bytes=None, groups=1,
):
    """Share of an aggregate pipe that elapses as the slowest output owner.

    ``SteadyMatmul`` pins each output tile with ``owned(core, core_count, ...)``.
    Idle units stay idle.  One tile on a 20-unit pipe therefore runs at
    ``1/20`` of the aggregate, and the makespan is the maximum private copy

        act_bytes * m_extent * k + weight_bytes * n_extent * k

    using the exact local tail from ``owner_tiles`` (a 288-wide N on a
    128 tile is 128, 128, 32, not the average 96).  Piecewise stage work
    sums every owner and, for the GEMM family, the fractal row pad.  Passing
    that summed ``charged_bytes`` makes

        charged / (aggregate * fraction) = max_copy / (aggregate / units)

    without rewriting the charged bytes and without reading a duration.
    HBM is unpinned and is not this share.  A missing unit count returns 1.
    """
    units = int(units)
    if units <= 0 or int(m) <= 0 or int(n) <= 0 or int(k) <= 0:
        return Fraction(1)
    owners = owner_tiles(
        int(m), int(n), max(1, int(tile_m)), max(1, int(tile_n)),
        units, int(groups))
    if not owners:
        return Fraction(1)
    act = Fraction(act_bytes)
    weight = Fraction(weight_bytes)
    kk = Fraction(int(k))
    m_tiles = max(1, (int(m) + max(1, int(tile_m)) - 1) // max(1, int(tile_m)))
    copies = [
        act * Fraction(extent_m) * kk
        + (weight * Fraction(extent_n) * kk / Fraction(m_tiles))
        for _count, extent_m, extent_n, _area in owners
    ]
    peak = max(copies)
    if peak <= 0:
        return Fraction(1)
    # The W panel is reused across the M tiles; the stage ledger charges W
    # once while each owner's local panel is replicated per M tile.  Use the
    # same schedule in the denominator.  For one M tile this is exactly the
    # old owner-copy law (sparse/tail/single-owner); only real tails lower it.
    basis = Fraction(charged_bytes) if charged_bytes is not None else sum(copies, Fraction(0))
    if basis <= 0:
        return Fraction(1)
    return min(Fraction(1), basis / (peak * Fraction(units)))


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


def freeze_owner_tiles(tile_sources):
    """Materialize a finite owner-tile declaration exactly once."""
    result = []
    for count, shapes in tile_sources:
        shapes = tuple((int(a), int(b)) for a, b in shapes)
        result.append((int(count), shapes))
    return tuple(result)


def owner_records(tile_sources):
    """Aggregate finite explicit owner rectangles without losing tails.

    ``tile_sources`` is the same shape accepted by ``HierarchicalMatmul``:
    ``((count, ((m, n), ...)), ...)`` for active owners.  The returned record
    is the common ``(count, summed_m, summed_n, output_area)`` tuple used by
    ``owner_tiles``.  This entry lets a grouped/source adapter supply unequal
    group rows without a second matrix geometry implementation.
    """
    result = []
    for count, shapes in tile_sources:
        shapes = tuple((int(a), int(b)) for a, b in shapes)
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or len(shapes) != count):
            raise ModelingError("matrix_explicit_owner_tiles_require_matching_count")
        if any(a < 0 or b < 0 for a, b in shapes):
            raise ModelingError("matrix_explicit_owner_tiles_require_nonnegative_extents")
        result.append((count, sum(a for a, _ in shapes),
                       sum(b for _, b in shapes),
                       sum(a * b for a, b in shapes)))
    return tuple(row for row in result if row[0])


def grouped_owner_rectangles(group_rows, n, tm, tn, cores, *, max_tiles=1_000_000):
    """Finite group-major owner rectangles for explicit group rows.

    Each group contributes its own tail rows; rotating owners never replaces a
    group's row extent by the widest group.  Empty groups consume no tile.  The
    bound prevents a large declaration from becoming an unbounded enumeration.
    This is the executable entry consumed by both the detailed and steady
    matrix recipes; ``grouped_owner_tiles`` is its aggregate projection.
    """
    rows = tuple(int(value) for value in group_rows)
    n, tm, tn, cores = int(n), int(tm), int(tn), int(cores)
    if cores < 1 or tm < 1 or tn < 1 or n < 0 or any(row < 0 for row in rows):
        raise ModelingError("matrix_grouped_owner_geometry_invalid")
    tiles = sum(math.ceil(row / tm) for row in rows) * math.ceil(n / tn)
    if tiles > int(max_tiles):
        raise ModelingError(f"matrix_grouped_owner_enumeration_exceeds_bound:{tiles}")
    owners = [[] for _ in range(cores)]
    index = 0
    for row in rows:
        for start in range(0, row, tm):
            a = min(tm, row - start)
            for column in range(0, n, tn):
                b = min(tn, n - column)
                owners[index % cores].append((a, b))
                index += 1
    return tuple((len(shapes), tuple(shapes)) for shapes in owners if shapes)


def grouped_owner_tiles(group_rows, n, tm, tn, cores, *, max_tiles=1_000_000):
    """Owner/tail aggregate for explicit, possibly unequal group rows."""
    return owner_records(grouped_owner_rectangles(
        group_rows, n, tm, tn, cores, max_tiles=max_tiles))


@dataclass(frozen=True)
class MatrixGeometry:
    """The four distinct matrix geometries plus finite owner/tail assignments.

    ``logical`` is the requested semantic extent, ``local`` is the rank/group
    extent executed here, ``storage`` is the declared padded address extent,
    and ``issued`` is the macro-tile extent that the selected mapping really
    launches.  Keeping them separate prevents a padded storage shape from
    silently becoming useful work, and prevents a logical tail from becoming a
    full tile charge without a declared mapping.
    """

    logical_shape: tuple[int, int, int]
    local_shape: tuple[int, int, int]
    storage_shape: tuple[int, int, int]
    issued_shape: tuple[int, int, int]
    groups: int
    group_rows: tuple[int, ...]
    explicit_groups: bool
    tile_m: int
    tile_n: int
    tile_k: int
    compute_m: int
    compute_n: int
    owners: tuple[tuple[int, int, int, int], ...]

    @property
    def m_per_group(self):
        """Per-group M extent used by the macro tile launcher.

        Equal groups use the exact quotient.  Explicit unequal groups use the
        widest group for issued macro extents; per-owner records retain the
        actual tail rows.
        """
        return max(self.group_rows, default=0)

    @property
    def n(self):
        return self.local_shape[1]

    @property
    def k(self):
        return self.local_shape[2]

    @property
    def total_rows(self):
        return sum(self.group_rows)

    @property
    def nm(self):
        return (math.ceil(self.m_per_group / self.tile_m)
                if self.m_per_group else 0)

    @property
    def nn(self):
        return math.ceil(self.n / self.tile_n) if self.n else 0

    @property
    def nk(self):
        return math.ceil(self.k / self.tile_k) if self.k else 0

    @property
    def tiles(self):
        return sum(record[0] for record in self.owners)

    @property
    def tile_counts(self):
        return {"m": self.nm, "n": self.nn, "k": self.nk}

    @property
    def output_tiles(self):
        return self.tiles

    @property
    def owner_waves(self):
        """Largest finite output-tile run assigned to one owner."""
        return max((record[0] for record in self.owners), default=0)

    @property
    def useful_flops(self):
        return 2 * self.total_rows * self.n * self.k

    @property
    def macro_tile_area(self):
        """Full macro-tile output area, including tail padding."""
        return sum(record[0] * self.tile_m * self.tile_n for record in self.owners)

    @property
    def macro_tile_envelope(self):
        """Full macro-tile arithmetic envelope, including M/N/K tails.

        This is the finite dispatch extent of the selected macro mapping, not
        executed/costed work and not a native instruction schedule.  Each
        fired tile carries the declared tile shape even when its useful tail
        is smaller; ``context.work``/``serviced`` remain the costed work.
        """
        return 2 * self.macro_tile_area * self.nk * self.tile_k

    def to_dict(self):
        return {
            "logical_shape": list(self.logical_shape),
            "local_shape": list(self.local_shape),
            "storage_shape": list(self.storage_shape),
            "issued_shape": list(self.issued_shape),
            "groups": self.groups,
            "group_rows": list(self.group_rows) if self.explicit_groups else [],
            "explicit_groups": self.explicit_groups,
            "tile": [self.tile_m, self.tile_n, self.tile_k],
            "compute_tile": [self.compute_m, self.compute_n],
            "tile_counts": {"m": self.nm, "n": self.nn, "k": self.nk},
            "owner_count": len(self.owners),
            "output_tiles": self.tiles,
            "owner_waves": self.owner_waves,
            "useful_flops": self.useful_flops,
            "macro_tile_area": self.macro_tile_area,
            "macro_tile_envelope": self.macro_tile_envelope,
            "macro_tile_envelope_basis": "full_macro_tile_extent_not_costed_work",
            "owners": [
                {"owner": index, "tiles": count, "m_extent": m_extent,
                 "n_extent": n_extent, "area": area}
                for index, (count, m_extent, n_extent, area)
                in enumerate(self.owners)
            ],
        }


def _validated_shape(value, fallback):
    if value is None or len(value) < 3:
        return fallback
    shape = tuple(int(item) for item in value[-3:])
    if any(item < 0 for item in shape):
        raise ModelingError("matrix_geometry_requires_nonnegative_shapes")
    return shape


def matrix_geometry(context, recipe, *, owners=None, group_rows=None):
    """Derive the common logical/local/storage/issued geometry once.

    ``owners`` accepts finite explicit owner rectangles for an unequal-group or
    source adapter.  Without it, equal group rows use the closed-form
    ``owner_tiles`` recurrence; unequal ``group_rows`` use the bounded explicit
    aggregate.  The returned object is the single shape/owner source consumed
    by mapping bounds, footprint screening and the executing recipes.
    """
    op = context.op
    local = _validated_shape(op.local_shape, ())
    if len(local) != 3:
        raise ModelingError("matrix_geometry_requires_three_local_dimensions")
    total_m, n, k = local
    groups = int(dict(op.metadata).get("matrix_groups", 1))
    if groups < 1:
        raise ModelingError("matrix_groups_require_positive_integer")
    if min(int(recipe.tile_m), int(recipe.tile_n), int(recipe.tile_k),
           int(recipe.compute_m), int(recipe.compute_n)) < 1:
        raise ModelingError("matrix_tile_dimensions_must_be_positive")
    if (int(recipe.tile_m) % int(recipe.compute_m)
            or int(recipe.tile_n) % int(recipe.compute_n)):
        raise ModelingError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
    if owners is not None:
        records = tuple(tuple(int(item) for item in row) for row in owners)
        if len(records) > int(context.core_count):
            raise ModelingError("matrix_explicit_owner_count_exceeds_physical_owners")
        if any(len(row) != 4 or row[0] < 0 or row[1] < 0 or row[2] < 0 or row[3] < 0
               for row in records):
            raise ModelingError("matrix_explicit_owner_records_invalid")
    else:
        records = None

    if group_rows is None:
        if total_m % groups:
            raise ModelingError("matrix_groups_require_equal_integer_rows")
        per_group = total_m // groups
        group_rows = (per_group,) * groups
        explicit_groups = False
    else:
        group_rows = tuple(int(row) for row in group_rows)
        if len(group_rows) != groups or any(row < 0 for row in group_rows):
            raise ModelingError("matrix_group_rows_require_one_nonnegative_row_count_per_group")
        if sum(group_rows) != total_m:
            raise ModelingError("matrix_group_rows_must_sum_to_local_rows")
        explicit_groups = True

    if records is None:
        if len(set(group_rows)) == 1:
            records = owner_tiles(group_rows[0], n, recipe.tile_m, recipe.tile_n,
                                  context.core_count, groups)
        else:
            records = grouped_owner_tiles(
                group_rows, n, recipe.tile_m, recipe.tile_n, context.core_count)
    else:
        if sum(row[3] for row in records) != total_m * n:
            raise ModelingError("matrix_explicit_owner_area_must_cover_local_output")
    if sum(row[3] for row in records) != total_m * n:
        raise ModelingError("matrix_owner_area_must_cover_local_output")

    logical = _validated_shape(op.semantic_shape, local)
    padded = _validated_shape(op.padded_shape, local)
    storage = tuple(max(local[index], padded[index]) for index in range(3))
    max_rows = max(group_rows, default=0)
    issued = (
        math.ceil(max_rows / recipe.tile_m) * recipe.tile_m,
        math.ceil(n / recipe.tile_n) * recipe.tile_n,
        math.ceil(k / recipe.tile_k) * recipe.tile_k,
    )
    return MatrixGeometry(
        logical_shape=logical, local_shape=local, storage_shape=storage,
        issued_shape=issued, groups=groups, group_rows=group_rows,
        explicit_groups=explicit_groups, tile_m=int(recipe.tile_m),
        tile_n=int(recipe.tile_n), tile_k=int(recipe.tile_k),
        compute_m=int(recipe.compute_m), compute_n=int(recipe.compute_n),
        owners=tuple(records),
    )


def physical_owner_units(context, resource):
    """Declared physical units reached by one Engine resource name."""
    values = {int(context.resource_units.get(role, context.core_count))
              for role, name in context.resource_names.items() if name == resource}
    if not values:
        return int(context.core_count)
    if len(values) != 1:
        raise ModelingError(f"matrix_resource_owner_units_conflict:{resource}")
    return values.pop()


def _declared_rate(context, resource):
    """Engine rate for one declared pipe; cube is canonical peak_ms work."""
    if resource in context.rates:
        return float(context.rates[resource])
    if resource == context.resource_names.get("cube"):
        return 1.0
    raise ModelingError(f"matrix_resource_rate_missing:{resource}")


def physical_owner_rate(context, resource, placement=None):
    """Isolated service of one owner's fixed physical unit subset.

    The declared Engine rate is an aggregate over all units.  A stage pinned to
    ``placement`` receives only its fraction of that aggregate; an unpinned
    stage (for example shared HBM) receives the full declared rate.  This is a
    capacity projection, not a calibration factor.
    """
    rate = _declared_rate(context, resource)
    if not placement:
        return rate
    units = physical_owner_units(context, resource)
    if units < 1 or len(placement) > units:
        raise ModelingError(f"matrix_placement_exceeds_resource_units:{resource}")
    return rate * len(placement) / units


def mapping_bound(context, recipe, *, geometry=None):
    """Resource-only lower bound with private copies and actual owner tails."""
    geometry = geometry or matrix_geometry(context, recipe)
    if len(set(geometry.group_rows)) > 1:
        raise ModelingError("matrix_mapping_bound_requires_equal_group_rows")
    total_m, n, k = geometry.local_shape
    if not total_m:
        return 0.0
    names, meta = context.resource_names, dict(context.op.metadata)
    groups = geometry.groups
    m = geometry.m_per_group
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    owners = geometry.owners
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
    bound = hbm / _declared_rate(context, names["hbm"])
    demands = {}
    for role, values in (
        (read_role(names), private),
        (read_role(names, aiv=FIRST), scales),
        ("mte1", inner),
        ("mte3", [context.op.mte3_bytes * area / (total_m * n) for _, _, _, area in owners]),
        ("cube", [context.work.get(names["cube"], ZERO) * area / (total_m * n)
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
                demands[key] = (demands.get(key, 0) + value / len(selected)
                               * units / _declared_rate(context, resource))
    bound = max(bound, max(demands.values(), default=0))
    return bound


def footprint(context, recipe, *, geometry=None):
    geometry = geometry or matrix_geometry(context, recipe)
    total_m, n, k = geometry.local_shape
    meta = dict(context.op.metadata)
    groups = geometry.groups
    m = geometry.m_per_group
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
_OPERATOR_SEQUENCE = _count()


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
        op.local_shape, op.semantic_shape, op.padded_shape,
        op.dtype, op.mte2_bytes, op.mte3_bytes, op.vector_ops,
        meta.get("matrix_groups", ONE), meta.get("compute_format"), meta.get("output_format"),
        meta.get("packed_weight_bytes", ZERO), meta.get("row_scale_bytes", ZERO),
        meta.get("column_scale_bytes", ZERO), meta.get("epilogue_ops_per_element", ZERO),
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
            getattr(context.physical_mechanisms, "signature", None),
        )
        if cache_key in _SELECT_CACHE:
            recipe, record = _SELECT_CACHE[cache_key]
            return recipe, deepcopy(record)
        records = []
        selected = None
        selected_geometry = None
        meta = dict(context.op.metadata)
        groups = int(meta.get("matrix_groups", ONE))
        if groups < ONE or context.op.local_shape[LAST_THREE] % groups:
            raise ModelingError("matrix_groups_require_equal_integer_rows")
        compute_format = meta.get("compute_format", context.op.dtype or "bf16")
        m = context.op.local_shape[LAST_THREE] // groups
        n = context.op.local_shape[SECOND_LAST]
        default_grid = math.ceil(m / REFILL_M) * math.ceil(n / REFILL_M) * groups
        a5_l2_supply = (
            float(context.rates.get("mte2_aic", ZERO)) >= A5_L2_MTE2_RATE
            or float(context.compute_peaks.get("bf16", ZERO)) >= A5_BF16_PEAK_FLOOR)
        automatic_subtiles = a5_l2_supply and (
            self.candidates is None
            or m < context.core_count * REFILL_M
            or default_grid < context.core_count)
        narrow = (NARROW_CANDIDATES if automatic_subtiles else ()
                  ) if self.narrow_candidates is None else self.narrow_candidates
        narrow = narrow if n <= MIN_NARROW_N else ()
        underfilled = (UNDERFILLED_CANDIDATES if automatic_subtiles else ()
                       ) if self.underfilled_candidates is None else self.underfilled_candidates
        underfilled = underfilled if default_grid < context.core_count else ()
        candidates = [(tile, False) for tile in
                      (DEFAULT_CANDIDATES if self.candidates is None else self.candidates)]
        candidates.extend((tile, True) for tile in narrow)
        candidates.extend((tile, True) for tile in underfilled if tile not in narrow)
        for (tm, tn, tk), is_narrow in candidates:
            del is_narrow
            cm = min(self.compute_m, tm)
            cn = min(self.compute_n, tn)
            k = context.op.local_shape[LAST]
            output_tiles = (math.ceil(m / tm) * math.ceil(n / tn) * groups)
            split_k = ONE
            if output_tiles < context.core_count and k > tk:
                split_k = min(
                    k // tk,
                    max(ONE, context.core_count // max(ONE, output_tiles)),
                    MAX_SPLIT_K,
                )
            recipe = HierarchicalMatmul(
                tm, tn, tk, cm, cn,
                self.slots, self.compute_slots, self.reuse_inputs,
                self.reuse_weights, split_k)
            if min(tm, tn, tk, self.compute_m, self.compute_n,
                   self.slots, self.compute_slots) < ONE:
                raise ModelingError("matrix_tile_dimensions_and_slots_must_be_positive")
            if tm % cm or tn % cn:
                raise ModelingError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
            geometry = matrix_geometry(context, recipe)
            buffers = footprint(context, recipe, geometry=geometry)
            exceeded = {pool: size for pool, size in buffers.items()
                        if pool in context.pool_capacities and size > context.pool_capacities[pool]}
            bound = None if exceeded else mapping_bound(context, recipe, geometry=geometry)
            waves = math.ceil(math.ceil(m / tm) * math.ceil(n / tn) * groups / context.core_count)
            records.append({"tile": [tm, tn, tk], "split_k": split_k,
                            "footprint_bytes_per_owner": buffers,
                            "exceeded_pools": exceeded, "resource_bound_ms": bound,
                            "maximum_owner_output_tiles": waves,
                            "logical_shape": list(geometry.logical_shape),
                            "local_shape": list(geometry.local_shape),
                            "storage_shape": list(geometry.storage_shape),
                            "issued_shape": list(geometry.issued_shape),
                            "useful_flops": geometry.useful_flops,
                            "macro_tile_area": geometry.macro_tile_area,
                            "macro_tile_envelope": geometry.macro_tile_envelope,
                            "output_tiles": geometry.tiles})
            if bound is not None:
                # Equal resource bounds can hide another serial output-tile
                # drain/prologue on an owner. Avoid those extra waves before
                # minimizing residency; smaller buffers alone are not faster.
                score = (bound, waves, -split_k, -tk, sum(buffers.values()), tm, tn)
                if selected is None or score < selected[0]:
                    selected = (score, recipe)
                    selected_geometry = geometry
        if selected is None:
            raise ModelingError(f"matrix_no_feasible_declared_mapping:"
                                f"shape={context.op.local_shape}:groups={groups}:dtype={context.op.dtype}")
        res = selected[ONE], {
            "basis": "declared_candidates_resource_lower_bound",
            "shape": list(context.op.local_shape[LAST_THREE:]),
            "groups": groups,
            "default_128_grid_tiles": default_grid,
            "physical_compute_owners": context.core_count,
            "subtile_policy": ("all_dtype_underfilled_or_explicit"
                               if automatic_subtiles else "explicit_only"),
            "underfilled_grid_candidates": bool(underfilled),
            "geometry": selected_geometry.to_dict(),
            "selected": asdict(selected[ONE]), "candidates": records,
            "compute_precision_peaks": dict(context.compute_peaks),
            "capacity_screening": "declared_pools_only",
            "declared_pools": dict(context.pool_capacities),
            "read_role_bindings": {
                "matrix_operands": read_role(context.resource_names),
                "output_scales": read_role(context.resource_names, aiv=FIRST),
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


@dataclass(frozen=True)
class MatrixOperatorResult:
    """Public matrix-operator occurrence result.

    This is a comparison/contract result, not a serving latency or a measured
    answer.  Duration, serviced work and completed task count are occurrence
    deltas.  A current-occurrence buffer peak is only observable on a fresh
    Engine; on a shared Engine it is ``None`` with an explicit ``peak_basis``
    instead of reporting whole-engine history as this occurrence's peak.
    """

    duration_ms: float
    geometry: dict
    mapping: dict
    declared_work: dict
    serviced: dict
    peak_allocated: dict | None
    completed_tasks: int
    release_ms: float = 0.0
    ready_ms: float = 0.0
    peak_basis: str = "occurrence_on_fresh_engine"
    historical_peak_allocated: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "duration_ms": self.duration_ms,
            "geometry": self.geometry,
            "mapping": self.mapping,
            "declared_work": self.declared_work,
            "serviced": self.serviced,
            "peak_allocated": self.peak_allocated,
            "completed_tasks": self.completed_tasks,
            "release_ms": self.release_ms,
            "ready_ms": self.ready_ms,
            "peak_basis": self.peak_basis,
            "historical_peak_allocated": self.historical_peak_allocated,
        }


def _default_engine(context):
    """Build the ordinary Engine from the declared context, never a device."""
    from .engine import Engine, Resource
    names = set(context.resource_names.values())
    resources = []
    for resource, rate in context.rates.items():
        if resource not in names:
            continue
        units = (1 if resource == context.resource_names.get("hbm")
                 else physical_owner_units(context, resource))
        resources.append(Resource(resource, float(rate), units=units))
    # A prepared ``for_phase``/stage context binds cube FLOPs to peak_ms work
    # but may leave the cube rate in the enclosing execution.  Add that one
    # declared physical resource rather than inventing a compute peak here.
    cube = context.resource_names.get("cube", "cube")
    if context.work.get(cube) and cube not in {item.name for item in resources}:
        resources.append(Resource(cube, 1.0, unit="peak_ms",
                                  units=physical_owner_units(context, cube)))
    if not resources:
        raise ModelingError("matrix_operator_requires_declared_resource_rates")
    pools = {}
    pool_units = {}
    for logical, physical in context.pool_names.items():
        size = context.pool_capacities.get(logical, context.pool_capacities.get(physical))
        if size is None:
            raise ModelingError(f"matrix_operator_requires_pool_capacity:{logical}")
        pools[physical] = float(size)
        pool_units[physical] = int(
            context.pool_units.get(logical, context.pool_units.get(
                physical, context.core_count)))
    return Engine(resources, pools=pools, pool_units=pool_units)


def simulate_matrix_operator(context, *, recipe=None, engine=None, release_ms=0.0):
    """Run one isolated matrix operator through the common causal Engine.

    The entry accepts an already typed ``RecipeContext`` (semantics, tensor
    work, hardware rates/units, buffers and service identity), selects the
    declared mapping if needed, then returns finite work, geometry, buffers and
    completion.  It does not read model labels, observations, serving state or
    a target duration.  Delayed input readiness is an explicit release time and
    is excluded from the returned operator duration just like other public
    isolated-operator checks.

    A caller-provided Engine may be reused serially after previous work has
    completed.  Before submission the entry requires an idle active frontier:
    live/queued tasks, live programs or live allocations are rejected with
    ``matrix_operator_requires_idle_engine_frontier`` before any Engine
    mutation.  This keeps the occurrence serviced/completed delta from
    absorbing other ready work and rejects not-yet-started queued tasks rather
    than misclassifying them as fresh history.  Duration is measured from the
    operator's actual ready event.  Buffer peak is occurrence-only on a fresh
    Engine.  After serial reuse the historical peak would be wrong, so it is
    returned as ``None`` with
    ``peak_basis="unavailable_shared_engine_slice_boundary"``.  Concurrent
    combinations remain available through ``recipe.lower(context)`` plus
    ``program.start(engine, ...)``; this isolated entry does not implement a
    whole-Engine attribution subsystem.
    """
    from .engine import ExecutionError
    from .matrix_steady import SteadyMatmul
    recipe = recipe or SteadyMatmul()
    records = []
    previous = context.record_mapping

    def record_mapping(value):
        records.append(value)
        if previous is not None:
            previous(value)

    ctx = replace(context, record_mapping=record_mapping)
    engine = engine or _default_engine(ctx)
    frontier = engine.summary()
    if (frontier["live_tasks"] or frontier["live_programs"]
            or any(frontier["live_allocated"].values())):
        raise ExecutionError("matrix_operator_requires_idle_engine_frontier")
    name = f"matrix{next(_OPERATOR_SEQUENCE)}"
    before_serviced = dict(engine.serviced)
    before_allocated = dict(engine.allocated)
    before_completed = engine.completed_count
    before_now = engine.now
    fresh = (before_now == 0 and before_completed == 0
             and not any(before_serviced.values())
             and not any(before_allocated.values()))
    program = recipe.lower(ctx)
    ready = engine.submit(name + ":ready", release=release_ms)
    done = program.start(engine, name, ready)
    engine.advance(until=(done,))
    if not done.done:
        raise ExecutionError("matrix_operator_incomplete")
    if any(key.startswith(name + ":") for key in engine.allocations):
        raise ExecutionError("matrix_operator_left_live_buffers")
    if fresh and (engine.summary()["live_tasks"]
                  or engine.summary()["live_programs"]):
        raise ExecutionError("matrix_operator_left_live_tasks")
    mapping = records[-1] if records else {}
    geometry = mapping.get("geometry") if isinstance(mapping, dict) else None
    if not geometry:
        geometry = matrix_geometry(ctx, recipe).to_dict()
        mapping = {"basis": "explicit_matrix_recipe", "geometry": geometry,
                   "compute_precision_peaks": dict(ctx.compute_peaks)}
    ready_time = ready.time if ready.time is not None else float(release_ms)
    serviced = {
        resource: value - before_serviced.get(resource, 0.0)
        for resource, value in engine.serviced.items()
        if abs(value - before_serviced.get(resource, 0.0)) > 1e-12
    }
    historical_peak = {pool: value for pool, value in engine.peak_allocated.items()
                       if value}
    return MatrixOperatorResult(
        duration_ms=done.time - ready_time,
        geometry=geometry,
        mapping=mapping,
        declared_work=dict(ctx.work),
        serviced=serviced,
        peak_allocated=(dict(historical_peak) if fresh else None),
        completed_tasks=engine.completed_count - before_completed,
        release_ms=float(release_ms),
        ready_ms=ready_time,
        peak_basis=("occurrence_on_fresh_engine" if fresh else
                    "unavailable_shared_engine_slice_boundary"),
        historical_peak_allocated=historical_peak,
    )
