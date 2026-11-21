"""Matrix tile ownership and per-owner resource supply.

[Role] Derive matrix owner rectangles, tail totals and pinned pipe supply.
[Invariants] Owner aggregates retain the exact ragged output extents. A pinned
  stage draws only its assigned fraction of the declared physical pipe rate.
[Timeouts & Locks] Closed-form owner arithmetic and bounded explicit tile
  enumeration; no locks.
[Side-Effect Policy] Pure calculations over caller-supplied shapes and rates.
"""
from fractions import Fraction
import math

from mindie_sim.schema.common import ModelingError


_FRACTAL_ROWS = 16


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
                               *, fractal=_FRACTAL_ROWS):
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
