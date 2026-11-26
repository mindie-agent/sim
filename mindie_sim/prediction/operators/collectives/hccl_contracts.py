"""Explicit HCCL expansion and source phase contracts.

[Role] Keep selected executors, topology and wire traffic distinct from timing.
[Invariants] A source-compatible selection is not a captured dispatch claim.
  Payload means local input bytes (RS input includes all destination shards).
  Exchange matrices count each directed message once; local HBM is separate.
[Timeouts & Locks] Pure bounded metadata, O(ranks squared); no locks or I/O.
[Side-Effect Policy] No device contexts, measurements or runtime source reads.
"""
from dataclasses import asdict, dataclass

from mindie_sim.schema.common import ModelingError

SOURCE_REVISION = "c2d6790464654ccf58aca39448de4b65fdc7b184"
SOURCE = "https://github.com/hicann/hccl/tree/" + SOURCE_REVISION
HCCL_32BIT_ELEMENT_BYTES = 4
HCCL_64BIT_ELEMENT_BYTES = 8
LOCAL_REDUCE_HBM_TRANSFERS = 3  # Two operand reads and one result write.

_LEGACY_AICPU_ONESHOT_LIMIT_BYTES = 8 * 1024**2
_LEGACY_AICPU_CHUNK_LIMIT_BYTES = 32 * 1024**2
_LEGACY_AICPU_CHUNK_PAYLOAD_WEIGHT = 8
_LEGACY_AIV_MESH_RANK_LIMIT = 8
_LEGACY_AIV_ONESHOT_LIMIT_BYTES = 128 * 1024


def exact_positive_int(value, label):
    """Reject bool, non-finite, fractional and non-positive source integers."""
    import math
    if isinstance(value, bool):
        raise ModelingError(f"{label}_requires_exact_positive_integer:{value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ModelingError(f"{label}_requires_exact_positive_integer:{value!r}")
    if number <= 0:
        raise ModelingError(f"{label}_requires_exact_positive_integer:{value!r}")
    return number


def validate_rail_id(value, error_code="hccl_phase_rail_id_invalid"):
    """Validate an optional physical rail selector (non-negative int or non-empty str)."""
    if isinstance(value, bool):
        raise ModelingError(error_code)
    if isinstance(value, int):
        if value < 0:
            raise ModelingError(error_code)
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ModelingError(error_code)
        return stripped
    raise ModelingError(error_code)


@dataclass(frozen=True)
class HcclSelection:
    expansion: str
    collective: str
    template: str
    ranks: int
    payload_bytes: int
    dtype_bytes: int = 2
    topology: str = "MESH_1D"
    topology_levels: int = 1
    source_revision: str = SOURCE_REVISION
    selection_basis: str = "explicit_source_compatible_declaration"
    block_limit: int | None = None
    buffer_mode: str = "separate"
    # Explicit source hierarchy; never inferred from a hardware SKU.
    level0_size: int | None = None
    max_loop_bytes: int | None = None
    signal_bytes: int | None = None
    # Source channel endpoint owner, indexed [source rank][peer rank].
    # The diagonal has no channel; these are CCU dies, not NPU-rank packages.
    channel_die_map: tuple[tuple[int | None, ...], ...] = ()
    # Explicit ordered native repeat/segment byte sequence.  Required by
    # rs_mesh_chunk and z_axis_detour_*; never inferred from a shape or time.
    segments: tuple[int, ...] = ()
    # Explicit level-1 channel byte-share weights (channels = 1 + len).
    # Required by z_axis_detour_*; not a hardware bandwidth inventory.
    level1_weights: tuple[int, ...] = ()
    # Optional explicit physical rail selector per detour channel
    # (length = 1 + len(level1_weights)); defaults to (0, 1, ..., channels - 1).
    channel_rails: tuple[int | str, ...] = ()

    def to_dict(self):
        value = asdict(self)
        # Keep existing selections' content identity stable when the new
        # optional fields are unused.
        if not self.segments:
            value.pop("segments", None)
        if not self.level1_weights:
            value.pop("level1_weights", None)
        if not self.channel_rails:
            value.pop("channel_rails", None)
        return value


@dataclass(frozen=True)
class HcclPhase:
    name: str
    after: tuple[str, ...] = ()
    # Physical local work in native units. Optional hccl_* services are retained
    # in this ledger even when a hardware profile has not priced them yet.
    work: tuple[tuple[str, float], ...] = ()
    steps: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    # Full-group send matrix: [source][destination], with an empty diagonal.
    exchange: tuple[tuple[int, ...], ...] = ()
    owner_domain: tuple[int, ...] = ()
    control_counts: tuple[tuple[str, int], ...] = ()
    # Directed remote notify endpoints, distinct from payload and local waits.
    signals: tuple[tuple[int, int], ...] = ()
    # Native primitive view: send/write, read, or local-only.  The exchange
    # matrix stays a physical send matrix; this field is provenance.
    native_direction: str = "send"
    # Optional declared physical rail/channel selector for multi-rail exchanges.
    rail_id: int | str | None = None

    def __post_init__(self):
        if self.native_direction not in {"send", "read", "local"}:
            raise ModelingError("hccl_phase_native_direction_unknown")
        if self.rail_id is not None:
            normalized = validate_rail_id(self.rail_id, "hccl_phase_rail_id_invalid")
            if normalized != self.rail_id:
                object.__setattr__(self, "rail_id", normalized)


@dataclass(frozen=True)
class HcclPlan:
    selection: HcclSelection
    rank: int
    phases: tuple[HcclPhase, ...]
    thread_count: int
    scratch_bytes: int
    control_counts: tuple[tuple[str, int], ...] = ()
    unmodeled: tuple[str, ...] = ()


def select(raw, *, collective, ranks, payload_bytes):
    """Validate an explicit physical contract; never infer an engine by SKU."""
    allowed = {"expansion", "template", "dtype_bytes", "topology", "topology_levels",
               "source_revision", "block_limit", "buffer_mode", "level0_size",
               "max_loop_bytes", "signal_bytes", "channel_die_map",
               "segments", "level1_weights", "channel_rails"}
    if set(raw) - allowed or not {"expansion", "template"} <= set(raw):
        raise ModelingError("hccl_source_requires_explicit_expansion_and_template")
    raw = dict(raw)
    if 'channel_die_map' in raw:
        raw['channel_die_map'] = tuple(map(tuple, raw['channel_die_map']))
    for name in ('segments', 'level1_weights'):
        if name in raw:
            raw[name] = tuple(raw[name])
            raw[name] = tuple(exact_positive_int(value, 'hccl_' + name)
                              for value in raw[name])
    if 'channel_rails' in raw:
        raw_rails = raw['channel_rails']
        if not isinstance(raw_rails, (list, tuple)) or not raw_rails:
            raise ModelingError('hccl_source_channel_rails_requires_nonempty_sequence')
        raw['channel_rails'] = tuple(
            validate_rail_id(value, 'hccl_source_channel_rail_invalid')
            for value in raw_rails
        )
    result = HcclSelection(collective=collective, ranks=ranks, payload_bytes=payload_bytes, **raw)
    if (isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1
            or not isinstance(payload_bytes, int) or payload_bytes < 0
            or result.dtype_bytes not in {1, 2, HCCL_32BIT_ELEMENT_BYTES, HCCL_64BIT_ELEMENT_BYTES}
            or payload_bytes % result.dtype_bytes):
        raise ModelingError("hccl_source_requires_integer_rank_and_element_geometry")
    if result.expansion not in {"AICPU_TS", "AIV", "CCU_MS", "CCU_SCHED", "DPU"}:
        raise ModelingError("hccl_source_unknown_expansion")
    if result.collective not in {"allreduce", "allgather", "reduce_scatter", "alltoallv"}:
        raise ModelingError("hccl_source_unknown_collective")
    if result.source_revision != SOURCE_REVISION:
        raise ModelingError("hccl_source_revision_not_implemented")
    if result.block_limit is not None and (isinstance(result.block_limit, bool)
            or not isinstance(result.block_limit, int) or result.block_limit < 1):
        raise ModelingError("hccl_source_block_limit_requires_positive_integer")
    if result.buffer_mode not in {"separate", "symmetric", "remote_output"}:
        raise ModelingError("hccl_source_unknown_buffer_mode")
    for name in ("level0_size", "max_loop_bytes", "signal_bytes"):
        value = getattr(result, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or value < (0 if name == "signal_bytes" else 1)):
            raise ModelingError("hccl_source_invalid_" + name)
    if (result.template != 'parallel_mesh_nhr_route0'
            and any(getattr(result, name) is not None for name in ('level0_size', 'max_loop_bytes'))):
        raise ModelingError('hccl_source_hierarchy_options_require_route0_template')
    if result.template != 'mesh_2die_broadcast_v1' and result.channel_die_map:
        raise ModelingError('hccl_source_channel_die_map_requires_two_die_template')
    if (result.signal_bytes is not None
            and result.template not in {'parallel_mesh_nhr_route0', 'mesh_2die_broadcast_v1'}):
        raise ModelingError('hccl_source_signal_bytes_require_explicit_notify_template')
    segmented = {'rs_mesh_chunk', 'z_axis_detour_allgather',
                 'z_axis_detour_reduce_scatter'}
    weighted = {'z_axis_detour_allgather', 'z_axis_detour_reduce_scatter'}
    if result.segments and result.template not in segmented:
        raise ModelingError('hccl_source_segments_not_supported_by_template')
    if result.level1_weights and result.template not in weighted:
        raise ModelingError('hccl_source_level1_weights_not_supported_by_template')
    if result.template in segmented and not result.segments:
        raise ModelingError('hccl_source_requires_explicit_segments')
    if result.template in weighted and not result.level1_weights:
        raise ModelingError('hccl_source_requires_explicit_level1_weights')
    if result.channel_rails and result.template not in weighted:
        raise ModelingError('hccl_source_channel_rails_not_supported_by_template')
    if result.channel_rails and len(result.channel_rails) != 1 + len(result.level1_weights):
        raise ModelingError('hccl_z_detour_channel_rails_count_mismatch')
    return result


def engine_candidates(expansion):
    """Pinned SelectorEngine fallback priority, not successful runtime routing."""
    return {
        "CCU_MS": ("CCU_MS", "CCU_SCHED", "AICPU_TS", "HOSTCPU"),
        "CCU_SCHED": ("CCU_SCHED", "AICPU_TS", "HOSTCPU"),
        "AIV": ("AIV", "AICPU_TS", "HOSTCPU"),
        "AIV_ONLY": ("AIV",),
        "AICPU_TS": ("AICPU_TS", "HOSTCPU"),
        "DPU": ("HOSTCPU",),
    }[expansion]


def legacy_mesh_candidate(*, collective, expansion, payload_bytes, ranks,
                          special_reduce=False):
    """Narrow old-selector candidate; current Mesh1D uses the cost selector.

    This helper intentionally does not select a program. It omits communicator
    overrides, plugin routing, topology detours and runtime cost-table dispatch.
    """
    if collective == "allreduce" and expansion == "AICPU_TS":
        if payload_bytes <= _LEGACY_AICPU_ONESHOT_LIMIT_BYTES:
            return "mesh_oneshot"
        if special_reduce:
            return "mesh_twoshot"
        if (payload_bytes * _LEGACY_AICPU_CHUNK_PAYLOAD_WEIGHT / ranks**2
                > _LEGACY_AICPU_CHUNK_LIMIT_BYTES):
            return "mesh_chunk_twoshot"
        return "mesh_twoshot"
    if collective == "allreduce" and expansion == "AIV" and ranks <= _LEGACY_AIV_MESH_RANK_LIMIT:
        return "mesh_oneshot" if payload_bytes < _LEGACY_AIV_ONESHOT_LIMIT_BYTES else "mesh_twoshot"
    return None
