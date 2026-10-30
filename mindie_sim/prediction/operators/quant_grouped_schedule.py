"""Source-geometry schedule for the arch35 INT8 grouped matmul branch.

[Role]
  Expose the per-owner tile ordering declared by the pinned
  ``V310_GMM_QUANT_MIX`` / ``GMM_DEQUANT_VECTOR`` / no-transpose
  ``GQmmMixRegbaseKernel`` source.  This module is bounded arithmetic only: it
  does not implement Engine work, a matrix recipe, host tiling or dispatch.

[Invariants]
  * ``quant_block_sch``'s four-row window rotates its start owner across
    nonempty groups; it never restarts every group at owner zero.
  * Zero-row groups advance source tensor/bank offsets but consume no tile and
    do not rotate the start owner.  Original group indices are preserved.
  * Only the original last group (index ``groupNum - 1``) may use final-round
    tail splitting.  A trailing zero group therefore disables splitting for a
    preceding nonempty expert.
  * Per-owner iteration is lazy and in source group/round/owner order.  Actual
    empty split rectangles emit no tile, matching the source early return.
  * Tiles partition each nonempty group's logical output rectangle exactly.
    NZ weight offsets use ``[E,N/32,K/16,16,32]``; scale/output offsets are
    global byte offsets.  No timing, traffic rate, cache or dispatch is implied.

[Timeouts & Locks]
  Pure arithmetic.  Retained state is O(groups + owners); per-owner iteration
  is O(1) active state plus one tile record.  No locks or external waits.

[Side-Effect Policy]
  No I/O, device/runtime access, measurement read, global mutation or
  registration.  Callers import this module explicitly to opt in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "IMPLEMENTATION_ID",
    "SOURCE_REVISION",
    "QuantGroupedTiling",
    "SourceMatrixTile",
    "QuantGroupedSchedule",
    "plan_quant_grouped",
]

IMPLEMENTATION_ID = "arch35_gqmm_mix_nz"
SOURCE_REVISION = "a3925fd49cd256ad478d2faf983bf2961b8ae2c8"
SOURCE_ROOT = "/Users/maoxx241/code/refs/ops-transformer"

# sha256 and byte length of the exact pinned files read for this adapter.
_SOURCE_FILES = (
    {
        "path": "gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp",
        "sha256": "320af4dac634a8c9398494cd484150801b67d3280861fda0dd945db1edef0175",
        "bytes": 35169,
    },
    {
        "path": "gmm/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/quant_block_sch.h",
        "sha256": "55196716e988e3d0b502e21795e0e82d6095a63a7d7d4a8a082ac442cc3f695e",
        "bytes": 17987,
    },
    {
        "path": "gmm/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h",
        "sha256": "a2bd4d5b5c296b852c18aca0e09f0221567f530d067d14bd1d5d0304072b8465",
        "bytes": 50576,
    },
    {
        "path": "gmm/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/quant_utils.h",
        "sha256": "3d58d63fb0311012441259890454e0b331ea02a998e9414305bb92a81eeae952",
        "bytes": 7659,
    },
    {
        "path": "gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp",
        "sha256": "68714d9bc213d807cae10e1392693a08958ef39b20ac6a10a748167453680ad6",
        "bytes": 89866,
    },
)

_WINDOW_LEN = 4
_NZ_N0 = 32
_NZ_K0 = 16
_CUBE_BLOCK = 16
# The pinned no-transpose branch uses the 128-element inner-axis split
# granule, including NZ weights.  NZ storage blocks alone do not set scheduling.
_TAIL_M_MIN = _CUBE_BLOCK
_TAIL_N_MIN = 128

_X_ELEMENT_BYTES = 1   # INT8 activation, ND
_W_ELEMENT_BYTES = 1   # INT8 weight, FRACTAL_NZ
_Y_ELEMENT_BYTES = 2   # BF16 output
_SCALE_BYTES = 4       # FP32 row/per-channel scale


def _require_int(value: object, reason: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{reason}:{value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{reason}:{value!r}")
    return value


def _require_multiple(value: int, multiple: int, reason: str) -> int:
    if value % multiple:
        raise ValueError(f"{reason}:{value!r}")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align(value: int, align: int) -> int:
    return _ceil_div(value, align) * align


@dataclass(frozen=True, slots=True)
class QuantGroupedTiling:
    """Caller-declared source geometry.  Host tiling is not inferred."""

    base_m: int
    base_n: int
    base_k: int
    used_cores: int

    def __post_init__(self) -> None:
        base_m = _require_int(self.base_m, "base_m_must_be_positive_int", minimum=1)
        base_n = _require_int(self.base_n, "base_n_must_be_positive_int", minimum=1)
        base_k = _require_int(self.base_k, "base_k_must_be_positive_int", minimum=1)
        _require_int(self.used_cores, "used_cores_must_be_positive_int", minimum=1)
        _require_multiple(base_m, 16, "base_m_must_be_multiple_of_16")
        _require_multiple(base_n, 32, "base_n_must_be_multiple_of_32")
        _require_multiple(base_k, 32, "base_k_must_be_multiple_of_32")


@dataclass(frozen=True, slots=True)
class SourceMatrixTile:
    """One source-ordered rectangle.

    ``m_offset``/``n_offset`` are element offsets within the original group.
    ``m``/``n`` are actual tile extents and ``k`` is the full K.  The five
    ``*_offset`` fields are global BYTE offsets into the whole INT8 X, INT8 NZ
    W, BF16 Y, FP32 row-scale and FP32 column-scale backing tensors.
    """

    group: int
    ordinal: int
    m_offset: int
    n_offset: int
    m: int
    n: int
    k: int
    x_offset: int
    weight_offset: int
    output_offset: int
    row_scale_offset: int
    column_scale_offset: int


@dataclass(frozen=True, slots=True)
class _GroupSchedule:
    group: int
    rows: int
    m_base_tail: int
    n_base_tail: int
    m_cnt: int
    n_cnt: int
    total_cnt: int
    m_core_num: int
    main_row: int
    m_tail_core_num: int
    start: int
    end: int
    rounds: int
    active_count: int
    m_tile: int
    n_tile: int
    split: bool
    split_end: int
    x_base: int
    weight_base: int
    output_base: int
    row_scale_base: int
    column_scale_base: int

    @property
    def effective_end(self) -> int:
        return self.split_end if self.split else self.end


class QuantGroupedSchedule:
    """Retained O(groups) geometry; no whole-model tile collection."""

    def __init__(
        self,
        expert_rows: tuple[int, ...],
        n: int,
        k: int,
        tiling: QuantGroupedTiling,
    ) -> None:
        rows = self._validate_rows(expert_rows)
        if not isinstance(tiling, QuantGroupedTiling):
            raise ValueError(f"tiling_must_be_QuantGroupedTiling:{tiling!r}")
        n = _require_int(n, "n_must_be_positive_int", minimum=1)
        k = _require_int(k, "k_must_be_positive_int", minimum=1)
        _require_multiple(n, 32, "n_must_be_multiple_of_32")
        _require_multiple(k, 16, "k_must_be_multiple_of_16")

        self.expert_rows = rows
        self.n = n
        self.k = k
        self.tiling = tiling
        self.total_m = sum(rows)
        self._group_count = len(rows)
        self._groups = self._build_groups()
        self.owner_tile_counts = tuple(
            self._owner_tile_count(owner) for owner in range(tiling.used_cores)
        )

    @staticmethod
    def _validate_rows(expert_rows: object) -> tuple[int, ...]:
        if isinstance(expert_rows, (str, bytes)) or not isinstance(
            expert_rows, (list, tuple)
        ):
            raise ValueError(f"expert_rows_must_be_sequence:{expert_rows!r}")
        rows = tuple(
            _require_int(value, "expert_rows_must_be_nonnegative_int", minimum=0)
            for value in expert_rows
        )
        if not rows:
            raise ValueError("expert_rows_must_not_be_empty")
        return rows

    def _build_groups(self) -> tuple[_GroupSchedule, ...]:
        n, k = self.n, self.k
        tiling = self.tiling
        used = tiling.used_cores
        previous_end = used - 1  # QuantASWBlockSch::Init<true>, endBlockIdx_
        x_base = weight_base = output_base = row_scale_base = column_scale_base = 0
        groups: list[_GroupSchedule] = []

        for group, rows in enumerate(self.expert_rows):
            if rows > 0:
                m_cnt = _ceil_div(rows, tiling.base_m)
                n_cnt = _ceil_div(n, tiling.base_n)
                total_cnt = m_cnt * n_cnt
                m_core_num = min(_WINDOW_LEN, m_cnt)
                main_row = m_cnt // m_core_num - 1
                m_tail_core_num = m_cnt - m_core_num * main_row
                start = 0 if previous_end == used - 1 else previous_end + 1
                end = (total_cnt + start - 1) % used
                rounds = _ceil_div(total_cnt, used)
                active_count = used if total_cnt % used == 0 else total_cnt % used
                m_base_tail = rows - (m_cnt - 1) * tiling.base_m
                n_base_tail = n - (n_cnt - 1) * tiling.base_n

                split = False
                m_tile = n_tile = 1
                split_end = end
                # IsLastGroupAndNeedSplit: literal original last group only.
                if group == self._group_count - 1 and end + 1 <= used // 2:
                    tail_cnt = end + 1
                    remain_tile = used // tail_cnt
                    if remain_tile > 1:
                        m_tile = min(
                            _ceil_div(m_base_tail, _TAIL_M_MIN), remain_tile
                        )
                        n_tile = min(
                            _ceil_div(n_base_tail, _TAIL_N_MIN), remain_tile
                        )
                        while m_tile * n_tile > remain_tile:
                            if m_tile >= n_tile:
                                m_tile -= 1
                            else:
                                n_tile -= 1
                        if m_tile * n_tile > 1:
                            if start != 0:
                                # UpdateBasicIndex<true> computes
                                # blockIdx/(mTailTile*nTailTile) and then
                                # subtracts startBlockIdx_.  For a rotated
                                # start, some small groups repeat rectangles
                                # after unsigned wrap.  Keep this source domain
                                # unmodeled instead of correcting its ordering.
                                raise ValueError(
                                    "unsupported_last_group_split_with_rotated_start:"
                                    f"group={group!r},start={start!r}"
                                )
                            split = True
                            split_end = m_tile * n_tile * (end + 1) - 1
                groups.append(
                    _GroupSchedule(
                        group=group,
                        rows=rows,
                        m_base_tail=m_base_tail,
                        n_base_tail=n_base_tail,
                        m_cnt=m_cnt,
                        n_cnt=n_cnt,
                        total_cnt=total_cnt,
                        m_core_num=m_core_num,
                        main_row=main_row,
                        m_tail_core_num=m_tail_core_num,
                        start=start,
                        end=end,
                        rounds=rounds,
                        active_count=active_count,
                        m_tile=m_tile,
                        n_tile=n_tile,
                        split=split,
                        split_end=split_end,
                        x_base=x_base,
                        weight_base=weight_base,
                        output_base=output_base,
                        row_scale_base=row_scale_base,
                        column_scale_base=column_scale_base,
                    )
                )
                previous_end = groups[-1].effective_end

            # UpdateGroupOffset advances even for a zero-row group; the next
            # group consumes the prior group's m and the current bank width.
            x_base += rows * k * _X_ELEMENT_BYTES
            weight_base += n * k * _W_ELEMENT_BYTES
            output_base += rows * n * _Y_ELEMENT_BYTES
            row_scale_base += rows * _SCALE_BYTES
            column_scale_base += n * _SCALE_BYTES

        return tuple(groups)

    def _group_rounds(self, group: _GroupSchedule, owner: int) -> int:
        rounds = group.rounds
        if (owner - group.start) % self.tiling.used_cores >= group.active_count:
            rounds -= 1
        if group.split and group.end < owner <= group.split_end:
            rounds += 1
        return rounds

    def _base_block_index(self, group: _GroupSchedule, owner: int, round_idx: int,
                          is_last_group_round: bool) -> tuple[int, int]:
        used = self.tiling.used_cores
        split_product = group.m_tile * group.n_tile
        new_block = owner // split_product if is_last_group_round else owner
        index = new_block + round_idx * used
        if owner < group.start:
            index += used - group.start
        else:
            index -= group.start
        if index >= group.total_cnt:
            raise ValueError(
                "source_schedule_index_out_of_range:"
                f"group={group.group!r},owner={owner!r},round={round_idx!r}"
            )

        row_idx = index // group.n_cnt // group.m_core_num
        if row_idx < group.main_row:
            m_index = row_idx * group.m_core_num + index % group.m_core_num
            n_index = (index // group.m_core_num) % group.n_cnt
        else:
            row_idx = group.main_row
            tail_index = index - group.main_row * group.m_core_num * group.n_cnt
            m_index = (
                group.main_row * group.m_core_num
                + tail_index % group.m_tail_core_num
            )
            n_index = (tail_index // group.m_tail_core_num) % group.n_cnt
        if row_idx & 1:
            n_index = group.n_cnt - 1 - n_index
        return m_index, n_index

    def _rectangle(self, group: _GroupSchedule, owner: int, round_idx: int,
                   rounds: int) -> tuple[int, int, int, int] | None:
        is_last = (
            group.group == self._group_count - 1
            and round_idx == rounds - 1
            and owner <= group.effective_end
        )
        m_index, n_index = self._base_block_index(group, owner, round_idx, is_last)
        single_m = (self.tiling.base_m if m_index != group.m_cnt - 1
                    else group.m_base_tail)
        single_n = (self.tiling.base_n if n_index != group.n_cnt - 1
                    else group.n_base_tail)
        m_offset = m_index * self.tiling.base_m
        n_offset = n_index * self.tiling.base_n
        if is_last and group.split:
            m_split = _ceil_div(single_m, group.m_tile)
            n_split = _align(_ceil_div(single_n, group.n_tile), _TAIL_N_MIN)
            split_product = group.m_tile * group.n_tile
            m_split_idx = (owner % split_product) % group.m_tile
            n_split_idx = (owner % split_product) // group.m_tile
            m_split_offset = m_split_idx * m_split
            n_split_offset = n_split_idx * n_split
            if m_split_offset >= single_m or n_split_offset >= single_n:
                return None
            single_m = min(m_split, single_m - m_split_offset)
            single_n = min(n_split, single_n - n_split_offset)
            m_offset += m_split_offset
            n_offset += n_split_offset
        return m_offset, n_offset, single_m, single_n

    def _owner_tile_count(self, owner: int) -> int:
        count = 0
        for group in self._groups:
            rounds = self._group_rounds(group, owner)
            count += rounds
            # Only the final split can return before producing its rectangle.
            # Count that endpoint without walking the owner's preceding tiles.
            if group.split and rounds and self._rectangle(
                    group, owner, rounds - 1, rounds) is None:
                count -= 1
        return count

    def iter_owner_tiles(self, owner: int) -> Iterator[SourceMatrixTile]:
        """Yield one owner's tiles in source group/round/block order."""
        owner = _require_int(owner, "owner_must_be_nonnegative_int", minimum=0)
        if owner >= self.tiling.used_cores:
            raise ValueError(f"owner_out_of_range:{owner!r}")

        n, k = self.n, self.k
        ordinal = 0
        for group in self._groups:
            rounds = self._group_rounds(group, owner)
            if rounds <= 0:
                continue
            for round_idx in range(rounds):
                rectangle = self._rectangle(group, owner, round_idx, rounds)
                if rectangle is None:
                    break
                m_offset, n_offset, single_m, single_n = rectangle

                yield SourceMatrixTile(
                    group=group.group,
                    ordinal=ordinal,
                    m_offset=m_offset,
                    n_offset=n_offset,
                    m=single_m,
                    n=single_n,
                    k=k,
                    x_offset=group.x_base + m_offset * k * _X_ELEMENT_BYTES,
                    weight_offset=(
                        group.weight_base + n_offset * k * _W_ELEMENT_BYTES
                    ),
                    output_offset=(
                        group.output_base
                        + (m_offset * n + n_offset) * _Y_ELEMENT_BYTES
                    ),
                    row_scale_offset=(
                        group.row_scale_base + m_offset * _SCALE_BYTES
                    ),
                    column_scale_offset=(
                        group.column_scale_base + n_offset * _SCALE_BYTES
                    ),
                )
                ordinal += 1

    def iter_weight_spans(
        self, tile: SourceMatrixTile, k_offset: int = 0, k_size: int | None = None
    ) -> Iterator[tuple[int, int]]:
        """Yield physical whole-NZ-block weight ranges for ``tile``.

        The returned byte ranges are storage identities only, never DMA
        requests, cache lines or traffic.  A K/16-aligned subpanel has one
        interval per 32-wide N block; full-K rectangles coalesce contiguously.
        """
        if not isinstance(tile, SourceMatrixTile):
            raise ValueError(f"tile_must_be_SourceMatrixTile:{tile!r}")
        if tile.k != self.k:
            raise ValueError(f"tile_k_mismatch:{tile.k!r}")
        k_offset = _require_int(k_offset, "k_offset_must_be_nonnegative_int", minimum=0)
        if k_offset % _NZ_K0:
            raise ValueError(f"unsupported_sub_block_k_offset:{k_offset!r}")
        if k_size is None:
            k_size = self.k - k_offset
        k_size = _require_int(k_size, "k_size_must_be_positive_int", minimum=1)
        if k_size % _NZ_K0 or k_offset + k_size > self.k:
            raise ValueError(f"unsupported_sub_block_k_size:{k_size!r}")
        if tile.n_offset % _NZ_N0 or tile.n % _NZ_N0:
            raise ValueError(
                "weight_spans_require_whole_nz_n_blocks:"
                f"n_offset={tile.n_offset!r},n={tile.n!r}"
            )
        expected_offset = tile.group * self.n * self.k + tile.n_offset * self.k
        if tile.weight_offset != expected_offset:
            raise ValueError(
                f"tile_weight_offset_not_from_schedule:{tile.weight_offset!r}"
            )

        if k_offset == 0 and k_size == self.k:
            yield tile.weight_offset, tile.n * tile.k
        else:
            for n_block in range(tile.n // _NZ_N0):
                yield (
                    tile.weight_offset + n_block * self.k * _NZ_N0
                    + k_offset * _NZ_N0,
                    k_size * _NZ_N0,
                )

    def summary(self) -> dict[str, object]:
        """Compact JSON-serializable declaration; never enumerates tiles."""
        nonempty = [group for group in self._groups]
        rows = self.expert_rows
        trailing_zeros = 0
        for value in reversed(rows):
            if value != 0:
                break
            trailing_zeros += 1
        last = nonempty[-1] if nonempty else None
        return {
            "implementation": IMPLEMENTATION_ID,
            "source": {
                "root": SOURCE_ROOT,
                "revision": SOURCE_REVISION,
                "files": tuple(dict(item) for item in _SOURCE_FILES),
                "dispatch": (
                    "V310_GMM_QUANT_MIX, GMM_DEQUANT_VECTOR, no transpose, "
                    "GQmmMixRegbaseKernel; explicit opt-in import"
                ),
                "host_tiling_recreated": False,
            },
            "support_domain": {
                "activation": "int8_nd",
                "weight": "int8_fractal_nz",
                "scales": "fp32_row_and_per_channel",
                "output": "bf16",
                "transpose": False,
                "bias": False,
                "activation_epilogue": False,
                "group_type": "split_m",
                "group_list_type": 1,
                "single_x_weight_output": True,
                "expert_rows": "explicit_nonnegative_integer_counts_all_groups",
                "n_multiple": 32,
                "k_multiple": 16,
            },
            "tiling": {
                "base_m": self.tiling.base_m,
                "base_n": self.tiling.base_n,
                "base_k": self.tiling.base_k,
                "used_cores": self.tiling.used_cores,
                "base_k_role": "declared_input_only_schedule_order_independent",
            },
            "geometry": {
                "groups": self._group_count,
                "nonempty_groups": len(nonempty),
                "total_m": self.total_m,
                "n": self.n,
                "k": self.k,
                "rows_min": min(rows),
                "rows_max": max(rows),
                "trailing_empty_groups": trailing_zeros,
                "last_group_splits": bool(last.split) if last else False,
            },
            "schedule": {
                "window_rows": _WINDOW_LEN,
                "owner_rotation": "persists_across_nonempty_groups",
                "zero_group_rotates": False,
                "tail_split_group": "original_last_group_only",
                "tail_min_m": _TAIL_M_MIN,
                "tail_min_n": _TAIL_N_MIN,
                "tail_n_alignment": _TAIL_N_MIN,
                "owner_tile_counts": self.owner_tile_counts,
                "nz_storage_shape": "[E,N/32,K/16,16,32]",
                "tile_ordinal": "per_owner_global_monotonic_zero_based",
            },
            "units": {
                "m_offset": "elements_within_group",
                "n_offset": "elements_within_group",
                "m_n_k": "elements",
                "x_weight_offset": "global_bytes",
                "output_offset": "global_bytes",
                "row_column_scale_offset": "global_bytes",
            },
            "omitted": (
                "no latency, resource rate, floor, multiplier or fit",
                "no cache/L2/full-bank residency or active-traffic inference",
                "no scalar group-loop, sync, startup or replay cost",
                "no MatmulImpl internal transaction, fractal padding or DMA claim",
                "no host tiling recreation or inferred base_m/base_n/base_k",
                "no loaded historical binary/tiling identity",
                "no native dispatch or default frontend selection change",
                "weight spans are storage identity only, not DMA requests/traffic",
            ),
        }


def plan_quant_grouped(
    expert_rows: tuple[int, ...],
    n: int,
    k: int,
    tiling: QuantGroupedTiling,
) -> QuantGroupedSchedule:
    """Build the source-ordered schedule for one explicit grouped invocation."""
    return QuantGroupedSchedule(expert_rows, n, k, tiling)
