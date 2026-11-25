"""Native aclnnRingAttentionUpdateV2 (Class D AIV LSE attention merge).

Upstream-ABI: aclnnRingAttentionUpdateV2

[Role] Lower rank-local TND/TND ring attention LSE merge (6 inputs: prev/cur
  attn_out + prev/cur max/sum width-8 FP32 stats -> 3 outputs: merged attn_out
  + merged max/sum) on Arch22 and Arch35 AIV cores.
[Invariants] OperatorPlan.work matches lowered stage work identically. Output
  write-allocation and reused stats stay in L2 only when total working set fits
  die L2 capacity; cold input reads and spilled sets charge HBM. Zero hardcoded
  setup constants or empirical turnaround binders.
[Timeouts & Locks] O(active_aivs) parallel AIV lowering; zero locks or I/O.
[Side-Effect Policy] Pure immutable plans and virtual Engine programs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from typing import Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import Geometry, OperatorPlan, Work
from ...ownership import read_role
from ...program import KernelProgram, ParallelProgram, RecipeContext, Stage
from ...results import FusedPipelineSpec, OpOccurrence, PipelinePhase, WorkItem
from ...runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from ...runtime.memory_physics import (
    AddressRun,
    AddressRunPlan,
    HbmBusSpec,
    MemoryAccessPattern,
    ReuseKey,
    VisibilityContract,
)
from ..attention.common import ARCH35, BF16_BYTES, FP32_BYTES
from ..contracts import OperatorPlanningContext

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

RING_UPDATE_ABI = "aclnnRingAttentionUpdateV2"
RING_UPDATE_RECIPE = "ring_attention_update_v2"
STATS_WIDTH = 8
INPUT_ATTN_TENSORS = 2
INPUT_STATS_TENSORS = 4
OUTPUT_STATS_TENSORS = 2
ATTN_ELEM_OPS = 6  # 2 FP32 upcasts + 2 muls + 1 add + 1 BF16 downcast (ring_attention_update_tnd.h:298)
STATS_LANE_OPS = 8  # max + 2*(sub+exp+mul) + add + 2*div (ring_attention_update_tnd.h:210)
_RING_UPDATE_SUPPORTED_DTYPES = frozenset({"fp16", "bf16", "fp32"})
_ARCH22_TND_HEAD_DIM_ALIGN = 64

RING_UPDATE_SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/ring_attention_update/op_kernel/ring_attention_update_tnd.h",
        source_sha256="6dfdf1fc08647c1633a450381f56c0eac16f2e7148d09fe46161cc063501bb04",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/ring_attention_update/op_host/ring_attention_update_tiling.cpp",
        source_sha256="8014ef958465506173aececc58d5f0f5dc1bd4b2201219a4b1eeda68d765d512",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/ring_attention_update/op_host/ring_attention_update_tiling_arch35.cpp",
        source_sha256="b01e22be9695fc97df35650abd7f061ad7580cdef275a33d37c7a340e477e943",
    ),
)
SOURCES: tuple[SourceRef, ...] = RING_UPDATE_SOURCES


@dataclass(frozen=True, slots=True)
class RingUpdateContract:
    tokens: int
    num_heads: int
    head_dim: int
    dtype: str = "bf16"
    abi: str = RING_UPDATE_ABI

    def __post_init__(self) -> None:
        for name, val in (("tokens", self.tokens), ("num_heads", self.num_heads), ("head_dim", self.head_dim)):
            if type(val) is not int or val < 0:
                raise ModelingError(f"ring_update_invalid_{name}:{val!r}")
        dtype_norm = str(self.dtype or "").strip().lower()
        if dtype_norm not in _RING_UPDATE_SUPPORTED_DTYPES:
            raise ModelingError(f"ring_update_unsupported_dtype:{self.dtype!r}")
        object.__setattr__(self, "dtype", dtype_norm)


def _l2_capacity_bytes(ctx: OperatorPlanningContext) -> int:
    hierarchy = getattr(ctx.geometry.physical_mechanisms, "memory_hierarchy", None)
    return int(getattr(hierarchy, "effective_l2_bytes", ctx.geometry.buffers.get("l2", 0)) or 0)


def _apportion_bytes(total: int, counts: Sequence[int]) -> tuple[int, ...]:
    """Distribute exact integer bytes across active AIVs without rounding drift."""
    total_units = sum(counts)
    if total <= 0 or total_units <= 0:
        return (0,) * len(counts)
    shares = [(total * count) // total_units for count in counts]
    remainder = total - sum(shares)
    for idx in range(remainder):
        shares[idx] += 1
    return tuple(shares)


def _apportion_hbm_l2(
    total_hbm: int, total_l2: int, counts: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Distribute HBM and L2 bytes conserving aggregate totals and per-AIV sums."""
    aiv_total = _apportion_bytes(total_hbm + total_l2, counts)
    aiv_l2 = list(_apportion_bytes(total_l2, counts))
    overflow = 0
    for idx, (tot, l2_val) in enumerate(zip(aiv_total, aiv_l2)):
        if l2_val > tot:
            overflow += l2_val - tot
            aiv_l2[idx] = tot
    for idx in range(len(aiv_l2)):
        if overflow <= 0:
            break
        add = min(aiv_total[idx] - aiv_l2[idx], overflow)
        aiv_l2[idx] += add
        overflow -= add
    aiv_hbm = tuple(tot - l2_val for tot, l2_val in zip(aiv_total, aiv_l2))
    return aiv_hbm, tuple(aiv_l2)


@dataclass(frozen=True, slots=True)
class _RingUpdateProgram:
    aiv_counts: tuple[int, ...]
    aiv_total: int
    aiv_read: str
    per_unit_read: int
    per_unit_write: int
    aiv_hbm: tuple[int, ...]
    aiv_l2: tuple[int, ...]
    per_unit_vec: int

    def __call__(self, ctx: RecipeContext) -> KernelProgram | ParallelProgram:
        branches = []
        for aiv_id, units in enumerate(self.aiv_counts):
            if units <= 0:
                continue
            r_b = units * self.per_unit_read
            w_b = units * self.per_unit_write
            h_b = self.aiv_hbm[aiv_id]
            l_b = self.aiv_l2[aiv_id]
            v_ops = units * self.per_unit_vec
            w = Work(resources={
                self.aiv_read: r_b,
                "mte3": w_b,
                "vector": v_ops,
                **({"hbm": h_b} if h_b else {}),
                **({"l2": l_b} if l_b else {}),
            })
            place = ctx.owned(aiv_id, self.aiv_total, {self.aiv_read, "mte3", "vector"})
            branches.append(KernelProgram((Stage("merge", ctx.bind(w), placement=place, coupled=True),)))
        return ParallelProgram(tuple(branches)) if branches else KernelProgram(())


def build_ring_update(op: OpSpec, geometry: Geometry | OperatorPlanningContext) -> OperatorPlan:
    """Build Class D AIV plan for aclnnRingAttentionUpdateV2."""
    if isinstance(geometry, OperatorPlanningContext):
        ctx = geometry
    else:
        ctx = OperatorPlanningContext(
            OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
            geometry, geometry.datapath, stream_owners=geometry.stream_owners,
        )
    raw = dict(op.metadata).get("ring_update_contract")
    if isinstance(raw, RingUpdateContract):
        contract = raw
    elif isinstance(raw, dict):
        contract = RingUpdateContract(**raw)
    else:
        shape = op.local_shape or op.semantic_shape
        contract = RingUpdateContract(int(shape[0]), int(shape[1]), int(shape[2]), dtype=op.dtype or "bf16")
    is_arch35 = ctx.compute_arch.arch_id == ARCH35
    if not is_arch35 and contract.head_dim % _ARCH22_TND_HEAD_DIM_ALIGN != 0:
        raise ModelingError(f"ring_update_arch22_tnd_head_dim_unaligned:{contract.head_dim}")
    aiv_total = max(1, int(ctx.datapath.aiv_count))
    aiv_read = read_role(ctx.geometry.resource_units, aiv=0)
    elem_b = BF16_BYTES if contract.dtype in ("bf16", "fp16") else FP32_BYTES
    attn_read = INPUT_ATTN_TENSORS * contract.head_dim * elem_b
    stats_read = INPUT_STATS_TENSORS * STATS_WIDTH * FP32_BYTES
    attn_write = contract.head_dim * elem_b
    stats_write = OUTPUT_STATS_TENSORS * STATS_WIDTH * FP32_BYTES
    read_per_th = attn_read + stats_read
    write_per_th = attn_write + stats_write
    vec_per_th = contract.head_dim * ATTN_ELEM_OPS + STATS_WIDTH * STATS_LANE_OPS
    total_th = contract.tokens * contract.num_heads
    has_l2_rate = "l2" in ctx.geometry.rates
    die = int(ctx.geometry.rank or 0)
    l2_cap = _l2_capacity_bytes(ctx)
    total_read = total_th * read_per_th
    total_write = total_th * write_per_th
    view = ctx.admission_memory_view
    if view is None:
        view = AdmissionMemoryView({die: l2_cap}, arch_id=str(ctx.compute_arch.arch_id))
    hbm_bus = HbmBusSpec(die, 1, 1, 0)
    read_hbm, write_hbm, l2_bytes = total_read, total_write, 0
    if total_read > 0:
        r_pat = MemoryAccessPattern(
            "ring_update_in", 0, die, 0, "read", aiv_read, total_read,
            AddressRunPlan((AddressRun(0, 1, total_read, total_read),)),
            reuse_key=ReuseKey(die, "ring_update_in", 0, reuse_stream="ring_update"),
        )
        r_led = bind_memory_access(
            r_pat, view, hbm_bus=hbm_bus, copy_hit_bytes_per_ms=1, copy_miss_bytes_per_ms=1,
            l2_bytes_per_ms=1, pipe_row_id=aiv_read, l2_row_id="l2", commit=False,
        ).ledger
        read_hbm = r_led.dram_sector_bytes
        l2_bytes += r_led.l2_hit_bytes
    if total_write > 0:
        w_pat = MemoryAccessPattern(
            "ring_update_out", 0, die, 0, "write", "mte3", total_write,
            AddressRunPlan((AddressRun(0, 1, total_write, total_write),)),
            reuse_key=ReuseKey(die, "ring_update_out", 0, reuse_stream="ring_update"),
            visibility=VisibilityContract(live_out=True, private_workspace=False),
        )
        w_led = bind_memory_access(
            w_pat, view, hbm_bus=hbm_bus, copy_hit_bytes_per_ms=1, copy_miss_bytes_per_ms=1,
            l2_bytes_per_ms=1, pipe_row_id="mte3", l2_row_id="l2", commit=False,
        ).ledger
        write_hbm = w_led.dram_sector_bytes
        l2_bytes += w_led.l2_hit_bytes
    l2_absorbed = l2_bytes > 0
    total_hbm = int(read_hbm + write_hbm)
    total_l2 = int(l2_bytes) if has_l2_rate else 0
    if is_arch35:
        total_units = total_th
        mult = 1
    else:
        total_units = contract.tokens
        mult = contract.num_heads
    rows_per_aiv = ceil(total_units / aiv_total) if total_units > 0 else 0
    active = ceil(total_units / rows_per_aiv) if rows_per_aiv > 0 else 0
    counts: list[int] = []
    rem = total_units
    for _ in range(active):
        take = min(rows_per_aiv, rem)
        counts.append(take)
        rem -= take
    aiv_hbm, aiv_l2 = _apportion_hbm_l2(total_hbm, total_l2, counts)
    prog = _RingUpdateProgram(
        tuple(counts), aiv_total, aiv_read,
        read_per_th * mult, write_per_th * mult, aiv_hbm, aiv_l2, vec_per_th * mult,
    )
    total_work = Work(resources={
        aiv_read: total_th * read_per_th,
        "mte3": total_th * write_per_th,
        "vector": total_th * vec_per_th,
        **({"hbm": total_hbm} if total_hbm else {}),
        **({"l2": total_l2} if total_l2 else {}),
    })
    soc_arch = "A5" if is_arch35 else "A3"
    abi_available = default_abi_registry().is_shipping_available(
        contract.abi,
        soc_arch,
        dtype_tuple=(contract.dtype, "fp32", "fp32", contract.dtype, "fp32", "fp32"),
    )
    items = tuple(WorkItem(r, "op" if r == "vector" else "bytes", n) for r, n in total_work.resources.items() if n)
    return OperatorPlan(
        work=total_work,
        program=prog,
        metadata={
            "implementation": RING_UPDATE_RECIPE,
            "abi": contract.abi,
            "abi_soc_arch": soc_arch,
            "abi_available_on_device": abi_available,
            "execution_class": "aiv_only",
            "arch_id": str(ctx.compute_arch.arch_id),
            "active_aivs": active,
            "l2_absorbed": l2_absorbed,
            "hbm_bytes": total_hbm,
            "l2_hit_bytes": int(l2_bytes),
            "total_transfer_bytes": int(total_read + total_write),
        },
        pipeline_spec=FusedPipelineSpec("attention_merge", RING_UPDATE_RECIPE,
                                        (PipelinePhase("merge", (), items),), RING_UPDATE_SOURCES),
    )


def make_ring_update_op(
    tokens: int, num_heads: int, head_dim: int, *, op_id: str = "ring_attention_update",
    deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = ("r0",), dtype: str = "bf16",
) -> OpSpec:
    """Create a governed OpSpec for aclnnRingAttentionUpdateV2."""
    contract = RingUpdateContract(tokens, num_heads, head_dim, dtype=dtype)
    shape = (tokens, num_heads, head_dim)
    elem_b = BF16_BYTES if contract.dtype in ("bf16", "fp16") else FP32_BYTES
    total_th = contract.tokens * contract.num_heads
    total_read = float(total_th * (INPUT_ATTN_TENSORS * contract.head_dim * elem_b + INPUT_STATS_TENSORS * STATS_WIDTH * FP32_BYTES))
    total_write = float(total_th * (contract.head_dim * elem_b + OUTPUT_STATS_TENSORS * STATS_WIDTH * FP32_BYTES))
    total_vec = float(total_th * (contract.head_dim * ATTN_ELEM_OPS + STATS_WIDTH * STATS_LANE_OPS))
    return OpSpec(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        family="attention_merge", phase="prefill", recipe=RING_UPDATE_RECIPE,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        vector_ops=total_vec, mte2_bytes=total_read, mte3_bytes=total_write,
        dtype=dtype, layout="TND", source_boundary="kernel",
        metadata=(
            ("ring_update_contract", asdict(contract)),
            ("native_kind", RING_UPDATE_ABI),
            ("ub_compute_dtype", "fp32"),
            ("ub_working_bytes_per_elem", FP32_BYTES),
        ),
    )
