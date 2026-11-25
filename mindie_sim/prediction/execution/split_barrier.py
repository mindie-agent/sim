"""Prefill Eager MoE All2AllV D2H split-size barrier model.

[Role]
  Model ``TokenDispatcherWithAll2AllV`` Prefill Eager host-consumption split
  barriers (``vllm-ascend/ops/fused_moe/token_dispatcher.py:647-682``):
  ``input_splits`` / ``output_splits`` D2H copies over coupled per-die +
  root-complex resources, per-transfer pageable cap, and conditional device
  fence when ``num_local_experts <= 1``.

[Invariants]
  * ``repeat_interleave`` depends on ``allgather_counts`` (never runs before
    EP counts finish gathering); ``input_splits`` D2H overlaps ``allgather``.
  * Per-die D2H stays pinned at physical ceiling; ``pinned=False`` adds a
    separate coupled pageable cap so mixed transfers stay order-independent.
  * Explicit ``torch.npu.synchronize()`` fence fires iff ``num_local_experts <= 1``.

[Timeouts & Locks]
  Pure deterministic Engine task submission; no locks or blocking I/O.

[Side-Effect Policy]
  Registers D2H, host, vector, and root-complex resources on caller Engine.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError
from ..collectives.ports import (
    GBPS_TO_BYTES_PER_MS,
    host_solo_resource_name,
    host_transfer_demand,
    root_complex_resources_for_dies,
)
from ..engine import Event, Resource

MS_PER_S = 1000.0
DEFAULT_SPLIT_DTYPE_BYTES = 8


@dataclass(frozen=True, slots=True)
class SplitBarrierSpec:
    """Declared parameters for Prefill Eager ``TokenDispatcherWithAll2AllV`` D2H."""

    ep_size: int
    num_local_experts: int
    split_dtype_bytes: int = DEFAULT_SPLIT_DTYPE_BYTES
    pinned: bool = True
    non_blocking: bool = True
    host_consume_us: float = 0.0
    histogram_ms: float = 0.0
    repeat_interleave_ms: float = 0.0
    allgather_ms: float = 0.0
    sync_overhead_us: float = 0.0

    def __post_init__(self):
        for field_name in ("ep_size", "num_local_experts", "split_dtype_bytes"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ModelingError(f"split_barrier_{field_name}_invalid:{val}")
        for field_name in ("host_consume_us", "histogram_ms",
                           "repeat_interleave_ms", "allgather_ms", "sync_overhead_us"):
            val = getattr(self, field_name)
            if (isinstance(val, bool) or not isinstance(val, (int, float))
                    or not math.isfinite(float(val)) or float(val) < 0.0):
                raise ModelingError(f"split_barrier_{field_name}_invalid:{val}")

    @property
    def input_splits_bytes(self) -> int:
        return self.ep_size * self.split_dtype_bytes

    @property
    def output_splits_bytes(self) -> int:
        return self.ep_size * self.split_dtype_bytes

    @property
    def total_d2h_bytes(self) -> int:
        return self.input_splits_bytes + self.output_splits_bytes

    @property
    def requires_explicit_device_sync(self) -> bool:
        """Matches upstream ``token_dispatcher.py:678-682``."""
        return self.num_local_experts <= 1


@dataclass(frozen=True, slots=True)
class SplitBarrierResult:
    """Causal events for one Prefill Eager ``TokenDispatcherWithAll2AllV`` barrier."""

    histogram: Event
    input_splits_d2h: Event
    allgather_counts: Event
    output_splits_d2h: Event
    device_branch_done: Event
    explicit_device_sync: Event | None
    host_consume: Event
    dispatch_ready: Event
    input_splits_bytes: int
    output_splits_bytes: int
    d2h_resources: tuple[str, ...]


class EagerSplitBarrier:
    """Prefill Eager ``TokenDispatcherWithAll2AllV`` D2H host-consumption barrier.

    Models upstream ``vllm-ascend/ops/fused_moe/token_dispatcher.py:647-682``:
    ``histogram`` -> ``input_splits`` D2H (overlaps ``allgather_counts``);
    ``ep_counts`` AllGather -> ``output_splits`` D2H;
    ``repeat_interleave`` after ``allgather_counts`` (``num_local_experts > 1``)
    OR explicit ``torch.npu.synchronize()`` device fence (``num_local_experts <= 1``);
    host consumes splits -> gates ``dispatch AllToAllV``.
    """

    @staticmethod
    def _d2h_channel_params(machine) -> tuple[float, float | None, float]:
        channels = dict(getattr(machine, "transfer_channels", ()) or ())
        decl = dict(channels.get("d2h") or ())
        if hasattr(machine, "host_solo_bw_gbps"):
            pinned_gbps = float(machine.host_solo_bw_gbps("d2h") or 0.0)
        else:
            pinned_gbps = float(decl.get("effective_uni_bw_gbps") or 0.0)
        if pinned_gbps <= 0.0:
            raise ModelingError("eager_split_barrier_d2h_channel_unconfigured")
        pageable_raw = decl.get("pageable_uni_bw_gbps")
        pageable_rate = (
            float(pageable_raw) * GBPS_TO_BYTES_PER_MS
            if pageable_raw is not None and float(pageable_raw) > 0.0
            else None
        )
        alpha_us = float(decl.get("latency_us", decl.get("alpha_us", 0.0)) or 0.0)
        return pinned_gbps * GBPS_TO_BYTES_PER_MS, pageable_rate, alpha_us / MS_PER_S

    @classmethod
    def ensure_resources(
        cls,
        engine,
        machine,
        die,
        *,
        pinned: bool = True,
        namespace: str = "",
        d2h_rank_resource: str | None = None,
        host_service_resource: str | None = None,
        device_vector_resource: str | None = None,
    ) -> tuple[str, str | None, str, str]:
        """Ensure per-die D2H, per-transfer pageable cap, RC, host, and device rows."""
        rank = int(getattr(die, "rank", 0))
        solo_name = host_solo_resource_name(die, "d2h", namespace=namespace)
        d2h_name = d2h_rank_resource or solo_name
        host_name = host_service_resource or f"{namespace}host@rank{rank}"
        vec_name = device_vector_resource or f"{namespace}vector@rank{rank}"
        pinned_rate, pageable_rate, _ = cls._d2h_channel_params(machine)
        add_res = []
        if solo_name not in engine.resources:
            add_res.append(Resource(solo_name, pinned_rate, "bytes"))
        if d2h_name != solo_name and d2h_name not in engine.resources:
            add_res.append(Resource(d2h_name, pinned_rate, "bytes"))
        pageable_name: str | None = None
        if not pinned and pageable_rate is not None:
            pageable_name = f"{namespace}d2h_pageable@rank{rank}"
            if pageable_name not in engine.resources:
                add_res.append(Resource(pageable_name, pageable_rate, "bytes"))
        if host_name not in engine.resources:
            add_res.append(Resource(host_name, 1.0, "ms", "fifo"))
        if vec_name not in engine.resources:
            add_res.append(Resource(vec_name, 1.0, "ms"))
        for rc_res in root_complex_resources_for_dies(
            machine, (die,), namespace=namespace
        ):
            if rc_res.name not in engine.resources:
                add_res.append(rc_res)
        if add_res:
            engine.extend(add_res)
        return d2h_name, pageable_name, host_name, vec_name

    @classmethod
    def submit(
        cls,
        engine,
        machine,
        die,
        spec: SplitBarrierSpec,
        ready: Event,
        *,
        name: str = "eager_split_barrier",
        namespace: str = "",
        d2h_rank_resource: str | None = None,
        host_service_resource: str | None = None,
        device_vector_resource: str | None = None,
        allgather_event: Event | None = None,
        independent_device_events: Iterable[Event] = (),
        owner: str = "",
    ) -> SplitBarrierResult:
        """Submit the Prefill Eager split-size D2H pipeline onto ``engine``."""
        d2h_name, pageable_name, host_name, vec_name = cls.ensure_resources(
            engine, machine, die,
            pinned=spec.pinned,
            namespace=namespace,
            d2h_rank_resource=d2h_rank_resource,
            host_service_resource=host_service_resource,
            device_vector_resource=device_vector_resource,
        )
        _, _, d2h_latency_ms = cls._d2h_channel_params(machine)

        histogram = (
            engine.submit(
                f"{name}:histogram",
                {vec_name: spec.histogram_ms},
                after=(ready,),
                owner=owner,
            )
            if spec.histogram_ms > 0.0
            else engine.submit(f"{name}:histogram", after=(ready,), owner=owner)
        )

        # Branch A: local counts sum -> input_splits D2H (overlaps AllGather)
        solo_name = host_solo_resource_name(die, "d2h", namespace=namespace)
        in_demand = host_transfer_demand(
            machine, die, "d2h", spec.input_splits_bytes,
            namespace=namespace, rank_resource=solo_name,
        )
        if d2h_name != solo_name:
            in_demand[d2h_name] = float(spec.input_splits_bytes)
        if pageable_name is not None:
            in_demand[pageable_name] = float(spec.input_splits_bytes)
        in_svc = engine.submit(
            f"{name}:input_splits_d2h:svc",
            in_demand,
            after=(histogram,),
            owner=owner,
            coupled=len(in_demand) > 1,
        )
        input_d2h = engine.submit(
            f"{name}:input_splits_d2h:arr",
            after=(in_svc,),
            latency=d2h_latency_ms,
            owner=owner,
        )

        # Branch B: EP counts AllGather -> local slice/sum -> output_splits D2H
        if allgather_event is not None:
            ag_done = engine.submit(
                f"{name}:ep_counts_allgather",
                after=(histogram, allgather_event),
                owner=owner,
            )
        else:
            ag_done = engine.submit(
                f"{name}:ep_counts_allgather",
                after=(histogram,),
                latency=spec.allgather_ms,
                owner=owner,
            )
        out_after = (ag_done,) if spec.non_blocking else (ag_done, input_d2h)
        out_demand = host_transfer_demand(
            machine, die, "d2h", spec.output_splits_bytes,
            namespace=namespace, rank_resource=solo_name,
        )
        if d2h_name != solo_name:
            out_demand[d2h_name] = float(spec.output_splits_bytes)
        if pageable_name is not None:
            out_demand[pageable_name] = float(spec.output_splits_bytes)
        out_svc = engine.submit(
            f"{name}:output_splits_d2h:svc",
            out_demand,
            after=out_after,
            owner=owner,
            coupled=len(out_demand) > 1,
        )
        output_d2h = engine.submit(
            f"{name}:output_splits_d2h:arr",
            after=(out_svc,),
            latency=d2h_latency_ms,
            owner=owner,
        )

        # Branch C: repeat_interleave AFTER allgather_counts (>1 local expert)
        # OR explicit torch.npu.synchronize() device fence (<=1 local expert).
        if spec.requires_explicit_device_sync:
            fence_deps = (ag_done, input_d2h, output_d2h, *tuple(independent_device_events))
            explicit_sync = engine.submit(
                f"{name}:torch_npu_synchronize",
                after=fence_deps,
                latency=spec.sync_overhead_us / MS_PER_S,
                owner=owner,
            )
            device_branch_done = explicit_sync
            host_deps = (explicit_sync,)
        else:
            explicit_sync = None
            device_branch_done = (
                engine.submit(
                    f"{name}:repeat_interleave",
                    {vec_name: spec.repeat_interleave_ms},
                    after=(ag_done,),
                    owner=owner,
                )
                if spec.repeat_interleave_ms > 0.0
                else engine.submit(
                    f"{name}:repeat_interleave",
                    after=(ag_done,),
                    owner=owner,
                )
            )
            host_deps = (input_d2h, output_d2h)

        host_consume = engine.submit(
            f"{name}:host_consume_splits",
            {host_name: spec.host_consume_us / MS_PER_S},
            after=host_deps,
            owner=owner,
        )
        dispatch_ready = engine.submit(
            f"{name}:dispatch_ready",
            after=(host_consume, device_branch_done),
            owner=owner,
        )
        return SplitBarrierResult(
            histogram=histogram,
            input_splits_d2h=input_d2h,
            allgather_counts=ag_done,
            output_splits_d2h=output_d2h,
            device_branch_done=device_branch_done,
            explicit_device_sync=explicit_sync,
            host_consume=host_consume,
            dispatch_ready=dispatch_ready,
            input_splits_bytes=spec.input_splits_bytes,
            output_splits_bytes=spec.output_splits_bytes,
            d2h_resources=tuple(sorted(in_demand)),
        )
