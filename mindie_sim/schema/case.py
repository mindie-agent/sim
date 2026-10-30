"""Hardware-neutral simulation inputs.

[Role] Bind model geometry, hardware resources, software policy and requests.
[Invariants] Runtime configuration contains execution mechanisms only.
  Calibration and validation have separate interfaces.
[Timeouts & Locks] Pure metadata validation; no I/O or hardware access.
[Side-Effect Policy] Immutable records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json

from .hardware import Hardware
from .model import ModelSpec, SloSpec, WorkloadSpec
from .parallel import ParallelSpec
from .common import ModelingError


@dataclass(frozen=True)
class Software:
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_model_len: int = 32768
    max_num_input_tokens: int | None = None
    enable_chunked_prefill: bool = True
    kv_block_size: int = 128
    gpu_memory_utilization: float = .9
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
    pp_layer_partition: tuple[int, ...] = ()
    pipeline_microbatches: int = 1
    async_scheduling: bool = False
    graph_mode: str = "auto"
    graph_capture_mode: str = "full"
    graph_buckets: tuple[int, ...] = ()
    graph_capture_keys: tuple[int, ...] = ()
    piecewise_capture_keys: tuple[int, ...] = ()
    mixed_mode: str = "none"
    splitting_ops: tuple[str, ...] | None = None
    invalid_full_modes: bool = False
    deepep_high_throughput: bool = False
    multistream_overlap_shared_expert: bool = False
    moe_recipe: str = "alltoall"
    mc2_chunks: int = 4
    mtp_acceptance_length: float = 1
    mtp_acceptance_by_output_tokens: tuple[tuple[int, float], ...] = ()
    dspark: object = None
    engram_cpu_offload: bool = False
    vllm_impl: str = ""
    vllm_version: str = ""
    kernel_format: str = ""
    kernels: dict = field(default_factory=dict)
    submission: dict = field(default_factory=dict)
    serving_io: dict = field(default_factory=dict)
    attention_policy: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    tensor_layouts: dict = field(default_factory=dict)
    final_norm_scope: str = "all"
    lowering_passes: tuple[str, ...] = ("activation_quantization",)
    memory: dict = field(default_factory=dict)
    metadata: tuple[tuple[str, object], ...] = ()

    @staticmethod
    def parallel_mapping(raw):
        """Combine the two accepted axis locations without hiding a declaration."""
        axes = {f.name for f in fields(ParallelSpec)}
        parallel = dict(raw.get("parallel") or {})
        for key in raw.keys() & axes:
            if key in parallel and parallel[key] != raw[key]:
                raise ModelingError(f"runtime_parallel_conflict:{key}")
            parallel[key] = raw[key]
        # A partial edit must update the physical axis, not leave a shadow alias.
        for alias, name in (("dcp", "cp"), ("prefill_context_parallel_size", "pcp"),
                            ("context_parallel_plan", "cp_plan")):
            if alias in parallel:
                if name in parallel and parallel[name] != parallel[alias]:
                    raise ModelingError(f"runtime_parallel_conflict:{name}:{alias}")
                parallel[name] = parallel.pop(alias)
        return parallel

    @classmethod
    def from_mapping(cls, raw):
        names = {f.name for f in fields(cls)}
        axes = {f.name for f in fields(ParallelSpec)}
        unknown = raw.keys() - names - axes
        if unknown:
            raise ModelingError("runtime_unknown_fields:" + ",".join(sorted(unknown)))
        kwargs = {key: value for key, value in raw.items() if key in names}
        kwargs["parallel"] = ParallelSpec.from_mapping(cls.parallel_mapping(raw))
        kwargs["metadata"] = tuple(sorted(raw.get("metadata", {}).items()))
        for key in ("graph_buckets", "graph_capture_keys", "piecewise_capture_keys", "pp_layer_partition",
                    "lowering_passes"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key] or ())
        if isinstance(kwargs.get("mtp_acceptance_by_output_tokens"), dict):
            kwargs["mtp_acceptance_by_output_tokens"] = tuple(
                (int(k), float(v)) for k, v in kwargs["mtp_acceptance_by_output_tokens"].items()
            )
        elif "mtp_acceptance_by_output_tokens" in kwargs:
            kwargs["mtp_acceptance_by_output_tokens"] = tuple(
                tuple(item) for item in kwargs["mtp_acceptance_by_output_tokens"])
        if isinstance(kwargs.get("dspark"), dict):
            from mindie_sim.schema.dspark import DsparkSpec
            kwargs["dspark"] = DsparkSpec.from_mapping(kwargs["dspark"])
        result = cls(**kwargs)
        if not 0 < result.gpu_memory_utilization <= 1:
            raise ModelingError("memory_utilization_must_be_in_0_1")
        if any(getattr(result, key) < 1 for key in (
            "max_num_batched_tokens", "max_num_seqs", "max_model_len", "kv_block_size", "pipeline_microbatches")):
            raise ModelingError("scheduler_capacities_must_be_positive")
        if result.moe_recipe not in {"alltoall", "mc2", "tp_join_reduce"} or result.mc2_chunks < 1:
            raise ModelingError("moe_requires_registered_recipe_and_positive_mc2_chunks")
        if result.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("final_norm_scope_requires_all_or_selected")
        return result

    def to_dict(self):
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        data["parallel"] = self.parallel.to_dict()
        return data


@dataclass(frozen=True)
class Case:
    machine: Hardware
    model: ModelSpec
    workload: WorkloadSpec
    runtime: Software = field(default_factory=Software)
    slo: SloSpec = field(default_factory=SloSpec)

    @classmethod
    def from_mapping(cls, value):
        from mindie_sim.prediction.structure import parse_case
        return parse_case(value)

    def to_dict(self):
        return {
            "machine": self.machine.to_dict(), "model": self.model.to_dict(),
            "workload": {"requests": [request.to_dict() for request in self.workload.requests]},
            "runtime": self.runtime.to_dict(), "slo": self.slo.to_dict(),
        }
