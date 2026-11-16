"""Hardware-neutral simulation inputs.

[Role] Bind model geometry, hardware resources, software policy and requests.
[Invariants] Runtime configuration contains execution mechanisms only.
  Calibration and validation have separate interfaces.
[Timeouts & Locks] Pure metadata validation; no I/O or hardware access.
[Side-Effect Policy] Immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import json
from typing import Any

from .hardware import Hardware
from .model import ModelSpec, SloSpec, WorkloadSpec
from .parallel import PARALLEL_AXIS_ALIASES, PARALLEL_SWITCHES, ParallelSpec
from .common import ModelingError
from . import runtime_features


#: vLLM cudagraph mode -> (simulation graph_capture_mode, full keys, piecewise keys).
_CUDAGRAPH_MODES = {
    "NONE": ("none", False, False),
    "PIECEWISE": ("piecewise", False, True),
    "FULL": ("full", True, False),
    "FULL_DECODE_ONLY": ("full", True, False),
    "FULL_AND_PIECEWISE": ("piecewise", True, True),
}

#: vLLM-Ascend ``all2all_backend`` -> (moe_recipe, deepep_high_throughput).
_ALL2ALL_BACKENDS = {
    "alltoallv": ("alltoall", False),
    "alltoall": ("alltoall", False),
    "mc2": ("mc2", False),
    "tp_join_reduce": ("tp_join_reduce", False),
    "local_expert_reduce": ("local_expert_reduce", False),
    "deepep_high_throughput": ("alltoall", True),
    "deepep_low_latency": ("alltoall", False),
}


def _capture_sizes(value: Any, reason: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ModelingError(reason)
    sizes = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ModelingError(reason)
        sizes.append(int(item))
    return tuple(sorted(set(sizes)))


def _compilation_fields(raw: Mapping[str, Any], declared: Mapping[str, Any]
                         ) -> dict[str, Any]:
    """Map a vLLM ``compilation_config`` onto the simulation graph fields.

    The raw cudagraph mode is preserved as ``captured_execution_mode`` metadata
    because the simulation has one full-graph mode while vLLM distinguishes
    ``FULL`` from ``FULL_DECODE_ONLY``.  A declaration made twice must agree.
    """
    config = raw.get("compilation_config")
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ModelingError("runtime_compilation_config_invalid")
    unknown = set(config) - {
        "cudagraph_mode", "cudagraph_capture_sizes", "splitting_ops",
        "level", "use_cudagraph", "cudagraph_num_of_warmups",
    }
    if unknown:
        raise ModelingError(
            "runtime_compilation_config_unknown:" + ",".join(sorted(unknown)))
    mode_raw = str(config.get("cudagraph_mode") or "").strip().upper()
    fields: dict[str, Any] = {}
    if mode_raw:
        if mode_raw not in _CUDAGRAPH_MODES:
            raise ModelingError(f"runtime_cudagraph_mode_unsupported:{mode_raw}")
        mode, full_keys, piecewise_keys = _CUDAGRAPH_MODES[mode_raw]
        if declared.get("graph_capture_mode") is not None and str(
                declared["graph_capture_mode"]) != mode:
            raise ModelingError(
                "runtime_compilation_conflict:graph_capture_mode:"
                f"{declared['graph_capture_mode']}!={mode}")
        fields["graph_capture_mode"] = mode
        sizes = _capture_sizes(config.get("cudagraph_capture_sizes"),
                               "runtime_cudagraph_capture_sizes_invalid")
        if sizes:
            if full_keys:
                if declared.get("graph_capture_keys") not in (None, (), sizes):
                    raise ModelingError(
                        "runtime_compilation_conflict:graph_capture_keys")
                fields["graph_capture_keys"] = sizes
            if piecewise_keys:
                if declared.get("piecewise_capture_keys") not in (None, (), sizes):
                    raise ModelingError(
                        "runtime_compilation_conflict:piecewise_capture_keys")
                fields["piecewise_capture_keys"] = sizes
        fields["metadata"] = (("captured_execution_mode", mode_raw),)
    ops = config.get("splitting_ops")
    if ops is not None:
        if not isinstance(ops, (list, tuple)) or not all(
                isinstance(item, str) and item for item in ops):
            raise ModelingError("runtime_splitting_ops_invalid")
        declared_ops = declared.get("splitting_ops")
        if declared_ops is not None and tuple(declared_ops) != tuple(ops):
            raise ModelingError("runtime_compilation_conflict:splitting_ops")
        fields["splitting_ops"] = tuple(ops)
    return fields


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
    enable_dbo: bool = False
    multistream_kda_proj: bool = False
    multistream_attention_indexer: bool = False
    multistream_window_compressed: bool = False
    moe_recipe: str = "alltoall"
    mc2_chunks: int = 4
    mtp_acceptance_length: float = 1
    mtp_acceptance_by_output_tokens: tuple[tuple[int, float], ...] = ()
    recover_ssm: bool = False
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
    #: AllToAll/AllToAllV startup model: legacy, ring, log2, concurrent
    #: (direct/concurrent normalize here) or topology (declared by hardware).
    collective_startup_mode: str = "legacy"
    #: ``auto`` resolves eligible generic packed FP4 work on FP8 hardware.
    matrix_compute_policy: str = "auto"
    #: Shared prefix pools assert that equal keys denote equal immutable tokens.
    prefix_cache_mode: str = "private"
    #: Whether the semantic indexer reuses a physical prefix across queries.
    prefix_read_mode: str = "auto"
    #: Homogeneous cohort semantic-graph reuse: off, auto or require.
    homogeneous_graph_reuse: str = "off"

    @staticmethod
    def parallel_mapping(raw):
        """Combine the two accepted axis locations without hiding a declaration."""
        axes = ({f.name for f in fields(ParallelSpec)}
                | set(PARALLEL_AXIS_ALIASES) | set(PARALLEL_SWITCHES))
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
        raw = runtime_features.normalize_runtime_features(raw)
        names = {f.name for f in fields(cls)}
        axes = ({f.name for f in fields(ParallelSpec)}
                | set(PARALLEL_AXIS_ALIASES) | set(PARALLEL_SWITCHES))
        aliases = {"compilation_config", "all2all_backend", "block_size"}
        unknown = raw.keys() - names - axes - aliases
        if unknown:
            raise ModelingError("runtime_unknown_fields:" + ",".join(sorted(unknown)))
        kwargs = {key: value for key, value in raw.items() if key in names}
        kwargs["parallel"] = ParallelSpec.from_mapping(cls.parallel_mapping(raw))
        declared = dict(raw)
        metadata = dict(raw.get("metadata", {}) or {})
        # --- vLLM graph / compilation configuration ---------------------
        compiled = _compilation_fields(raw, declared)
        merged_metadata = dict(compiled.pop("metadata", ()))
        for key, value in merged_metadata.items():
            if key in metadata and metadata[key] != value:
                raise ModelingError(f"runtime_metadata_conflict:{key}")
            metadata[key] = value
        for key, value in compiled.items():
            if key in kwargs and kwargs[key] != value:
                raise ModelingError(f"runtime_compilation_conflict:{key}")
            kwargs[key] = value
        # --- vLLM-Ascend MoE token dispatcher ---------------------------
        backend = raw.get("all2all_backend")
        if backend is not None:
            entry = _ALL2ALL_BACKENDS.get(str(backend).strip().lower())
            if entry is None:
                raise ModelingError(f"runtime_all2all_backend_unsupported:{backend}")
            recipe, high_throughput = entry
            if "moe_recipe" in kwargs and kwargs["moe_recipe"] != recipe:
                raise ModelingError(
                    f"runtime_all2all_backend_conflict:moe_recipe:"
                    f"{kwargs['moe_recipe']}!={recipe}")
            if ("deepep_high_throughput" in kwargs
                    and bool(kwargs["deepep_high_throughput"]) != high_throughput
                    and str(backend).strip().lower().startswith("deepep")):
                raise ModelingError(
                    "runtime_all2all_backend_conflict:deepep_high_throughput")
            kwargs["moe_recipe"] = recipe
            kwargs["deepep_high_throughput"] = high_throughput
            if "all2all_backend" in metadata and metadata["all2all_backend"] != str(backend):
                raise ModelingError("runtime_metadata_conflict:all2all_backend")
            metadata["all2all_backend"] = str(backend).strip().lower()
        # --- vLLM block size spelling ------------------------------------
        if raw.get("block_size") is not None:
            block = raw["block_size"]
            if isinstance(block, bool) or not isinstance(block, int) or block < 1:
                raise ModelingError("runtime_block_size_invalid")
            if "kv_block_size" in kwargs and kwargs["kv_block_size"] != block:
                raise ModelingError(
                    f"runtime_block_size_conflict:{kwargs['kv_block_size']}!={block}")
            kwargs["kv_block_size"] = block
        kwargs["metadata"] = tuple(sorted(metadata.items()))
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
        if result.moe_recipe not in {"alltoall", "mc2", "tp_join_reduce",
                                     "local_expert_reduce"} or result.mc2_chunks < 1:
            raise ModelingError("moe_requires_registered_recipe_and_positive_mc2_chunks")
        if result.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("final_norm_scope_requires_all_or_selected")
        if not isinstance(result.recover_ssm, bool):
            raise ModelingError("runtime_recover_ssm_invalid")
        runtime_features.validate_runtime_features(result)
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
            "workload": self.workload.to_dict(),
            "runtime": self.runtime.to_dict(), "slo": self.slo.to_dict(),
        }
