"""Deterministic instantiation of canonical subgraphs and static ``Theta``.

[Role]
  Project ``(ModelSpec, LayerPlan, FeatureConfig)`` into immutable, typed
  ``SubgraphInstance`` records.  Each operator receives a fully fixed
  ``StaticSignature``: weight dimensions, local head/expert geometry,
  quantisation role bytes, parallel axes and declared tiles.  Only sequence
  geometry and explicit state ports remain dynamic.

[Invariants]
  * Instantiation is a pure function of model/layer/features and declared
    implementation tiles; no measurement, model-name branch or request id is
    consulted.
  * Semantic macros reference child subgraphs instead of duplicating their
    work.  A child kind present in ``SubgraphTemplate.children`` contributes no
    second operator list here.
  * Quantisation is resolved per role; an override on one role cannot rewrite
    another role's weight payload, scale or zero-point accounting.
  * Repeated layers intern the same template object; instance identity and
    layer ids stay outside the template and performance key.
  * Unsupported mechanisms remain ``CoverageHole`` records, never fabricated
    operators or measured latency.

[Timeouts & Locks]
  Bounded per-layer construction; no iteration history, waits or locks.

[Side-Effect Policy]
  Returns new immutable records and mutates only the caller-supplied bounded
  template cache.  No I/O, device access or global model state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.subgraphs import (
    BoundOperator, CoverageHole, CoverageReport, FusionContract,
    OperatorTemplate, StatePortSpec, StaticSignature, SubgraphInstance,
    SubgraphTemplate,
)
from .registry import (
    TemplateRegistry, build, port, register_builder, template,
    validate_combination,
)

_DEFAULT_MT = 128
_DEFAULT_NT = 128
_DEFAULT_KV_TILE = 128
_DEFAULT_CORES = 24
_DEFAULT_MOE_ROWS = 128
_DEFAULT_SPLIT_TARGET = 4096
_DEFAULT_MAX_SPLITS = 8

@dataclass(frozen=True)
class FeatureConfig:
    """Normalised projection of Runtime/ParallelSpec/QuantSpec.

    This is deliberately not a second user-facing schema: it is constructed
    from the existing runtime/model records and only adds implementation
    tiles that the static contract must fix.
    """

    tp: int = 1
    ep: int = 1
    etp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    pcp: int = 1
    activation_dtype: str = "bf16"
    quant_activation_format: str = ""
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    index_dtype: str = "int8"
    quant: QuantSpec | None = None
    graph_buckets: tuple[int, ...] = ()
    multistream_shared_expert: bool = False
    multistream_kda_proj: bool = False
    multistream_attention_indexer: bool = False
    multistream_window_compressed: bool = False
    moe_recipe: str = "alltoall"
    mc2_chunks: int = 4
    lowering_passes: tuple[str, ...] = ("activation_quantization",)
    implementation: str = "analytical"
    final_norm_scope: str = "all"
    token_tile: int = _DEFAULT_MT
    kv_tile: int = _DEFAULT_KV_TILE
    cores: int = _DEFAULT_CORES
    moe_row_tile: int = _DEFAULT_MOE_ROWS
    split_target: int = _DEFAULT_SPLIT_TARGET
    max_splits: int = _DEFAULT_MAX_SPLITS
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("tp", "ep", "etp", "pp", "dp", "cp", "pcp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_parallel_invalid:{name}")
        for name in ("token_tile", "kv_tile", "cores", "moe_row_tile",
                     "split_target", "max_splits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_tile_invalid:{name}")
        for name in ("activation_dtype", "weight_dtype", "kv_dtype", "index_dtype"):
            dtype_bytes(getattr(self, name))
        if self.quant_activation_format:
            dtype_bytes(self.quant_activation_format)
        if self.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("feature_final_norm_scope_invalid")
        if self.moe_recipe not in {"alltoall", "mc2", "tp_join_reduce",
                                   "local_expert_reduce"}:
            raise ModelingError(f"feature_moe_recipe_invalid:{self.moe_recipe}")
        if self.mc2_chunks < 1:
            raise ModelingError("feature_mc2_chunks_invalid")
        object.__setattr__(self, "graph_buckets", tuple(sorted(set(self.graph_buckets))))
        object.__setattr__(self, "lowering_passes", tuple(self.lowering_passes))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", tuple(sorted(self.metadata.items())))
        else:
            object.__setattr__(self, "metadata", tuple(self.metadata))

    @property
    def activation_format(self) -> str:
        return self.quant_activation_format or self.activation_dtype or "bf16"

    @property
    def world_size(self) -> int:
        return self.dp * self.pp * self.tp * self.pcp

    @property
    def etp_width(self) -> int: return max(1, self.etp)

    @property
    def max_bucket(self) -> int: return max(self.graph_buckets) if self.graph_buckets else 0

    def project(self, *, tp: int = 0, dp: int = 0, ep: int = 0, pp: int = 0,
                cores: int = 0, graph_buckets: Iterable[int] = (), kv_dtype: str = "",
                kv_bf16_override: bool = False) -> "FeatureConfig":
        tp_v, dp_v = max(1, tp or self.tp), max(1, dp or self.dp)
        ep_v = max(1, int(ep or (max(self.ep, dp_v * tp_v) if self.ep > 1 else 1)))
        etp_v = max(1, (dp_v * tp_v) // ep_v) if ep_v > 1 else tp_v
        meta = dict(self.metadata)
        if kv_bf16_override: meta["kv_bf16_override"] = 1
        updates: dict[str, Any] = dict(tp=tp_v, dp=dp_v, ep=ep_v, etp=etp_v,
                                       pp=max(1, pp or self.pp), graph_buckets=tuple(graph_buckets),
                                       metadata=tuple(sorted(meta.items())))
        if cores > 0: updates["cores"] = int(cores)
        if kv_dtype: updates["kv_dtype"] = kv_dtype
        return replace(self, **updates)

    @classmethod
    def from_runtime(cls, runtime: Any, model: ModelSpec | None = None) -> "FeatureConfig":
        parallel = getattr(runtime, "parallel", ParallelSpec()) if runtime is not None else ParallelSpec()
        if not isinstance(parallel, ParallelSpec):
            raise ModelingError("feature_parallel_record_invalid")
        quant = getattr(model, "quant", None) if model is not None else None
        return cls(
            tp=parallel.tp, ep=parallel.ep, etp=max(1, getattr(parallel, "etp", 1)),
            pp=parallel.pp, dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
            activation_dtype=getattr(model, "activation_dtype", "bf16") if model else "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=getattr(model, "weight_dtype", "bf16") if model else "bf16",
            kv_dtype=getattr(quant, "kv_dtype", "") if quant else (
                getattr(model, "kv_dtype", "bf16") if model else "bf16"),
            index_dtype=getattr(quant, "index_dtype", "int8") if quant else "int8",
            quant=quant,
            graph_buckets=tuple(getattr(runtime, "graph_capture_keys", ()) or getattr(runtime, "graph_buckets", ())),
            multistream_shared_expert=bool(getattr(runtime, "multistream_overlap_shared_expert", False)),
            multistream_kda_proj=bool(getattr(runtime, "multistream_kda_proj", False)),
            multistream_attention_indexer=bool(getattr(runtime, "multistream_attention_indexer", False)),
            multistream_window_compressed=bool(getattr(runtime, "multistream_window_compressed", False)),
            moe_recipe=str(getattr(runtime, "moe_recipe", "alltoall")),
            mc2_chunks=int(getattr(runtime, "mc2_chunks", 4)),
            lowering_passes=tuple(getattr(runtime, "lowering_passes", ("activation_quantization",))),
            final_norm_scope=str(getattr(runtime, "final_norm_scope", "all")),
            metadata=tuple(getattr(runtime, "metadata", ()) or ()),
        )

    @classmethod
    def from_model(cls, model: ModelSpec, **overrides: Any) -> "FeatureConfig":
        quant = getattr(model, "quant", None)
        base = dict(
            activation_dtype=model.activation_dtype or "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=model.weight_dtype or "bf16",
            kv_dtype=((getattr(quant, "kv_dtype", "") if quant else "")
                      or model.kv_dtype or "bf16"),
            index_dtype=((getattr(quant, "index_dtype", "") if quant else "")
                         or "int8"),
            quant=quant,
        )
        base.update(overrides)
        return cls(**base)

    def with_parallel(self, parallel: ParallelSpec) -> "FeatureConfig":
        return replace(self, tp=parallel.tp, ep=parallel.ep, pp=parallel.pp,
                       dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp)

@dataclass(frozen=True)
class _Ctx:
    model: ModelSpec
    layer: LayerPlan
    features: FeatureConfig
    stage: int
    role: str = "body"
    variant: str = ""
    layers: int = 0

    def __post_init__(self) -> None:
        if self.layer.index < 0:
            raise ModelingError("layer_index_invalid")
        if self.stage < 0 or self.stage >= self.features.pp:
            raise ModelingError(f"layer_stage_invalid:{self.stage}")
        if self.layers and self.layer.index >= self.layers:
            raise ModelingError(f"layer_index_exceeds_model:{self.layer.index}")

    @property
    def hidden(self) -> int: return int(self.model.hidden_size)
    @property
    def tp(self) -> int: return self.features.tp
    @property
    def ep(self) -> int: return self.features.ep
    @property
    def etp(self) -> int: return self.features.etp
    @property
    def heads(self) -> int: return int(self.layer.num_heads or self.model.num_heads)
    @property
    def local_heads(self) -> int: return max(1, self.heads // self.tp)
    @property
    def kv_heads(self) -> int: return int(self.layer.num_kv_heads or self.model.num_kv_heads or self.heads)
    @property
    def local_kv_heads(self) -> int: return local_kv_heads(self.kv_heads, self.tp)
    @property
    def group_heads(self) -> int: return max(1, self.local_heads // self.local_kv_heads)
    @property
    def width(self) -> int: return int(self.layer.head_dim or self.model.head_dim)
    @property
    def intermediate(self) -> int: return int(self.layer.intermediate_size or self.model.intermediate_size)
    @property
    def local_intermediate(self) -> int: return max(1, math.ceil(self.intermediate / self.tp))
    @property
    def local_experts(self) -> int: return max(1, math.ceil(self.model.num_experts / self.ep))

    @property
    def expert_width(self) -> int:
        width = int(self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        return max(1, math.ceil(width / max(1, self.etp)))

    @property
    def latent(self) -> int:
        return int(self.layer.expert_latent_size or self.hidden)

    @property
    def shared_width(self) -> int:
        width = int(self.layer.shared_intermediate_size
                    or self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        count = int(self.layer.num_shared_experts or self.model.num_shared_experts or 1)
        return width * max(1, count)

    @property
    def top_k(self) -> int:
        return int(self.layer.top_k or self.model.top_k or 1)

    @property
    def quant(self) -> QuantSpec | None:
        return self.features.quant

    @property
    def kv_dtype(self) -> str:
        return self.features.kv_dtype or self.model.kv_dtype

    @property
    def kv_row_elements(self) -> int:
        return int(self.layer.cache_row_elements)

    @property
    def kv_row_bytes(self) -> float:
        # Ordinary GQA/MQA owns one local KV row per rank (replicated when the
        # global KV head count is below TP); Gemma K=V aliases the value plane.
        # MLA and compressed representations keep their declared cache layout.
        if (self.layer.attention in {"gqa", "mha", "mqa"}
                and self.layer.cache_kind != "window_compressed"):
            facts = dict(self.model.catalog_facts)
            k_eq_v = bool(self.layer.k_eq_v) or (
                self.layer.index in {int(v) for v in facts.get("gemma4_k_eq_v_layers", ())})
            elements = self.local_kv_heads * (self.width if k_eq_v else 2 * self.width)
        else:
            elements = self.kv_row_elements
        return elements * dtype_bytes(self.kv_dtype)

    @property
    def shared_consumer(self) -> bool:
        owner = int(self.layer.kv_owner_layer_id)
        return 0 <= owner < self.layer.index and not self.layer.owns_long_kv

    @property
    def attention_kind(self) -> str:
        if self.layer.cache_kind == "window_compressed":
            return "A2"
        if self.layer.attention == "mla":
            return "A1"
        if self.layer.attention == "gdn":
            return "A3"
        if self.layer.attention == "kda":
            return "A4"
        return "A0"

def _sig(role: str, **groups: Any) -> StaticSignature:
    return StaticSignature(role, **groups)

def _parallel(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    return (("tp", ctx.tp), ("ep", ctx.ep), ("etp", ctx.etp),
            ("pp", ctx.features.pp), ("dp", ctx.features.dp),
            ("cp", ctx.features.cp), ("pcp", ctx.features.pcp),
            ("world", ctx.features.world_size), ("ranks_per_node", int(dict(ctx.features.metadata).get("ranks_per_node", 8) or 8)))

def _tiles(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    features = ctx.features
    return (("mt", features.token_tile), ("nt", features.kv_tile),
            ("cores", features.cores), ("moe_rows", features.moe_row_tile),
            ("split_target", features.split_target),
            ("max_splits", features.max_splits),
            ("graph_buckets", features.graph_buckets))

_LOW_ACT = frozenset({"mxfp8", "mxfp4", "fp8_e4m3", "int8"})
_HIGH_ACT = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})

def _matrix_act_format(ctx: _Ctx, category: str) -> str:
    stream = ctx.features.activation_dtype or "bf16"
    quant = ctx.quant
    if quant is None:
        declared = ctx.features.quant_activation_format
        return declared if declared in _LOW_ACT else stream
    resolved = quant.for_category(category)
    if str(resolved.format_for(category)) in _HIGH_ACT:
        return stream
    act = str(resolved.activation_format or ctx.features.quant_activation_format or "")
    return act if act in _LOW_ACT else stream


def _quant_fields(ctx: _Ctx, category: str, rows: int, cols: int) -> tuple[tuple[str, Any], ...]:
    rows, cols = int(rows), int(cols)
    quant = ctx.quant
    act_format = _matrix_act_format(ctx, category)
    fallback = dtype_bytes(act_format)
    if rows <= 0 or cols <= 0:
        return (("weight_format", ctx.features.weight_dtype), ("act_format", act_format),
                ("weight_payload", 0.0), ("weight_scale", 0.0), ("weight_zero", 0.0),
                ("weight_total", 0.0), ("residency", 0.0), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    if quant is None:
        weight_bytes = float(rows * cols) * dtype_bytes(ctx.features.weight_dtype)
        return (("weight_format", ctx.features.weight_dtype),
                ("act_format", act_format),
                ("weight_payload", weight_bytes), ("weight_scale", 0.0),
                ("weight_zero", 0.0), ("weight_total", weight_bytes),
                ("residency", weight_bytes), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    resolved = quant.for_category(category)
    payload = resolved.matrix_payload_bytes(rows, cols, category)
    total = resolved.matrix_bytes(rows, cols, category)
    zero = 0.0
    fmt = resolved.format_for(category)
    if fmt in {"int8", "w8a8"} and resolved.int8_weight_metadata == "scale_offset":
        zero = float(rows) * float(resolved.scale_bytes)
    scale = max(0.0, total - payload - zero)
    group = 0
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        group = int(resolved.weight_block[1] if resolved.weight_block else 0)
        if not group:
            group = int(resolved.expert_block or 0)
    return (("weight_format", fmt), ("act_format", act_format),
            ("weight_payload", payload), ("weight_scale", scale),
            ("weight_zero", zero), ("weight_total", total),
            ("residency", total), ("scale_group", group),
            ("act_bytes", dtype_bytes(act_format)),
            ("quantized", int(fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})))

def _matrix(role: str, k: int, n: int, ctx: _Ctx, category: str,
            *, extra: tuple[tuple[str, Any], ...] = (),
            slots: tuple[str, ...] = ("weight_total",)) -> StaticSignature:
    return _sig(role, matrix=(("K", int(k)), ("N", int(n))),
                quant=_quant_fields(ctx, category, n, k),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _elementwise(role: str, ctx: _Ctx, *, elements: int, ops: int,
                 read_bytes: float, write_bytes: float,
                 extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", dtype_bytes(ctx.features.activation_format)),),
                parallel=_parallel(ctx), tiles=_tiles(ctx),
                extras=(("ops_per_element", ops), ("read_bytes", read_bytes),
                        ("write_bytes", write_bytes), *extra))

def _copy(role: str, ctx: _Ctx, *, elements: int, bytes_per_element: float,
          extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", bytes_per_element),),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _collective(role: str, ctx: _Ctx, *, axis: str, members: int,
                bytes_per_token: float, stages: int = 2,
                extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, parallel=_parallel(ctx), tiles=_tiles(ctx),
                quant=(("wire_bytes", float(bytes_per_token)),),
                extras=(("group_axis", axis), ("members", int(members)),
                        ("stages", int(stages)), *extra))

def _state(role: str, ctx: _Ctx, *, row_bytes: float, rows: int = 1,
           extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("rows", int(rows)), ("row_bytes", float(row_bytes))),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _attention_sig(role: str, ctx: _Ctx, *, local_heads: int, local_kv_heads: int,
                   group_heads: int, width: int, value_width: int,
                   kv_row_bytes: float, extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(
        role,
        heads=(("q", int(local_heads)), ("kv", int(local_kv_heads)),
               ("g", int(group_heads)), ("dk", int(width)), ("dv", int(value_width))),
        quant=(("kv_bytes", float(kv_row_bytes)),
               ("act_bytes", dtype_bytes(ctx.features.activation_format))),
        parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=(("window", int(ctx.layer.window)), ("causal", int(not ctx.layer.noncausal_attention)),
                ("noncausal", int(ctx.layer.noncausal_attention)),
                *extra),
    )

# ---------------------------------------------------------------------------
# Reusable ports

def _in(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "in", representation, **kwargs)

def _out(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "out", representation, **kwargs)

_HIDDEN_IN = _in("hidden", "activation")
_HIDDEN_OUT = _out("hidden", "activation")
_RESIDUAL_IN = _in("residual", "activation")
_RESIDUAL_OUT = _out("residual", "activation")
_KV_IN = _in("kv_cache", "kv_cache")
_KV_OUT = _out("kv_cache", "kv_cache")
_KV_READY_OUT = _out("kv_ready", "readiness", scope="stage")

@dataclass(frozen=True)
class _Built:
    template: SubgraphTemplate
    theta: tuple[tuple[str, StaticSignature], ...]

    def signature(self, name: str) -> StaticSignature:
        for key, value in self.theta:
            if key == name:
                return value
        raise ModelingError(f"builder_signature_missing:{self.template.kind}:{name}")

def _cross_facts(ctx: _Ctx) -> dict[str, Any]:
    routed = str(dict(_quant_fields(ctx, "routed_expert", 1, 1))["weight_format"]) if int(ctx.model.num_experts or 0) else ""
    kv_bf16 = int(bool(dict(ctx.features.metadata).get("kv_bf16_override")))
    return {"kv_dtype": ctx.kv_dtype, "kv_bf16": kv_bf16, "routed_weight_format": routed, "max_bucket": ctx.features.max_bucket}

def _built(kind: str, ctx: _Ctx, variant: str, title: str,
           specs: Iterable[tuple[str, str, str]], signatures: Mapping[str, StaticSignature],
           *, children: tuple[str, ...] = (), fusions: tuple[FusionContract, ...] = (),
           state_in: tuple[StatePortSpec, ...] = (), state_out: tuple[StatePortSpec, ...] = (),
           coverage: str = "structural", contract: tuple[tuple[str, str], ...] = (),
           streams: Mapping[str, str] | None = None) -> _Built:
    operators, theta, facts, smap = [], [], _cross_facts(ctx), dict(streams or ())
    for name, family, role in specs:
        if name not in signatures:
            raise ModelingError(f"builder_signature_missing:{kind}:{name}")
        operators.append(OperatorTemplate(name=name, family=family, role=role, stream=smap.get(name, "main")))
        theta.append((name, signatures[name].relocated(**facts)))
    return _Built(SubgraphTemplate(kind=kind, variant=variant, title=title, operators=tuple(operators),
                  state_in=state_in, state_out=state_out, children=children, fusions=fusions,
                  coverage=coverage, contract=contract), tuple(theta))

def _variant(ctx: _Ctx, default: str) -> str:
    return ctx.variant or default

def _guard_low_act(specs: list, signatures: dict, ctx: _Ctx, name: str, *,
                   elements: int, category: str) -> None:
    if not name:
        return
    fmt = _matrix_act_format(ctx, category)
    if fmt not in _LOW_ACT or (specs and specs[-1][1] == "quantize"):
        return
    op = "DynamicMxQuant" if fmt in {"mxfp8", "mxfp4"} else "QuantV2"
    qname = f"{name}_{op}"
    count = int(elements)
    specs.append((qname, "quantize", "quantize"))
    signatures[qname] = _elementwise(
        "quantize", ctx, elements=count, ops=2,
        read_bytes=count * dtype_bytes(ctx.features.activation_dtype or "bf16"),
        write_bytes=count * dtype_bytes(fmt),
        extra=(("quant_op", op), ("out_format", fmt), ("category", category)))


def _precede(specs: list, signatures: dict, ctx: _Ctx, name: str,
             elements: int, category: str) -> None:
    """Insert the quant guard immediately before ``name`` when that operator exists."""
    if not name:
        return
    index = next((i for i, item in enumerate(specs) if item[0] == name), None)
    if index is None:
        return
    head = list(specs[:index])
    _guard_low_act(head, signatures, ctx, name, elements=elements, category=category)
    if len(head) > index:
        specs[index:index] = head[index:]


from .builders_attention import BUILDERS as _ATTENTION_BUILDERS
from .builders_ffn import BUILDERS as _FFN_BUILDERS

_BUILDERS = {**_ATTENTION_BUILDERS, **_FFN_BUILDERS}

for _kind, _builder in _BUILDERS.items():
    register_builder(_kind, _builder)

@dataclass(frozen=True)
class LayerSlot:
    kind: str
    variant: str
    role: str

@dataclass(frozen=True)
class LayerInstances:
    layer: int
    stage: int
    subgraphs: tuple[SubgraphInstance, ...]
    slots: tuple[LayerSlot, ...]
    holes: tuple[CoverageHole, ...] = ()

    def find(self, kind: str, role: str | None = None) -> SubgraphInstance:
        for instance in self.subgraphs:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{self.layer}:{role}")

@dataclass(frozen=True)
class InstantiatedModel:
    model: ModelSpec
    features: FeatureConfig
    layers: tuple[LayerInstances, ...]
    globals: tuple[SubgraphInstance, ...]
    coverage: CoverageReport
    stage_of: tuple[int, ...]

    @property
    def subgraphs(self) -> tuple[SubgraphInstance, ...]:
        return tuple(instance for layer in self.layers for instance in layer.subgraphs) + self.globals

    def find(self, kind: str, *, layer: int | None = None, role: str | None = None) -> SubgraphInstance:
        candidates = self.globals if layer is None else self.layers[layer].subgraphs
        for instance in candidates:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{layer}:{role}")

    def performance_keys(self) -> tuple[tuple, ...]:
        return tuple(instance.performance_key() for instance in self.subgraphs)

    def fingerprints(self) -> tuple[str, ...]:
        return tuple(instance.fingerprint() for instance in self.subgraphs)

def _stage_for(layer: int, layers: int, pp: int) -> int:
    return 0 if pp <= 1 or layers <= 1 else min(pp - 1, layer * pp // layers)

def _validate_parallel(model: ModelSpec, features: FeatureConfig) -> None:
    """Reject layouts whose local ownership would silently drop heads."""
    tp = max(1, features.tp)
    if tp > 1:
        heads = int(model.num_heads)
        if heads % tp:
            raise ModelingError(f"heads_not_divisible_by_tp:{heads}%{tp}")
        for layer in model.layer_plans:
            local = int(layer.num_heads or heads)
            if local % tp:
                raise ModelingError(
                    f"layer_heads_not_divisible_by_tp:{layer.index}:{local}%{tp}")
    ep = max(1, features.ep)
    if ep > 1 and int(model.num_experts or 0):
        if int(model.num_experts) % ep:
            raise ModelingError(
                f"experts_not_divisible_by_ep:{model.num_experts}%{ep}")

def _layer_has_ple(model: ModelSpec, layer: LayerPlan) -> bool:
    facts = dict(model.catalog_facts)
    layers = facts.get("gemma4_ple_layers")
    if layers is not None:
        return layer.index in {int(value) for value in layers}
    if facts.get("ple") is not None:
        return bool(facts.get("ple"))
    return bool(layer.engram and facts.get("gemma4"))

def _layer_slots(model: ModelSpec, layer: LayerPlan,
                 features: FeatureConfig) -> tuple[LayerSlot, ...]:
    slots: list[LayerSlot] = []
    hc = bool(layer.hyper_connections)

    def boundary(name: str) -> None:
        if hc:
            slots.append(LayerSlot("R0", name, name))

    if layer.residual_block_size:
        slots.append(LayerSlot("R1", "block_stack", "residual"))
    slots.append(LayerSlot("N0", "rmsnorm", "attn_norm"))
    boundary("attn_pre")
    attention_kind = ("A2" if layer.cache_kind == "window_compressed"
                      else "A1" if layer.attention == "mla"
                      else "A3" if layer.attention == "gdn"
                      else "A4" if layer.attention == "kda" else "A0")
    shared_consumer = (0 <= layer.kv_owner_layer_id < layer.index
                       and not layer.owns_long_kv)
    kv_slot = LayerSlot("K0", "read" if shared_consumer else "store", "kv")
    if attention_kind == "A0":
        variant = "mqa" if (layer.num_kv_heads or 0) == 1 and layer.num_heads > 1 else "gqa"
        if shared_consumer:
            variant = "shared_consumer"
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk",
                                   "index_select"))
        slots.append(LayerSlot(attention_kind, variant, "attention"))
        slots.append(kv_slot)
    elif attention_kind == "A1":
        if layer.compress_ratio or layer.cache_kind == "window_compressed":
            slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"), "compress"))
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk", "index_select"))
        slots.append(LayerSlot(attention_kind,
                               "latent_mqa" if layer.mla_variant == "latent_mqa"
                               else "latent_mla", "attention"))
        if shared_consumer:
            slots.append(kv_slot)
    elif attention_kind in {"A3", "A4"}:
        slots.append(LayerSlot(attention_kind, "", "attention"))
    else:
        slots.append(LayerSlot("A2", "window_compressed", "attention_prolog"))
        slots.append(LayerSlot("K0", "layout", "kv"))
        if layer.compress_ratio or layer.cache_kind == "window_compressed":
            slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"),
                                   "compress"))
        if layer.indexer:
            slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
            slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk",
                                   "index_select"))
        slots.append(LayerSlot("A2", "", "attention"))
    if attention_kind not in {"A1", "A2"} and (layer.compress_ratio or layer.cache_kind == "window_compressed"):
        slots.append(LayerSlot("K1", str(layer.compressor_kind or "weighted_pool"), "compress"))
    if attention_kind not in {"A0", "A1", "A2"} and layer.indexer:
        slots.append(LayerSlot("I0", str(layer.indexer_type or "full"), "index_produce"))
        slots.append(LayerSlot("I1", "candidate" if layer.candidate_filter else "topk", "index_select"))
    boundary("attn_post")
    slots.append(LayerSlot("N0", "residual", "attn_residual"))
    slots.append(LayerSlot("N0", "layernorm" if layer.ffn_activation == "gelu" and False
                           else "rmsnorm", "ffn_norm"))
    boundary("ffn_pre")
    if layer.ffn == "moe":
        slots.append(LayerSlot("F1", str(layer.router_mode or "softmax"), "router"))
        slots.append(LayerSlot("F2", features.moe_recipe, "transport"))
        slots.append(LayerSlot("F3", "latent" if layer.expert_latent_size else "packed",
                               "routed_ffn"))
        slots.append(LayerSlot("F2", features.moe_recipe, "transport_combine"))
        if layer.num_shared_experts:
            slots.append(LayerSlot("F4", "gated" if layer.shared_expert_gate else "plain",
                                   "shared_ffn"))
            slots.append(LayerSlot("F4", "join", "join"))
    else:
        activation = str(layer.ffn_activation or "silu")
        if activation not in {"silu", "gelu", "gelu_tanh", "situ", "ungated"}:
            activation = "silu"
        slots.append(LayerSlot("F0", activation, "dense_ffn"))
    boundary("ffn_post")
    if layer.engram:
        slots.append(LayerSlot("E1", "engram", "engram"))
    elif _layer_has_ple(model, layer):
        slots.append(LayerSlot("E1", "ple", "ple"))
    return tuple(slots)

def instantiate_layer(model: ModelSpec, layer: LayerPlan,
                      features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None,
                      layers: int | None = None,
                      stage: int | None = None) -> LayerInstances:
    total = int(layers if layers is not None else len(model.layer_plans) or model.layers)
    _validate_parallel(model, features)
    stage_value = _stage_for(layer.index, total, features.pp) if stage is None else int(stage)
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    slots = _layer_slots(model, layer, features)
    validate_combination(slot.kind for slot in slots)
    subgraphs: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []
    for slot in slots:
        ctx = _Ctx(model=model, layer=layer, features=features, stage=stage_value,
                   role=slot.role, variant=slot.variant, layers=total)
        built = build(slot.kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        instance = SubgraphInstance(
            instance_id=f"{slot.kind}:{template.variant}:L{layer.index}:{slot.role}",
            template=template, layer=layer.index, stage=stage_value,
            operators=bound, state_in=template.state_in,
            state_out=template.state_out,
            owner_layer=(int(layer.kv_owner_layer_id)
                         if 0 <= layer.kv_owner_layer_id < layer.index else None),
            metadata=(("role", slot.role), ("variant", slot.variant),
                      ("attention_kind", "A1" if layer.attention == "mla" else layer.attention)),
        )
        subgraphs.append(instance)
        if template.coverage != "structural":
            holes.append(CoverageHole(slot.kind, "declared_partial", template.title))
    return LayerInstances(layer.index, stage_value, tuple(subgraphs), slots, tuple(holes))

_DEFAULT_REGISTRY = TemplateRegistry()

def _has_ple(model: ModelSpec) -> bool:
    facts = dict(model.catalog_facts)
    return bool(facts.get("ple") or dict(model.metadata).get("ple") or facts.get("gemma_ple"))

def instantiate_model(model: ModelSpec, features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None) -> InstantiatedModel:
    if not model.layer_plans:
        raise ModelingError("instantiate_requires_explicit_layer_plans")
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    total = len(model.layer_plans)
    layers = tuple(
        instantiate_layer(model, layer, features, registry=registry,
                          layers=total, stage=_stage_for(layer.index, total, features.pp))
        for layer in model.layer_plans)
    globals_: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []

    def add_global(kind: str, variant: str, role: str, stage: int) -> None:
        ctx = _Ctx(model=model, layer=model.layer_plans[0], features=features,
                   stage=min(features.pp - 1, max(0, stage)), role=role,
                   variant=variant, layers=total)
        built = build(kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        globals_.append(SubgraphInstance(
            instance_id=f"{kind}:{template.variant}:G:{role}", template=template,
            layer=0, stage=min(features.pp - 1, max(0, stage)), operators=bound,
            state_in=template.state_in, state_out=template.state_out,
            metadata=(("role", role), ("global", True))))
        if template.coverage != "structural":
            holes.append(CoverageHole(kind, "declared_partial", template.title))

    add_global("E0", "media" if model.vision is not None else "token", "input", 0)
    if model.vision is not None:
        add_global("V0", "vision", "media", 0)
    if model.has_optional_draft or model.is_standalone_draft:
        draft_var = ("dspark" if model.auxiliary and model.auxiliary.dspark and not model.auxiliary.mtp
                     else ("mtp" if model.has_optional_draft else "standalone"))
        add_global("D0", draft_var, "draft", features.pp - 1)
    add_global("O0", features.final_norm_scope, "output", features.pp - 1)
    entries = []
    for layer in layers:
        for instance in layer.subgraphs:
            entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    for instance in globals_:
        entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    holes.extend(layer_hole for layer in layers for layer_hole in layer.holes)
    if _has_ple(model) and not any(layer_slot.kind == "E1" for layer in layers
                                   for layer_slot in layer.slots):
        if not dict(model.catalog_facts).get("gemma4"):
            holes.append(CoverageHole("E1", "ple_not_instantiated",
                                      "PLE declared without an explicit per-layer table record"))
    coverage = CoverageReport(tuple(entries), tuple(holes))
    return InstantiatedModel(model=model, features=features, layers=layers, globals=tuple(globals_),
                             coverage=coverage, stage_of=tuple(layer.stage for layer in layers))

def instantiate(model: ModelSpec, features: FeatureConfig | None = None, *,
                runtime: Any = None, registry: TemplateRegistry | None = None) -> InstantiatedModel:
    return instantiate_model(model, features or FeatureConfig.from_runtime(runtime, model), registry=registry)
