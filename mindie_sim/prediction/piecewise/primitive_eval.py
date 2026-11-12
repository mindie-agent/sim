"""Evaluate one standalone primitive specification as piecewise physical work.

[Role]
  Bind a declarative primitive spec (shape, format, parallelism, request
  geometry) to the same guarded family compilers used by the full subgraph
  path, then report per-operator time grouped by mechanism and core.

[Invariants]
  * Shapes and formats are structural inputs; no model name, measurement or
    fitted latency selects a branch.
  * Every returned operator is compiled by a registered family compiler; an
    unknown primitive raises instead of falling back to a generic cost.
  * ``operator_ms`` is the exact sum of the compiled operator durations, and
    the mechanism/core aggregates conserve that sum.
  * Request rows conserve ``tokens`` when a total row count is supplied; the
    last token of a causal decode sees ``kv_len`` context.
  * No import-time I/O, device context, weight load or global mutation.

[Timeouts & Locks]
  Pure bounded in-memory arithmetic; no waits, locks or remote work.

[Side-Effect Policy]
  Returns plain immutable-derived dictionaries; mutates no files or globals.
"""
from __future__ import annotations
from fractions import Fraction
from functools import partial
from typing import Any, Mapping
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature
from .compiler import AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply
from .geometry import RequestGeometry, WorkloadGeometry

_ALIASES = {
    "gemm": "gemm", "matmul": "gemm", "linear": "gemm",
    "grouped_matmul": "grouped_matmul", "grouped_gemm": "grouped_matmul", "gmm": "grouped_matmul",
    "moe_native": "moe_native", "moe": "moe_native", "moe_ffn": "moe_native",
    "fia": "fia", "gqa_attention": "fia", "gqa": "fia", "attention": "fia", "flash_attention": "fia",
    "mla_attention": "mla_attention", "mla": "mla_attention",
    "gdn": "gdn", "gated_delta": "gdn", "gated_delta_net": "gdn", "kda": "gdn", "kda_recurrent": "gdn",
    "conv1d": "conv1d", "conv": "conv1d", "causal_conv1d": "conv1d", "short_conv": "conv1d",
    "rmsnorm": "rmsnorm", "rms_norm": "rmsnorm", "layernorm": "rmsnorm", "norm": "rmsnorm",
    "swiglu": "swiglu", "geglu": "swiglu", "glu": "swiglu", "silu": "silu", "gelu": "silu",
    "activation": "silu", "rotary": "rotary", "rope": "rotary", "rotary_embedding": "rotary",
    "quant": "quant", "quantize": "quant", "dynamic_quant": "quant",
    "collective": "collective", "allreduce": "collective", "all_reduce": "collective",
    "reduce_scatter": "collective", "all_gather": "collective", "alltoall": "collective",
    "all_to_all": "collective", "etp_allreduce": "collective",
}
_SELECTORS = ("op", "operator", "primitive", "kind", "family", "type", "name")
_QUANT_DTYPES = {"fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "mxfp8": "fp8_e4m3",
                 "mxfp8_e4m3": "fp8_e4m3", "w8a8": "int8", "int8": "int8"}

def _pick(spec, keys, default=None):
    for key in keys:
        if spec.get(key) is not None:
            return spec[key]
    return default

def _num(value, label):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ModelingError(f"primitive_parameter_invalid:{label}:{value}") from None

def _int(spec, keys, default):
    return max(1, _num(_pick(spec, keys, default), keys[0]))

def _zero(spec, keys, default):
    return max(0, _num(_pick(spec, keys, default), keys[0]))

def _real(spec, keys, default):
    raw = _pick(spec, keys, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ModelingError(f"primitive_parameter_invalid:{keys[0]}:{raw}") from None

def _fmt(spec, keys, default):
    return str(_pick(spec, keys, default) or default).strip().lower()

def _width(fmt, default=2.0):
    try:
        return float(dtype_bytes(fmt))
    except ModelingError:
        return default

def _bind(name, family, role, *, matrix=(), heads=(), experts=(), quant=(),
          parallel=(), tiles=(), extras=()):
    return BoundOperator(
        OperatorTemplate(name=name, family=family, role=role),
        StaticSignature(role=name, matrix=matrix, heads=heads, experts=experts,
                        quant=quant, parallel=parallel, tiles=tiles, extras=extras))

def _gemm(spec, aic):
    k = _int(spec, ("in_features", "hidden_size", "hidden", "K"), 4096)
    n = _int(spec, ("out_features", "intermediate_size", "N"), 4096)
    wfmt, afmt = _fmt(spec, ("weight_format", "w_dtype", "dtype"), "bf16"), _fmt(spec, ("act_format", "a_dtype"), "bf16")
    ofmt, act = _fmt(spec, ("output_format", "out_dtype"), "bf16"), _real(spec, ("act_bytes",), _width(afmt))
    op = _bind("gemm", "gemm", "gemm", matrix=(("K", k), ("N", n)),
               quant=(("act_bytes", act), ("act_format", afmt), ("weight_format", wfmt), ("output_format", ofmt)),
               tiles=(("cores", aic),))
    return ((op, "E1", "gemm", "AI_CORE"),)

def _grouped(spec, aic):
    k, n = _int(spec, ("K", "in_features", "hidden_size", "hidden"), 4096), _int(spec, ("N", "out_features", "intermediate_size"), 2048)
    experts, ep = _int(spec, ("E", "experts", "num_experts"), 64), _int(spec, ("ep",), 1)
    local = _int(spec, ("local", "local_experts", "E_local"), max(1, experts // ep))
    top_k = _int(spec, ("top_k", "k", "topk"), 8)
    op = _bind("grouped_matmul", "grouped_gemm", "grouped_gemm", matrix=(("K", k), ("N", n)),
               experts=(("E", experts), ("local", local), ("k", top_k)),
               quant=(("weight_format", _fmt(spec, ("weight_format", "w_dtype"), "bf16")),),
               tiles=(("cores", aic),))
    return ((op, "F3", "moe_grouped", "MIX_AIC"),)

def _moe(spec, aic):
    hidden = _int(spec, ("hidden", "hidden_size", "K"), 4096)
    inter = _int(spec, ("inter", "moe_intermediate_size", "intermediate_size", "N"), 1536)
    experts, top_k = _int(spec, ("E", "experts", "num_experts"), 64), _int(spec, ("top_k", "k", "topk"), 8)
    tp, ep = _int(spec, ("tp",), 1), _int(spec, ("ep",), 1)
    etp = _int(spec, ("etp",), tp if ep <= 1 else 1)
    wfmt, act = _fmt(spec, ("weight_format", "w_dtype"), "bf16"), _real(spec, ("act_bytes",), 2.0)
    local, width, mid, wire = max(1, experts // ep), max(1, 2 * inter // etp), max(1, inter // etp), hidden * act
    items = [
        (_bind("router_gemm", "gemm", "router_gemm", matrix=(("K", hidden), ("N", experts)),
               quant=(("act_bytes", act), ("weight_format", "bf16")), tiles=(("cores", aic),)),
         "F1", "gemm", "AI_CORE"),
        (_bind("router", "router", "router", experts=(("E", experts), ("k", top_k), ("elements", experts)),
               tiles=(("cores", aic),)), "F1", "moe_router", "AI_VECTOR_CORE"),
    ]
    if ep > 1:
        for name in ("dispatch", "combine"):
            items.append((_bind(name, "transport", name, parallel=(("ep", ep), ("members", ep)),
                                matrix=(("row_bytes", wire),), experts=(("top_k", top_k),),
                                quant=(("wire_bytes", wire),)), "F2", "collective", "COMMUNICATION"))
    items.extend((
        (_bind("expert_gate_up", "grouped_gemm", "expert_gate_up", matrix=(("K", hidden), ("N", width)),
               experts=(("E", experts), ("local", local), ("k", top_k)),
               quant=(("weight_format", wfmt),), tiles=(("cores", aic),)), "F3", "moe_grouped", "MIX_AIC"),
        (_bind("swiglu", "elementwise", "swiglu", matrix=(("elements", mid),),
               quant=(("act_bytes", act),), experts=(("E", experts), ("local", local), ("k", top_k)),
               extras=(("read_bytes", 2 * mid * act), ("write_bytes", mid * act), ("ops_per_element", 6))),
         "F3", "activation", "AI_VECTOR_CORE"),
        (_bind("expert_down", "grouped_gemm", "expert_down", matrix=(("K", mid), ("N", hidden)),
               experts=(("E", experts), ("local", local), ("k", top_k)),
               quant=(("weight_format", wfmt),), tiles=(("cores", aic),)), "F3", "moe_grouped", "MIX_AIC"),
    ))
    if etp > 1:
        items.append((_bind("etp_allreduce", "collective", "etp_allreduce",
                            parallel=(("axis", "etp"), ("members", etp), ("etp", etp), ("tp", tp), ("ep", ep)),
                            experts=(("E", experts), ("local", local), ("k", top_k)),
                            quant=(("wire_bytes", wire), ("bytes_per_token", wire))),
                      "F3", "collective", "COMMUNICATION"))
    return tuple(items)

def _attention(spec, aic, mla):
    q_heads = _int(spec, ("q_heads", "q", "heads", "num_attention_heads"), 32)
    act = _real(spec, ("act_bytes",), 2.0)
    if mla:
        kv_heads, dn, dr = _int(spec, ("kv_heads", "kv"), 1), _int(spec, ("dn", "qk_nope_head_dim"), 128), _int(spec, ("dr", "qk_rope_head_dim"), 64)
        rkv, val_w = _int(spec, ("rkv", "kv_lora_rank"), 512), _int(spec, ("val_width", "dv", "v_head_dim"), 128)
        op = _bind("mla_attention", "mla", "mla_attention",
                   heads=(("q", q_heads), ("kv", kv_heads), ("dn", dn), ("dr", dr), ("rkv", rkv), ("dv", val_w)),
                   quant=(("act_bytes", act),), tiles=(("mt", 128), ("nt", 128), ("cores", aic)),
                   extras=(("absorb", _zero(spec, ("absorb",), 1)), ("index_topk", _zero(spec, ("index_topk",), 0))))
        return ((op, "A1", "attention", "MIX_AIC"),)
    tp = _int(spec, ("tp",), 1)
    kv_heads = _int(spec, ("kv_heads", "kv", "num_key_value_heads"), 8)
    global_q, global_kv = _pick(spec, ("global_q_heads",), None), _pick(spec, ("global_kv_heads",), None)
    if global_q is not None:
        q_heads = max(1, -(-_num(global_q, "global_q_heads") // tp))
    if global_kv is not None:
        kv_heads = max(1, -(-_num(global_kv, "global_kv_heads") // tp))
    key_w = _int(spec, ("key_width", "dk", "head_dim"), 128)
    val_w = _int(spec, ("val_width", "dv", "v_head_dim"), key_w)
    kv_elem = _real(spec, ("kv_elem_bytes",), _width(_fmt(spec, ("kv_dtype",), "bf16"), act))
    op = _bind("fia", "attention", "attention",
               heads=(("q", q_heads), ("kv", kv_heads), ("dk", key_w), ("dv", val_w)),
               quant=(("act_bytes", act), ("kv_bytes", kv_heads * (key_w + val_w) * kv_elem)),
               tiles=(("mt", 128), ("nt", 128), ("cores", aic)),
               extras=(("window", _zero(spec, ("window",), 0)), ("causal", _zero(spec, ("causal",), 1))))
    return ((op, "A0", "attention", "MIX_AIC"),)

def _ew(spec, name, family, role, kind, mechanism, *, read, write, ops, quantized=False):
    elements = _int(spec, ("elements", "intermediate_size", "hidden", "hidden_size", "width", "dim", "K"), 4096)
    act = _real(spec, ("act_bytes",), 2.0)
    group = (("act_bytes", act),)
    if quantized:
        group += (("quantized", 1), ("output_format", _fmt(spec, ("output_format", "out_dtype", "quant_dtype", "dst_dtype"), "fp8_e4m3")))
    op = _bind(name, family, role, matrix=(("elements", elements),), quant=group,
               extras=(("ops_per_element", _int(spec, ("ops_per_element",), ops)),
                       ("read_bytes", read * elements * act), ("write_bytes", write * elements * act)))
    return ((op, kind, mechanism, "AI_VECTOR_CORE"),)

def _norm(spec, aic):
    factor = 2 if _zero(spec, ("residual",), 0) else 1
    return _ew(spec, "rmsnorm", "elementwise", "norm", "N0", "norm_residual", read=factor, write=factor, ops=6)

def _swiglu(spec, aic):
    return _ew(spec, "swiglu", "elementwise", "activation", "F0", "activation", read=2, write=1, ops=6)

def _silu(spec, aic):
    return _ew(spec, "silu", "elementwise", "activation", "F0", "activation", read=1, write=1, ops=5)

def _quant(spec, aic):
    return _ew(spec, "dynamic_quant", "quantize", "quantize", "N0", "elementwise_quant", read=1, write=1, ops=5, quantized=True)

def _rotary(spec, aic):
    heads = _pick(spec, ("q_heads", "heads"), None)
    if heads is None:
        return _ew(spec, "rotary", "elementwise", "rope", "A0", "rope", read=1, write=1, ops=6)
    key_w = _int(spec, ("key_width", "dk", "head_dim"), 128)
    elements = (max(1, _num(heads, "q_heads")) + _int(spec, ("kv_heads", "kv"), 8)) * key_w
    act = _real(spec, ("act_bytes",), 2.0)
    op = _bind("rotary", "elementwise", "rope", matrix=(("elements", elements),),
               quant=(("act_bytes", act),),
               extras=(("ops_per_element", _int(spec, ("ops_per_element",), 6)),))
    return ((op, "A0", "rope", "AI_VECTOR_CORE"),)

def _gdn(spec, aic):
    key_heads, val_heads = _int(spec, ("key_heads", "k_heads", "kv_heads"), 16), _int(spec, ("value_heads", "v_heads", "heads"), 32)
    key_w = _int(spec, ("key_width", "dk", "head_k_dim", "head_dim"), 128)
    val_w = _int(spec, ("val_width", "dv", "head_v_dim"), key_w)
    op = _bind("gated_delta", "recurrent", "gated_delta",
               heads=(("key_heads", key_heads), ("value_heads", val_heads), ("dk", key_w), ("dv", val_w),
                      ("chunk", _int(spec, ("chunk", "chunk_size"), 64)), ("gate_rank", _zero(spec, ("gate_rank",), 0))),
               quant=(("act_bytes", _real(spec, ("act_bytes",), 2.0)),))
    return ((op, "A3", "gdn_kda", ""),)

def _conv(spec, aic):
    channels = _int(spec, ("channels", "width", "dim", "hidden_size", "hidden", "qkv_width"), 4096)
    kernel = _int(spec, ("kernel", "kernel_size", "d_conv"), 4)
    act = _real(spec, ("act_bytes",), 2.0)
    op = _bind("conv", "recurrent", "conv", matrix=(("row_bytes", channels * act),),
               quant=(("act_bytes", act),),
               extras=(("kernel", kernel), ("rows", max(1, kernel - 1)), ("history_rows", kernel - 1),
                       ("qkv_width", channels)))
    return ((op, "A3", "gdn_kda", "AI_VECTOR_CORE"),)

def _collective(spec, aic, key, devices):
    name = str(spec.get("name") or spec.get("collective") or key).strip().lower().replace("-", "_")
    ranks = max(1, _num(_pick(spec, ("members", "ranks", "tp", "ep", "world_size"), devices), "members"))
    rpn = max(1, min(ranks, _num(_pick(spec, ("ranks_per_node",), min(ranks, 8)), "ranks_per_node")))
    hidden = _int(spec, ("hidden", "elements", "hidden_size", "width"), 4096)
    act = _real(spec, ("act_bytes",), 2.0)
    wire = _real(spec, ("wire_bytes", "bytes_per_token"), hidden * act)
    group = []
    for keys, label in ((("E", "experts"), "E"), (("local", "local_experts"), "local"), (("k", "top_k"), "k")):
        raw = _pick(spec, keys, None)
        if raw is not None:
            group.append((label, _num(raw, label)))
    kind = "F3" if "etp" in name else ("F2" if "alltoall" in name else "R1")
    op = _bind(name, "collective", name, parallel=(("members", ranks), ("ranks", ranks), ("ranks_per_node", rpn)),
               experts=tuple(group), quant=(("wire_bytes", wire), ("bytes_per_token", wire), ("act_bytes", act)),
               extras=(("collective", name),))
    return ((op, kind, "collective", "COMMUNICATION"),)

_BUILDERS = {"gemm": _gemm, "grouped_matmul": _grouped, "moe_native": _moe,
             "mla_attention": partial(_attention, mla=True), "fia": partial(_attention, mla=False),
             "gdn": _gdn, "conv1d": _conv, "rmsnorm": _norm, "swiglu": _swiglu,
             "silu": _silu, "rotary": _rotary, "quant": _quant}

def _compute_dtype(spec, hw):
    fmt = str(_pick(spec, ("act_format", "a_dtype", "activation_format", "compute_format", "act_dtype"), "bf16") or "bf16")
    candidate = _QUANT_DTYPES.get(fmt.strip().lower())
    if candidate is None:
        return "bf16"
    try:
        hw.dies[0].peak(candidate)
    except ModelingError:
        return "bf16"
    return candidate

def _request_shape(spec):
    batch_n = _int(spec, ("batch", "batch_size", "spr"), 1)
    rows, query = _pick(spec, ("M", "m", "tokens", "num_tokens"), None), _pick(spec, ("q_len", "seq_len", "tps", "prompt_tokens"), None)
    cache = _pick(spec, ("kv_len", "s_kv", "context"), None)
    cache_n = max(1, _num(cache, "kv_len")) if cache is not None else None
    if query is not None:
        query_n = max(1, _num(query, "q_len"))
        return (query_n,) * batch_n, query_n, cache_n
    if rows is not None:
        base, rem = divmod(max(1, _num(rows, "tokens")), batch_n)
        lens = tuple(base + (1 if index < rem else 0) for index in range(batch_n))
        return lens, max(lens), cache_n
    query_n = 1 if cache_n is not None else 128
    return (query_n,) * batch_n, query_n, cache_n

def _phase(spec, query_n, total_kv, cache_n):
    declared = spec.get("phase")
    if declared is not None and str(declared).strip():
        return str(declared).strip().lower()
    if (query_n <= 4 and total_kv > query_n) or (query_n == 1 and cache_n is not None):
        return "decode"
    return "prefill"

def evaluate_primitive_spec(spec: Mapping[str, Any], *, sku: str = "Ascend950DT_9582",
                            num_devices: int = 8) -> dict[str, Any]:
    """Compile and evaluate one standalone primitive specification."""
    if not isinstance(spec, Mapping):
        raise ModelingError("primitive_spec_mapping_required")
    key = next((str(spec[item]).strip().lower().replace("-", "_")
                for item in _SELECTORS if spec.get(item) not in (None, "")), "")
    if not key:
        raise ModelingError("primitive_selector_missing")
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ModelingError(f"primitive_unknown:{key}")
    from mindie_sim.compose import resolve_hardware_sku
    devices = max(1, _num(_pick(spec, ("num_devices",), num_devices), "num_devices"))
    hw = Hardware.from_mapping({"backend": "ascend",
                                "sku": resolve_hardware_sku(str(spec.get("sku") or sku)),
                                "num_devices": devices})
    supply = PipeSupply.from_hardware(hw, dtype=_compute_dtype(spec, hw))
    aic = max(1, int(hw.dies[0].compute_units))
    domain = AnalysisDomain(max_tokens=10**8, max_moe_rows=10**8, max_query_tiles=10**8,
                            max_kv_tiles=10**8, max_index_rows=10**11, max_work=10**15,
                            max_q=10**8, max_s_kv=10**8, max_seq=10**8, max_batch=10**6)
    compiler = OperatorPiecewiseCompiler(supply, ImplementationPolicy(cores=aic), domain=domain)
    lengths, query_n, cache_n = _request_shape(spec)
    total_kv = max(max(lengths), cache_n if cache_n is not None else query_n)
    phase = _phase(spec, query_n, total_kv, cache_n)
    geometry = WorkloadGeometry(tuple(RequestGeometry(length, total_kv - length, phase=phase)
                                       for length in lengths), phase=phase)
    items = _collective(spec, aic, key, devices) if canonical == "collective" else _BUILDERS[canonical](spec, aic)
    total = Fraction(0)
    by_mechanism, by_core, details = {}, {}, []
    for operator, kind, mechanism, core in items:
        cost = compiler.compile_operator(operator, kind=kind).evaluate(geometry)
        if not core:
            core = "MIX_AIC" if float(dict(cost.work).get("cube", 0.0)) > 0 else "AI_VECTOR_CORE"
        total += cost.duration_ms
        by_mechanism[mechanism] = by_mechanism.get(mechanism, Fraction(0)) + cost.duration_ms
        by_core[core] = by_core.get(core, Fraction(0)) + cost.duration_ms
        details.append({"name": cost.name, "family": cost.family, "kind": cost.kind,
                        "mechanism": mechanism, "core": core, "operator_ms": float(cost.duration_ms),
                        "regime": dict(cost.regime),
                        "work": {pipe: float(value) for pipe, value in cost.work},
                        "derived": {name: float(value) for name, value in cost.derived}})
    operator_ms = float(total)
    repeat_n = max(1, _num(_pick(spec, ("repeat", "num_layers"), 1), "repeat"))
    return {"step_ms": operator_ms * repeat_n, "operator_ms": operator_ms,
            "by_mechanism_ms": {name: float(value) for name, value in by_mechanism.items()},
            "by_core_ms": {name: float(value) for name, value in by_core.items()},
            "sku": hw.sku, "num_devices": devices, "primitive": canonical, "operators": details}
