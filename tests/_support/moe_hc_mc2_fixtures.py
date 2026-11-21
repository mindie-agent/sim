"""Shared controls for MoE routing, grouped operators, and physical owners.

[Role] Build explicit routing, transport, MC2, HyperConnections, AttnRes,
  grouped dequantization, and router Split-K cases for analytical tests.
[Invariants] Case builders produce fresh inputs with the original routing,
  quantization, and owner geometry. Graph and program helpers use the same
  lowering and resource bindings as their callers.
[Timeouts & Locks] Synchronous local reads and in-memory construction. No locks.
[Side-Effect Policy] Read repository examples only. No writes or device access.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

from mindie_sim import target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.operators.hyper_connections.native_fusion_hc_regbase import POST_PROFILE, PRE_PROFILE
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.moe_routing.moe_init_mxfp8 import RECIPE
from mindie_sim.prediction.rewrite_ffn_rules import router_gemm
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext
from mindie_sim.prediction.structure import parse_case


_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_MXFP8_CASES = _EXAMPLES / "cases/high-concurrency-20261030/glm-flash-ws9"
_GROUPED_DEQUANT_CASES = _EXAMPLES / "cases/high-concurrency-20261030/qwen35-hccl-conditional"

_ROUTE_TOKENS = 32
_ROUTE_TOP_K = 2
_ROUTE_EXPERTS = 8
_ROUTE_EP = 4
_EXPERTS_PER_RANK = _ROUTE_EXPERTS // _ROUTE_EP
_ROUTE_ROWS = ((0, 1), (2, 3), (4, 5), (6, 7),
               (0, 2), (1, 3), (4, 6), (5, 7))

SELECTOR = {
    "kind": "separate_quantized_payload", "payload_dtype": "int8",
    "scale_dtype": "fp32", "scale_granularity": "row", "histogram_dtype": "int64",
}


def _raw_base_case(*, dp=1, tp=1, pp=1, ep=1, prompt=4, output=3, requests=None):
    ranks = dp * tp * pp
    return {
        "machine": {
            "name": "declared-gpu", "architecture": "declared", "num_devices": ranks,
            "device": {"memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                       "rates": {"memory": 1e9, "vector": 1e9}, "compute_units": 4},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": list(range(ranks)), "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 2, "hidden_size": 32, "num_heads": 4,
            "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": requests or [
            {"id": "r", "prompt_tokens": prompt, "output_tokens": output}]},
        "runtime": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "max_num_batched_tokens": 16},
    }


def explicit_route_case(*, tokens=_ROUTE_TOKENS, quantized=False, route=None, balanced=False):
    raw = _raw_base_case(tp=_ROUTE_EP, ep=_ROUTE_EP, prompt=tokens, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=_ROUTE_EXPERTS, top_k=_ROUTE_TOP_K,
        hidden_size=32, moe_intermediate_size=32,
    )
    raw["machine"]["device"]["compute_peaks"].setdefault("int8", 1e9)
    raw["runtime"]["max_num_batched_tokens"] = max(16, tokens)
    if quantized:
        raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
        raw["runtime"]["kernels"] = {"moe_transport": {
            "kind": "separate_quantized_payload", "payload_dtype": "int8",
            "scale_dtype": "fp32", "scale_granularity": "row", "histogram_dtype": "int64",
        }}
    if balanced:
        raw["runtime"]["routing"] = {"policy": "balanced"}
    elif route is not None:
        owner_counts, cohort = route
        ids, owners = [], []
        for source, count in enumerate(owner_counts):
            for index in range(count):
                ids.append(_ROUTE_ROWS[(index + source) % len(_ROUTE_ROWS)])
                owners.append(source)
        placement = tuple(divmod(expert, _EXPERTS_PER_RANK) for expert in range(_ROUTE_EXPERTS))
        realization = {
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": len(ids),
            "topk_ids": ids, "expert_placement": placement, "source_owners": owners,
        }
        if cohort:
            realization["cohort"] = True
        raw["runtime"]["routing"] = {"realizations": [realization]}
    return raw


def mxfp8_case(batch, selected=True, fused=False):
    path = _MXFP8_CASES / f"a5_glm53flash_tp8_c{batch}_p4k.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    bindings = raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})
    if selected:
        bindings["moe_init_routing"] = RECIPE
    if fused:
        bindings["native_fusion"] = {
            "implementation": "declared_native_fusion",
            "options": {
                "roles": ["grouped_gate_up_activation_quant"],
                "program_profiles": {"grouped_gate_up_activation_quant": "ascend_mxfp8_grouped_swiglu_quant"},
                "program_options": {"grouped_gate_up_activation_quant": {"l1_k": 512}},
            },
        }
    return raw


def mxfp8_graph(raw, batch):
    return target_graphs(raw, scope="body", phase="decode", tokens=4,
                         context=4096, sequences=batch)[0]


def mc2_case(quant="mx", width=512):
    raw = _raw_base_case(dp=2, ep=2)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2,
        hidden_size=width, head_dim=width // 4,
        intermediate_size=width, moe_intermediate_size=256,
    )
    device = raw["machine"]["device"]
    device.update(
        rates={"memory": 1e9, "vector": 1e9, "mte2_aic": 1e9, "mte2_aiv": 1e9, "mte3": 1e9},
        resource_units={"vector": 4, "mte2_aiv": 4, "mte3": 4},
    )
    device["compute_peaks"]["fp8_e4m3"] = 1e9
    raw["machine"]["links"][0].update(semantics="per_pair", capacity_bytes_per_ms=1e6, latency_us=5)
    raw["runtime"]["kernels"] = {
        "families": {family: {"kind": "pairwise_collective"}
                     for family in ("alltoall", "expert_dispatch", "expert_combine")},
        "native_moe_protocol": {
            "kind": "arch35_fullmesh", "max_inflight_per_rank": 8, "dispatch_quant": quant,
        },
    }
    raw["runtime"]["routing"] = {"policy": "balanced"}
    parsed = parse_case(raw)
    quantization = QuantSpec(
        "explicit-mx32", weight_format="bf16", expert_weight_format="mxfp4",
        activation_format="fp8_e4m3", expert_block=32, weight_block=(32, 32),
        scale_format="fp8_e8m0", scale_bytes=1, activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        exclusions=("router", "norm"),
    )
    return replace(parsed, model=replace(parsed.model, quant=quantization))


def attnres_case(aiv, *, split=False, hidden=64):
    rates = {"memory": 1e30, "vector": 1e9, "mte3": 1e30}
    units = {"vector": aiv, "mte3": aiv}
    if split:
        rates.update(mte2_aic=1e30, mte2_aiv=1e30)
        units.update(mte2_aic=max(1, aiv // 2), mte2_aiv=aiv)
    else:
        rates["mte2"] = 1e30
        units["mte2"] = aiv
    return parse_case({
        "machine": {"num_devices": 1, "device": {
            "memory_bytes": 2**30, "compute_units": max(1, aiv // 2),
            "compute_peaks": {"bf16": 1e14}, "rates": rates,
            "resource_units": units, "buffers": {"ub": 196608},
            "buffer_units": {"ub": aiv}}},
        "model": {"name": "owner-control", "layers": 1, "hidden_size": hidden,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": hidden,
                  "intermediate_size": 2 * hidden, "vocab_size": 16,
                  "activation_dtype": "bf16"},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    })


def grouped_case(*, source=False):
    raw = _raw_base_case(prompt=4, output=3)
    raw["model"].update(layers=1, ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"]["routing"] = {"realizations": [
        {"scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
         "expert_rows": [3, 0, 3, 2]},
        {"scope": "body", "layer": 0, "phase": "decode", "tokens": 1,
         "expert_rows": [1, 0, 1, 0]},
    ]}
    if source:
        raw["runtime"]["kernels"] = {"lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}}
    return raw


def raw_case():
    path = _EXAMPLES / "source-families/v41-hc-source-control.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    options = raw["runtime"]["kernels"]["lowerings"]["native_fusion"]["options"]
    options["traffic_profile"] = {"hc_pre": PRE_PROFILE, "hc_post": POST_PROFILE}
    return raw


def balanced_case(*, tp=1, dp=1, ep=1, experts=8, tokens=32):
    raw = _raw_base_case(tp=tp, dp=dp, ep=ep, prompt=tokens, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=experts, top_k=2, moe_intermediate_size=64)
    raw["runtime"].update(routing={"policy": "balanced"}, max_num_batched_tokens=max(16, tokens))
    raw["runtime"]["kernels"] = {"families": {"alltoall": {"kind": "pairwise_collective"}}}
    return raw


def ordinary(batch=64, capacity=2048):
    """Return the parsed case, selected dequantization op, and body graph."""
    path = _GROUPED_DEQUANT_CASES / f"qwen35-35b-a5-tp4-c{batch}-hccl-source.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model"]["layers"] = 1
    kernels = raw["runtime"]["kernels"]
    kernels["native_moe_protocol"] = {
        "kind": "arch35_fullmesh", "max_inflight_per_rank": 4, "dispatch_quant": "per_token_int8",
    }
    kernels["moe_token_partition"] = {
        "kind": "vllm_ascend_mc2_tp_slice", "gather_output_materialization": "view",
    }
    kernels["lowerings"]["grouped_int8_caller"] = {
        "implementation": "ascend_grouped_int32_dequant_swiglu",
        "options": {"dequant_program": "arch35_full_load", "input_capacity_rows": capacity},
    }
    original = deepcopy(raw)
    ops = target_operators(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=batch)
    assert raw == original
    return parse_case(raw), next(op for op in ops if op.recipe == "grouped_int32_dequant_swiglu"), ops


def selected(rows, width=512, capacity=2048):
    """Return one grouped dequantization op with explicit rows and capacity."""
    case, op, _ = ordinary()
    config = {**dict(op.metadata)["grouped_dequant"], "expert_rows": tuple(rows),
              "width": width, "input_capacity_rows": capacity}
    op = replace(op, local_shape=(sum(rows), width), deps=(), metadata=tuple({
        **dict(op.metadata), "grouped_dequant": config,
    }.items()))
    return case, op


def transport_case(*, dp=2, tp=1, ep=2, tokens=4, selected=True):
    raw = _raw_base_case(dp=dp, tp=tp, ep=ep, prompt=tokens, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2, hidden_size=32,
        moe_intermediate_size=32, activation_dtype="bf16",
        quant=asdict(quant_preset("w8a8_int8")),
    )
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e9
    if selected:
        raw["runtime"]["kernels"] = {"moe_transport": dict(SELECTOR)}
    return raw


def _router_case(path, sku):
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["machine"]["sku"] = sku
    raw["machine"]["hardware_key"] = sku
    return raw


def _router(m, n, k):
    plan = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                           num_experts=n, router_input_size=k, index=0)
    model = SimpleNamespace(
        quant=QuantSpec(name="mx", weight_format="mxfp8", activation_format="mxfp8"),
        activation_dtype="bf16",
    )
    _, router, _, _ = router_gemm(
        prefix="p", plan=plan, deps=("x",), stage=0, requests=(), tokens=m,
        model=model, hidden=k, mode="decode",
    )
    return router


def _geometry(raw):
    return Geometry.from_devices(parse_case(raw).machine.dies)


def _splitk_context(raw, op):
    """Bind declared rates in work per millisecond, as in operator costing."""
    device = parse_case(raw).machine.dies[0]
    rates = {role: rate / 1000 for role, rate in dict(device.rates).items()}
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    units = {role: device.units(role) for role in rates}
    units["cube"] = device.compute_units
    return RecipeContext(
        op, {}, names, {names[role]: rates[role] for role in rates}, device.compute_units,
        resource_units=units,
        compute_peaks=dict(device.compute_peaks),
        service_modes={mode: (role, rate) for mode, (role, rate) in device.service_modes},
        pool_names={pool: f"{pool}@r0:s0" for pool, _ in device.buffers},
        pool_units={pool: device.pool_units(pool) for pool, _ in device.buffers},
        pool_capacities=dict(device.buffers),
        physical_mechanisms=device.physical_mechanisms,
    )


def _bound_context(raw, plan, op):
    context = _splitk_context(raw, plan.work.apply(op))
    return replace(context, work=context.bind(plan.work))


def _stages(program):
    """Yield finite work stages, including those in nested programs."""
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            yield stage
            if stage.program is not None:
                yield from _stages(stage.program)
    elif isinstance(program, PipelineProgram):
        for index in range(program.count):
            for stage in program.tile(index):
                yield stage
                if stage.program is not None:
                    yield from _stages(stage.program)
    elif isinstance(program, ParallelProgram):
        for branch in program.programs:
            yield from _stages(branch)


def _stage_work(program):
    issued = {}
    for stage in _stages(program):
        for name, amount in stage.work.items():
            issued[name] = issued.get(name, 0.0) + float(amount)
    return issued


router_splitk = SimpleNamespace(
    V41=_EXAMPLES / "cases/v41-a5-dp4-body-current.json",
    DT="Ascend950DT_9582",
    _case=_router_case,
    _router=_router,
    _geometry=_geometry,
    _bound_context=_bound_context,
    _stage_work=_stage_work,
)
