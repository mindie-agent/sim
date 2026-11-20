"""Compact explicit model, hardware, workload, and Case builders for tests.

[Role] Provide two-to-four-layer explicit models, in-memory hardware, and
  deterministic workload records so domain tests never re-implement ad-hoc
  ``_case`` / ``_hardware`` builders or load the full catalog unless testing it.
[Invariants] Every builder returns fresh immutable schema records; no catalog
  scan, file I/O, or device access occurs during construction.
[Timeouts & Locks] Pure in-memory construction; no I/O, sleeps, or locks.
[Side-Effect Policy] Side-effect-free; zero global mutation.
"""
from __future__ import annotations

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.hardware import Device, Hardware, Link
from mindie_sim.schema.model import LayerPlan, ModelSpec, RequestSpec, WorkloadSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

DEFAULT_PEAKS = (("bf16", 1.0e12), ("fp32", 5.0e11), ("fp8_e4m3", 2.0e12))
DEFAULT_RATES = (("hbm", 1.0e12), ("memory", 1.0e12), ("vector", 1.0e12))
DEFAULT_MEMORY_BYTES = 1 << 30
DEFAULT_UB_BYTES = 1024
DEFAULT_FABRIC_BYTES_PER_MS = 1.0e6


def device(
    rank: int,
    *,
    node: int = 0,
    peaks=DEFAULT_PEAKS,
    rates=DEFAULT_RATES,
    compute_units: int = 4,
    vector_units: int = 8,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    package: int = 0,
) -> Device:
    return Device(
        rank=rank,
        memory_bytes=int(memory_bytes),
        compute_peaks=tuple(peaks),
        rates=tuple(rates),
        compute_units=int(compute_units),
        node=int(node),
        package=int(package),
        resource_units=(("vector", int(vector_units)),),
        buffer_units=(("ub", int(vector_units)),),
        buffers=(("ub", DEFAULT_UB_BYTES),),
        launch_us=0.0,
    )


def hardware(
    ranks: int = 2,
    *,
    ranks_per_node: int | None = None,
    peaks=DEFAULT_PEAKS,
    rates=DEFAULT_RATES,
    compute_units: int = 4,
    vector_units: int = 8,
) -> Hardware:
    width = ranks if ranks_per_node is None else int(ranks_per_node)
    dies = tuple(
        device(
            rank,
            node=rank // width,
            peaks=peaks,
            rates=rates,
            compute_units=compute_units,
            vector_units=vector_units,
        )
        for rank in range(int(ranks))
    )
    link = Link("fabric", DEFAULT_FABRIC_BYTES_PER_MS, tuple(range(int(ranks))), latency_us=0.0)
    return Hardware("tiny", dies, (link,), ranks_per_node=width)


def dense_layer(index: int, **overrides) -> LayerPlan:
    values = dict(
        index=index,
        attention="gqa",
        ffn="dense",
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        kv_cache_elements=32,
        kv_owner_layer_id=index,
        owns_long_kv=True,
    )
    values.update(overrides)
    return LayerPlan(**values)


def dense_model(layers: int = 2, **overrides) -> ModelSpec:
    values = dict(
        name="compact-dense",
        layers=layers,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        attention="gqa",
        ffn="dense",
        layer_plans=tuple(dense_layer(index) for index in range(layers)),
        quant=QuantSpec("bf16"),
    )
    values.update(overrides)
    return ModelSpec(**values)


def dense_case(
    *,
    layers: int = 2,
    dp: int = 1,
    tp: int = 1,
    ep: int = 1,
    ranks_per_node: int | None = None,
    **runtime,
) -> Case:
    ranks = max(dp * tp, ep, 1)
    parallel = ParallelSpec(dp=dp, tp=tp, ep=ep)
    return Case(
        machine=hardware(ranks, ranks_per_node=ranks_per_node),
        model=dense_model(layers),
        workload=WorkloadSpec(tuple(RequestSpec(f"r{i}", 0.0, 2, 1) for i in range(max(1, dp)))),
        runtime=Software(parallel=parallel, **runtime),
    )


def linear_layer(index: int, kind: str = "kda", **overrides) -> LayerPlan:
    values = dict(
        index=index,
        attention=kind,
        ffn="dense",
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        linear_heads=4,
        linear_key_heads=4,
        linear_value_heads=4,
        linear_head_dim=16,
        linear_key_head_dim=16,
        conv_kernel=4,
        recurrent_state_elements=4 * 16 * 16,
        conv_state_elements=4 * 16 * 4,
        recurrent_state_dtype="fp32",
        conv_state_dtype="bf16",
        kv_cache_elements=0,
        kv_owner_layer_id=-1,
        owns_long_kv=False,
    )
    values.update(overrides)
    return LayerPlan(**values)


def linear_model(kind: str = "kda", layers: int = 2) -> ModelSpec:
    return ModelSpec(
        name=f"compact-{kind}",
        layers=layers,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        attention=kind,
        ffn="dense",
        layer_plans=tuple(linear_layer(index, kind) for index in range(layers)),
        quant=QuantSpec("bf16"),
    )


def plan(*views: RequestView, iteration: int = 0) -> IterationPlan:
    return IterationPlan(tuple(views), iteration=iteration)


def view(
    request_id: str = "r0",
    q_len: int = 1,
    context: int = 0,
    *,
    prefix: int = 0,
    advanced: int = 0,
    proposal: int = 0,
    padding: bool = False,
) -> RequestView:
    return RequestView(
        request_id,
        int(q_len),
        int(context),
        True,
        "decode",
        prefix_tokens=int(prefix),
        padding=bool(padding),
        draft_advanced_tokens=int(advanced),
        draft_proposal_tokens=int(proposal),
        prompt_tokens=int(context),
    )


def quant(**overrides) -> QuantSpec:
    values = dict(name="probe", weight_format="bf16", activation_format="bf16")
    values.update(overrides)
    return QuantSpec(**values)
