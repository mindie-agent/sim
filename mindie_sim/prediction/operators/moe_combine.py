"""Source-derived Triton MoE expert weighted-combination plan.

[Role]
  Translate ``moe_fused_mul_sum_kernel`` (vLLM v0.25.1 current checkout) into
  explicit tile ownership, FP32 accumulation work and masked tensor traffic.

[Invariants]
  * Only the no-expert-map branch is selected. ``topk_ids``/``expert_map`` are
    passed to the launcher but are never loaded by this branch.
  * The kernel applies ``topk_weights`` itself; ``apply_router_weight_on_input``
    must be false or the caller has already applied the same weights elsewhere.
  * Arithmetic spans every lane of a power-of-two tile even when row/column
    masks disable the corresponding load or store. Only valid lanes move bytes.
  * ``topk_weights`` is re-read once per K tile because each program owns one
    ``(pid_k, pid_m)`` tile. This is real repeated traffic, not a cache hit.
  * Accumulation is FP32 and output is the input dtype. No matrix/tensor-core
    unit is charged; the selected kernel performs elementwise CUDA-core work.
  * Every grid program is one CTA. The ``(pid_k, pid_m)`` grid is split
    round-robin across at most ``min(grid, compute_units)`` physical owners;
    a CTA's vector stage and, where the backend declares split copy engines,
    its mte2/mte3 stages are pinned to that one owner's units. Idle SMs
    therefore cannot serve a CTA the source assigned to one SM.
  * No routing, all-to-all, workspace round-trip or latency floor is added.

[Timeouts & Locks]
  O(1) aggregate arithmetic and at most ``min(grid, geometry.compute_units)``
  lazy per-SM CTA streams, each holding one resident tile. No per-token event
  lists.

[Side-Effect Policy]
  Pure metadata and program construction. Imports no device runtime and reads
  no measurement; the returned program submits only to the caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

#: Current checkout identity, not the historical capture image.  The campaign
#: manifest records these exact digests.
SOURCE_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/"
    "fused_moe/moe_fused_mul_sum.py"
)
SOURCE_SHA256 = "30fbb37b84b1cf6a8962d674abd9c3ab4d2be56d028fac82f55833c88d9729e8"
SOURCE_BYTES = 5999
SOURCE_LINES = (
    "kernel 10-63; mask 31-33; weight load 41; A load 50-55; "
    "fma 56; store 58-63; heuristic 66-132; wrapper 135-202; launch 177-200"
)
CONTRACT_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/"
    "fused_moe/topk_weight_and_reduce.py"
)
CONTRACT_SHA256 = "6a269547fc2bddb5e1784676f78d63a1b3e796699a2c9bfe2c8e446e95fd3425"
CONTRACT_BYTES = 6018
CONTRACT_LINES = "TopKWeightAndReduceContiguous 80-121; apply flag 107-108"
SOURCE_ORIGIN = "current_checkout_not_historical_capture_image"

IMPLEMENTATION = "triton_moe_fused_mul_sum"
KERNEL = "moe_fused_mul_sum_kernel"
RECIPE = "triton_moe_fused_mul_sum"
COMPUTE_CAPABILITY = 89

_SUPPORTED_DTYPES = frozenset({"bf16", "fp16", "fp32"})
_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
}
_REQUIRED = (
    "tokens", "hidden", "topk", "dtype", "weight_dtype",
    "compute_capability", "implementation",
    "apply_router_weight_on_input", "has_expert_map",
)

#: Semantic scalar ledger.  These are source-level operations, not a claim
#: about the compiled SASS instruction schedule (instruction issue, predicates
#: and address arithmetic remain unresolved primitives).
_ACC_INIT_OPS_PER_LANE = 1
_FMA_OPS_PER_LANE_PER_EXPERT = 2  # one multiply, one add; not collapsed into FMA


@dataclass(frozen=True)
class _TileGeometry:
    """Source-derived launch and per-tile work geometry."""

    tokens: int
    topk: int
    hidden: int
    block_m: int
    block_k: int
    num_warps: int
    num_stages: int
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    is_fp32: bool

    @property
    def m_tiles(self) -> int:
        return (self.tokens + self.block_m - 1) // self.block_m

    @property
    def k_tiles(self) -> int:
        return (self.hidden + self.block_k - 1) // self.block_k

    @property
    def grid(self) -> tuple[int, int]:
        """Triton grid ``(cdiv(size, BLOCK_K), cdiv(tokens, BLOCK_M))``."""
        return self.k_tiles, self.m_tiles

    @property
    def programs(self) -> int:
        return self.k_tiles * self.m_tiles

    @property
    def accumulator_ops(self) -> int:
        return self.block_m * self.block_k

    @property
    def vector_ops_per_program(self) -> int:
        a_upconvert = 0 if self.input_bytes == 4 else 1
        w_upconvert = 0 if self.weight_bytes == 4 else 1
        out_downconvert = 0 if self.output_bytes == 4 else 1
        return (
            self.accumulator_ops * _ACC_INIT_OPS_PER_LANE
            + self.accumulator_ops * self.topk
            * (_FMA_OPS_PER_LANE_PER_EXPERT + a_upconvert)
            + self.block_m * self.topk * w_upconvert
            + self.accumulator_ops * out_downconvert
        )

    @property
    def weight_read_bytes(self) -> int:
        """Every CTA in a K column re-reads its top-k weights."""
        return self.k_tiles * self.tokens * self.topk * self.weight_bytes

    @property
    def weight_reads_per_k_tile(self) -> int:
        return self.tokens * self.topk

    @property
    def input_read_bytes(self) -> int:
        return self.tokens * self.topk * self.hidden * self.input_bytes

    @property
    def output_write_bytes(self) -> int:
        return self.tokens * self.hidden * self.output_bytes

    def coordinates(self, index: int) -> tuple[int, int, int, int]:
        """Return ``(pid_k, pid_m, valid_m, valid_k)`` for a linear CTA id.

        Triton's grid tuple is ``(pid_k, pid_m)`` with the first dimension
        varying fastest, so linear id ``pid_m * k_tiles + pid_k``.
        """
        if not 0 <= index < self.programs:
            raise ModelingError(f"moe_combine_tile_index_out_of_range:{index}")
        pid_m, pid_k = divmod(index, self.k_tiles)
        valid_m = min(self.block_m, self.tokens - pid_m * self.block_m)
        valid_k = min(self.block_k, self.hidden - pid_k * self.block_k)
        return pid_k, pid_m, valid_m, valid_k

    def tile_work(self, index: int) -> tuple[Work, Work, Work]:
        """Load, compute and store work of one program, in source order."""
        _, _, valid_m, valid_k = self.coordinates(index)
        a_bytes = self.topk * valid_m * valid_k * self.input_bytes
        w_bytes = self.topk * valid_m * self.weight_bytes
        load_bytes = a_bytes + w_bytes
        store_bytes = valid_m * valid_k * self.output_bytes
        return (
            Work(resources={"hbm": load_bytes, "mte2": load_bytes}),
            Work(resources={"vector": self.vector_ops_per_program}),
            Work(resources={"hbm": store_bytes, "mte3": store_bytes}),
        )

    def total_work(self) -> Work:
        load = self.input_read_bytes + self.weight_read_bytes
        return Work(resources={
            "hbm": load + self.output_write_bytes,
            "mte2": load,
            "mte3": self.output_write_bytes,
            "vector": self.programs * self.vector_ops_per_program,
        })


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"moe_combine_requires_positive_integer:{name}")
    return value


def _canonical_dtype(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"moe_combine_requires_dtype_string:{name}")
    dtype = _DTYPE_ALIASES.get(value.strip().lower())
    if dtype is None:
        raise ModelingError(f"moe_combine_unsupported_dtype:{name}:{value}")
    return dtype


def _explicit_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelingError(f"moe_combine_requires_explicit_boolean:{name}")
    return value


def next_power_of_two(value: int) -> int:
    """``triton.next_power_of_2`` equivalent for positive integers."""
    if value < 1:
        raise ModelingError(f"moe_combine_requires_positive_integer:size:{value}")
    return 1 << (value - 1).bit_length()


def _heuristic_config(
    num_tokens: int,
    top_k: int,
    size: int,
    element_size: int,
    compute_capability: int,
) -> tuple[int, int, int, int]:
    """Reproduce ``_heuristic_config`` lines 66-132 for the selected SM89 path.

    ``top_k`` is accepted because the source signature accepts it; the source
    body does not use it.  Only the explicit capability-89 branch is modeled:
    SM90+ and pre-SM80 select different tile/warp ceilings and are not the
    implementation named by this campaign case.
    """
    if compute_capability != COMPUTE_CAPABILITY:
        raise ModelingError(
            "moe_combine_requires_explicit_compute_capability_89:"
            f"{compute_capability}"
        )
    if min(num_tokens, top_k, size) < 1 or element_size not in (2, 4):
        raise ModelingError("moe_combine_invalid_heuristic_geometry")
    is_fp32 = element_size > 2

    if is_fp32:
        if num_tokens <= 4:
            block_m = 1
        elif num_tokens <= 32:
            block_m = 2
        else:
            block_m = 4
        max_block_k = 256
    else:
        if num_tokens <= 4:
            block_m = 1
        elif num_tokens <= 32:
            block_m = 2
        elif num_tokens <= 128:
            block_m = 4
        elif num_tokens <= 1024:
            block_m = 16
        else:
            block_m = 8
        max_block_k = 1024

    block_k = min(next_power_of_two(size), max_block_k)
    block_k = max(block_k, 256)

    total = block_m * block_k
    if is_fp32:
        num_warps = max(8, min(16, total // 64))
    else:
        num_warps = max(4, min(16, total // 256))
    # Capability 89 is neither pre-SM80 nor SM90+, so the source's final else
    # branch applies.
    num_stages = 4 if total <= 2048 else 2
    return block_m, block_k, num_warps, num_stages


def _derive(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise ModelingError("moe_combine_params_must_be_a_mapping")
    unknown = sorted(set(params) - set(_REQUIRED))
    if unknown:
        raise ModelingError(f"moe_combine_unknown_parameters:{','.join(unknown)}")
    missing = [name for name in _REQUIRED if name not in params]
    if missing:
        raise ModelingError(f"moe_combine_missing_parameters:{','.join(missing)}")

    tokens = _positive_int(params["tokens"], "tokens")
    hidden = _positive_int(params["hidden"], "hidden")
    topk = _positive_int(params["topk"], "topk")
    dtype = _canonical_dtype(params["dtype"], "dtype")
    weight_dtype = _canonical_dtype(params["weight_dtype"], "weight_dtype")
    if dtype not in _SUPPORTED_DTYPES or weight_dtype not in _SUPPORTED_DTYPES:
        raise ModelingError("moe_combine_unsupported_operand_dtype")
    capability = params["compute_capability"]
    if isinstance(capability, bool) or not isinstance(capability, int):
        raise ModelingError(
            f"moe_combine_requires_explicit_compute_capability_89:{capability}"
        )
    if params["implementation"] != IMPLEMENTATION:
        raise ModelingError(
            f"moe_combine_unsupported_implementation:{params['implementation']}"
        )
    has_expert_map = _explicit_bool(params["has_expert_map"], "has_expert_map")
    apply_on_input = _explicit_bool(
        params["apply_router_weight_on_input"], "apply_router_weight_on_input"
    )
    if has_expert_map:
        # The kernel has this branch (lines 42-49), but the campaign selects the
        # branch that never loads topk_ids or expert_map.  Reject rather than
        # silently price a different implementation.
        raise ModelingError("moe_combine_expert_map_branch_not_selected")
    if apply_on_input:
        # moe_fused_mul_sum always multiplies by topk_weights (line 56).  The
        # topk_weight_and_reduce contract (lines 107-108) skips that multiply
        # when the caller already applied the weights on the input.
        raise ModelingError(
            "moe_combine_apply_router_weight_on_input_not_supported_by_selected_kernel"
        )

    input_bytes = int(dtype_bytes(dtype))
    weight_bytes = int(dtype_bytes(weight_dtype))
    output_bytes = input_bytes
    block_m, block_k, warps, stages = _heuristic_config(
        tokens, topk, hidden, input_bytes, capability,
    )
    geometry = _TileGeometry(
        tokens=tokens, topk=topk, hidden=hidden,
        block_m=block_m, block_k=block_k,
        num_warps=warps, num_stages=stages,
        input_bytes=input_bytes, weight_bytes=weight_bytes,
        output_bytes=output_bytes, is_fp32=input_bytes > 2,
    )
    return {
        "tokens": tokens, "hidden": hidden, "topk": topk,
        "dtype": dtype, "weight_dtype": weight_dtype,
        "implementation": params["implementation"],
        "compute_capability": capability,
        "has_expert_map": has_expert_map,
        "apply_router_weight_on_input": apply_on_input,
        "input_bytes": input_bytes, "weight_bytes": weight_bytes,
        "output_bytes": output_bytes, "geometry": geometry,
    }


def _metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
    g: _TileGeometry = cfg["geometry"]
    last_m = min(g.block_m, g.tokens - (g.m_tiles - 1) * g.block_m)
    last_k = min(g.block_k, g.hidden - (g.k_tiles - 1) * g.block_k)
    dispatch = {
        "implementation": cfg["implementation"],
        "kernel": KERNEL,
        "wrapper": "moe_fused_mul_sum",
        "source_kernel_launches": 1,
        "fake_tensor_guard": "launch_only_for_concrete_tensors",
        "apply_router_weight_on_input": cfg["apply_router_weight_on_input"],
        "has_expert_map": cfg["has_expert_map"],
        "compute_capability": cfg["compute_capability"],
        "input_dtype": cfg["dtype"],
        "weight_dtype": cfg["weight_dtype"],
        "output_dtype": cfg["dtype"],
        "accumulation_dtype": "fp32",
        "element_size": g.input_bytes,
        "is_fp32": g.is_fp32,
        "BLOCK_M": g.block_m,
        "BLOCK_K": g.block_k,
        "num_warps": g.num_warps,
        "num_stages": g.num_stages,
        "grid": g.grid,
    }
    return {
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "source_lines": SOURCE_LINES,
        "source_origin": SOURCE_ORIGIN,
        "source_functions": "moe_fused_mul_sum_kernel / _heuristic_config / moe_fused_mul_sum",
        "contract_source_path": CONTRACT_PATH,
        "contract_source_sha256": CONTRACT_SHA256,
        "contract_source_bytes": CONTRACT_BYTES,
        "contract_source_lines": CONTRACT_LINES,
        "implementation": IMPLEMENTATION,
        "kernel": KERNEL,
        "recipe": RECIPE,
        "dispatch": dispatch,
        "compute_capability": cfg["compute_capability"],
        "input_dtype": cfg["dtype"],
        "output_dtype": cfg["dtype"],
        "weight_dtype": cfg["weight_dtype"],
        "input_element_bytes": g.input_bytes,
        "weight_element_bytes": g.weight_bytes,
        "output_element_bytes": g.output_bytes,
        "accumulation_dtype": "fp32",
        "inputs_layout": "contiguous",
        "topk_weights_layout": "contiguous",
        "tokens": g.tokens,
        "top_k": g.topk,
        "hidden": g.hidden,
        "block_m": g.block_m,
        "block_k": g.block_k,
        "k_tiles": g.k_tiles,
        "m_tiles": g.m_tiles,
        "grid": g.grid,
        "programs": g.programs,
        "num_warps": g.num_warps,
        "num_stages": g.num_stages,
        "last_tile_rows": last_m,
        "last_tile_cols": last_k,
        "padding_rows": g.m_tiles * g.block_m - g.tokens,
        "padding_cols": g.k_tiles * g.block_k - g.hidden,
        "arithmetic_lanes_per_program": g.block_m * g.block_k,
        "vector_ops_per_program": g.vector_ops_per_program,
        "vector_ops_formula": (
            "programs*(BM*BK*(1 + topk*(2 + [input<4B]) + [output<4B])"
            " + BM*topk*[weight<4B])"
        ),
        "weight_read_bytes": g.weight_read_bytes,
        "weight_reads_per_k_tile": g.weight_reads_per_k_tile,
        "weight_read_policy": "repeated_once_per_k_tile",
        "input_read_bytes": g.input_read_bytes,
        "output_write_bytes": g.output_write_bytes,
        "hbm_formula": "input_read_bytes + weight_read_bytes + output_write_bytes",
        "masked_arithmetic": True,
        "tail_store": True,
        "weight_broadcast_lanes": g.block_k,
        "routing_bytes": 0,
        "communication_bytes": 0,
        "expert_map_branch": False,
        "routing_ids_read": False,
        "kernel_latency_floor_ms": 0.0,
        "cta_residency_source": "min(grid,geometry.compute_units)_owned_cta_streams",
        "cta_owner_count_rule": "min(grid_programs,geometry.compute_units)",
        "cta_ownership": "one_physical_owner_per_cta",
        "cta_assignment": "linear_cta_index_modulo_owner_count",
        "resident_ctas_per_owner": 1,
        "vector_ownership": "pinned_per_cta",
        "copy_ownership": (
            "mte2/mte3_pinned_per_cta_when_declared; hbm_remains_shared"
        ),
        "workspace_bytes": 0,
        "allocation_pools": (),
        "capture_dispatch_claim": "none_no_historical_observation_for_this_kernel",
        "unresolved_primitives": (
            "instruction_issue", "predicate_ops", "address_arithmetic",
            "memory_response", "cta_occupancy", "l1_weight_reuse",
            "triton_codegen_schedule",
        ),
    }


def make_op(params: dict) -> OpSpec:
    """Build the explicit case op for the selected no-map SM89 path."""
    cfg = _derive(params)
    g: _TileGeometry = cfg["geometry"]
    work = g.total_work()
    op = OpSpec(
        op_id="moe_fused_mul_sum",
        family="moe_combine",
        phase="operator",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(g.tokens, g.topk, g.hidden),
        local_shape=(g.tokens, g.topk, g.hidden),
        padded_shape=(g.m_tiles * g.block_m, g.topk, g.k_tiles * g.block_k),
        launch_shape=(g.k_tiles, g.m_tiles, g.block_m, g.block_k),
        dtype=cfg["dtype"],
        layout="contiguous",
        recipe=RECIPE, source_boundary="kernel",
        metadata=tuple(sorted(_metadata(cfg).items())),
        notes=(
            "selected source is vLLM v0.25.1 current checkout, not a capture image",
            "no-expert-map branch; topk_ids/expert_map are not loaded",
            "topk_weights applied by this kernel, not on the input",
            "FP32 accumulation; BF16/FP16/FP32 I/O follows the declared dtype",
            "masked tile arithmetic; only valid lanes read/write memory",
            "topk_weights re-read once per K tile; no routing or comm bytes",
            "one CTA per owned SM partition; idle SMs cannot serve a CTA",
            "no workspace allocation; caller tensors and register tile only",
        ),
    )
    return work.apply(op)


def _owned(context, core: int, owners: int, roles: tuple[str, ...]):
    """Resolve one CTA owner to its exclusive physical unit partition."""
    try:
        return context.owned(core, owners, roles)
    except ValueError as exc:
        raise ModelingError(
            "moe_combine_ownership_geometry_incompatible:"
            f"core={core}:owners={owners}:{exc}"
        ) from None


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower the source grid to one lazy CTA stream per physical owner."""
    if not isinstance(op, OpSpec):
        raise ModelingError("moe_combine_build_requires_opspec")
    metadata = dict(op.metadata)
    if metadata.get("implementation") != IMPLEMENTATION:
        raise ModelingError(
            f"moe_combine_selected_source_mismatch:{metadata.get('implementation')}"
        )
    if op.recipe and op.recipe != RECIPE:
        raise ModelingError(f"moe_combine_recipe_mismatch:{op.recipe}")
    if op.layout not in ("", "contiguous"):
        raise ModelingError(f"moe_combine_requires_contiguous_layout:{op.layout}")
    if len(op.local_shape) != 3:
        raise ModelingError("moe_combine_local_shape_requires_tokens_topk_hidden")
    tokens, topk, hidden = (int(value) for value in op.local_shape)
    if min(tokens, topk, hidden) < 1:
        raise ModelingError(f"moe_combine_shape_invalid:{op.local_shape}")
    dtype = _canonical_dtype(op.dtype, "dtype")
    weight_dtype = _canonical_dtype(metadata.get("weight_dtype"), "weight_dtype")
    capability = metadata.get("compute_capability")
    if (isinstance(capability, bool) or not isinstance(capability, int)
            or capability != COMPUTE_CAPABILITY):
        raise ModelingError(
            f"moe_combine_requires_explicit_compute_capability_89:{capability}"
        )
    if metadata.get("input_dtype") not in (None, dtype):
        raise ModelingError("moe_combine_metadata_mismatch:input_dtype")
    if metadata.get("output_dtype") not in (None, dtype):
        raise ModelingError("moe_combine_metadata_mismatch:output_dtype")
    input_bytes = int(dtype_bytes(dtype))
    weight_bytes = int(dtype_bytes(weight_dtype))
    block_m, block_k, warps, stages = _heuristic_config(
        tokens, topk, hidden, input_bytes, capability,
    )
    g = _TileGeometry(
        tokens=tokens, topk=topk, hidden=hidden,
        block_m=block_m, block_k=block_k,
        num_warps=warps, num_stages=stages,
        input_bytes=input_bytes, weight_bytes=weight_bytes,
        output_bytes=input_bytes, is_fp32=input_bytes > 2,
    )
    # Reject a stale envelope rather than trusting metadata that no longer
    # matches the OpSpec geometry/dtype.
    for key, expected in {
        "tokens": tokens, "top_k": topk, "hidden": hidden,
        "block_m": g.block_m, "block_k": g.block_k,
        "k_tiles": g.k_tiles, "m_tiles": g.m_tiles,
        "grid": g.grid, "programs": g.programs,
        "num_warps": g.num_warps, "num_stages": g.num_stages,
    }.items():
        if key in metadata and metadata[key] != expected:
            raise ModelingError(f"moe_combine_metadata_mismatch:{key}")
    dispatch = metadata.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise ModelingError("moe_combine_dispatch_missing")
    for key, expected in {
        "implementation": IMPLEMENTATION, "kernel": KERNEL,
        "has_expert_map": False, "apply_router_weight_on_input": False,
        "compute_capability": COMPUTE_CAPABILITY,
        "BLOCK_M": g.block_m, "BLOCK_K": g.block_k,
        "num_warps": g.num_warps, "num_stages": g.num_stages,
        "element_size": g.input_bytes, "is_fp32": g.is_fp32,
        "grid": g.grid,
    }.items():
        if dispatch.get(key) != expected:
            raise ModelingError(f"moe_combine_dispatch_mismatch:{key}")
    # Newer provenance flags are checked when present; make_op always writes
    # them, while a hand-built equivalent OpSpec may omit the same defaults.
    for key, expected in {
        "wrapper": "moe_fused_mul_sum", "source_kernel_launches": 1,
        "fake_tensor_guard": "launch_only_for_concrete_tensors",
        "input_dtype": dtype, "weight_dtype": weight_dtype,
        "output_dtype": dtype, "accumulation_dtype": "fp32",
    }.items():
        if key in dispatch and dispatch[key] != expected:
            raise ModelingError(f"moe_combine_dispatch_mismatch:{key}")
    compute_units = int(geometry.compute_units)
    if compute_units < 1:
        raise ModelingError("moe_combine_requires_positive_compute_units")
    total = g.total_work()
    # A physical owner is an SM partition, never "all idle SMs". The number of
    # active owners follows the source grid exactly; a grid smaller than the
    # device leaves the remaining SMs idle, as the kernel launch does.
    active_owners = min(g.programs, compute_units)

    def program(context):
        def tile_stream(core: int):
            # Linear CTA ids are ordered pid_m * k_tiles + pid_k (Triton's
            # first grid dimension varies fastest) and assigned round-robin to
            # physical owners. Each owner runs one resident CTA at a time.
            count = (g.programs - core - 1) // active_owners + 1

            def tile(local_index: int):
                index = core + local_index * active_owners
                load, compute, store = g.tile_work(index)
                return (
                    Stage("load", context.bind(load),
                          placement=_owned(context, core, compute_units,
                                           ("mte2",))),
                    Stage("fma", context.bind(compute), after=("load",),
                          placement=_owned(context, core, compute_units,
                                           ("vector",))),
                    Stage("store", context.bind(store), after=("fma",),
                          placement=_owned(context, core, compute_units,
                                           ("mte3",))),
                )

            return PipelineProgram(count, 1, tile)

        return ParallelProgram(
            tuple(tile_stream(core) for core in range(active_owners))
        )

    return OperatorPlan(total, program)


__all__ = [
    "build", "make_op", "RECIPE", "IMPLEMENTATION", "KERNEL",
    "COMPUTE_CAPABILITY", "SOURCE_PATH", "SOURCE_SHA256", "SOURCE_BYTES",
    "SOURCE_ORIGIN", "CONTRACT_PATH", "CONTRACT_SHA256", "CONTRACT_BYTES",
    "next_power_of_two", "_heuristic_config", "_TileGeometry",
]
