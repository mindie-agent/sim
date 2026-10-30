"""Source-derived Triton causal convolution prefill with state and tails.

[Role] Model causal_conv1d_fn's explicit non-APC forward launch.
[Invariants] One CTA per sequence chunk and BLOCK_N channel block owns one SM.
  Both explicit no-null and wrapper null_block_id=0 branches are represented;
  a null state slot exits before work. Zero-length sequences launch no CTA.
  Per-sequence length, state presence and cache index are preserved. Only first
  chunks write state. Arithmetic pads channel lanes; traffic retains real tails,
  duplicate state reads and repeated weights. The update kernel never substitutes
  for prefill; no measured duration or inferred occupancy enters the program.
[Timeouts & Locks] Lazy per-owner streams and finite slot windows; no waits/locks.
[Side-Effect Policy] Immutable metadata and caller Engine tasks only; no device I/O.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec


IMPLEMENTATION = "triton_causal_conv1d_fwd"
KERNEL_FUNCTION = "_causal_conv1d_fwd_kernel"
WRAPPER_FUNCTION = "causal_conv1d_fn"
BLOCK_M = 8
BLOCK_N = 256
NUM_STAGES = 2
# Declared per-owner in-flight CTA window.  It is a dependency-window model
# parameter, deliberately not Triton's num_stages=2 and not an occupancy claim.
OWNER_CTA_WINDOW = 2
SOURCE_PATH = "/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
SOURCE_SHA256 = "5f3d16d730fd6dc427cf6981a3de58b1e1e625e624ae8fa0fe77891f58b8bd5c"
SOURCE_BYTES = 49533
SOURCE_LINES = "16-465 (kernel); 468-745 (wrapper)"
NULL_BLOCK_SOURCE_PATH = "/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/backends/utils.py"
NULL_BLOCK_SOURCE_SHA256 = "192e6b5df24ff52ae57a6a23d6cf58d1b4dd34f14c9283ce268e5f88b1123b96"
NULL_BLOCK_SOURCE_LINE = 46
PAD_SLOT_SOURCE_LINE = 45
PAD_SLOT_ID = -1
_SOURCE_LINES_DETAIL = {
    "kernel": "16-465",
    "wrapper": "468-745",
    "grid": "689-693",
    "block": "578, 636, 741-743",
    "state_small": "196-235",
    "state_large_init": "237-280",
    "state_large_noinit": "281-307",
    "prior_tokens": "309-338",
    "null_block": "136-139",
    "null_block_default": "478, 738",
    "apc_off": "116-121",
    "token_loop": "411-465",
}

_ALLOWED_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_CANONICAL_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
_ACTIVATIONS = frozenset({"silu", "swish"})
_CORE_KEYS = (
    "lengths", "channels", "width", "dtype", "has_initial_state", "bias",
    "activation", "block_size_to_align", "num_cache_lines", "cache_index_stride",
    "has_null_block", "null_block_id", "cache_indices",
)
_PARAM_KEYS = frozenset({
    "lengths", "channels", "width", "dtype", "initial_state", "bias",
    "activation", "implementation", "is_apc_enabled", "has_null_block",
    "null_block_id", "cache_indices", "block_size_to_align", "num_cache_lines",
    "cache_index_stride",
})


@dataclass(frozen=True)
class _ConvSpec:
    """Normalized structural inputs; every field is declared by the wrapper."""

    lengths: tuple[int, ...]
    channels: int
    width: int
    dtype: str
    has_initial_state: tuple[bool, ...]
    bias: bool
    activation: str | None
    block_size_to_align: int
    num_cache_lines: int
    cache_index_stride: int
    has_null_block: bool
    null_block_id: int | None
    cache_indices: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.lengths, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.lengths):
            raise ModelingError("causal_conv_prefill_lengths_must_be_nonnegative_ints")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels < 1:
            raise ModelingError("causal_conv_prefill_channels_must_be_positive")
        # The fwd token loop has explicit branches only for 2, 3 and 4 (lines
        # 411-452).  Width-5 state loads exist but the token loop has no width-5
        # branch; the update kernel's 5/6 branches are a different kernel.
        if isinstance(self.width, bool) or not isinstance(self.width, int) or not 2 <= self.width <= 4:
            raise ModelingError("causal_conv_prefill_kernel_width_unsupported:2..4")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ModelingError(f"causal_conv_prefill_dtype_unsupported:{self.dtype}")
        if len(self.has_initial_state) != len(self.lengths):
            raise ModelingError("causal_conv_prefill_initial_state_length_mismatch")
        if any(not isinstance(value, bool) for value in self.has_initial_state):
            raise ModelingError("causal_conv_prefill_initial_state_flags_must_be_bool")
        if not isinstance(self.bias, bool):
            raise ModelingError("causal_conv_prefill_bias_must_be_bool")
        if self.activation not in (None, "silu", "swish"):
            raise ModelingError(f"causal_conv_prefill_activation_unsupported:{self.activation}")
        if (isinstance(self.block_size_to_align, bool)
                or not isinstance(self.block_size_to_align, int)
                or self.block_size_to_align < 0
                or (self.block_size_to_align and self.block_size_to_align % BLOCK_M)):
            raise ModelingError("causal_conv_prefill_block_alignment_must_be_multiple_of_block_m")
        if (isinstance(self.num_cache_lines, bool) or not isinstance(self.num_cache_lines, int)
                or self.num_cache_lines < max(1, len(self.lengths))):
            raise ModelingError("causal_conv_prefill_num_cache_lines_below_batch")
        if (isinstance(self.cache_index_stride, bool) or not isinstance(self.cache_index_stride, int)
                or self.cache_index_stride < 1):
            raise ModelingError("causal_conv_prefill_cache_index_stride_must_be_positive")
        if not isinstance(self.has_null_block, bool):
            raise ModelingError("causal_conv_prefill_has_null_block_must_be_bool")
        if self.has_null_block:
            if (isinstance(self.null_block_id, bool) or not isinstance(self.null_block_id, int)
                    or self.null_block_id < 0):
                raise ModelingError("causal_conv_prefill_null_block_id_required")
        elif self.null_block_id is not None:
            # Source line 738 derives HAS_NULL_BLOCK from null_block_id, so a
            # false flag must be represented by the None specialization.
            raise ModelingError("causal_conv_prefill_null_block_id_must_be_none_when_disabled")
        if len(self.cache_indices) != len(self.lengths) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.cache_indices):
            raise ModelingError("causal_conv_prefill_cache_indices_must_be_nonnegative_ints_per_sequence")
        for index, cache_index in enumerate(self.cache_indices):
            if self.has_null_block and cache_index == self.null_block_id:
                continue
            if cache_index >= self.num_cache_lines:
                raise ModelingError("causal_conv_prefill_cache_index_exceeds_num_cache_lines")

    @property
    def state_len(self):
        return self.width - 1

    @property
    def np2_state_len(self):
        state_len = self.state_len
        return 1 << (state_len - 1).bit_length()

    @property
    def feature_blocks(self):
        return _ceil_div(self.channels, BLOCK_N)

    @property
    def element_bytes(self):
        return float(dtype_bytes(self.dtype))

    @property
    def activation_ops(self):
        # acc / (1 + exp(-acc)): negate, exp, add and divide.
        return 4.0 if self.activation in _ACTIVATIONS else 0.0

    @property
    def output_cast_ops(self):
        # The accumulator is fp32.  A bf16/fp16 output store converts; an fp32
        # output store does not.
        return 0.0 if self.dtype == "fp32" else 1.0

    @property
    def per_token_vector_ops(self):
        # acc = acc_preload; W multiply-adds; W-1 rolling-window register moves;
        # SiLU's negate/exp/add/div; optional fp32-to-out-dtype store conversion.
        return (1.0 + 2.0 * self.width + (self.width - 1) +
                self.activation_ops + self.output_cast_ops)

    @property
    def skipped(self):
        """Sequences whose cache index selects the HAS_NULL_BLOCK early return."""
        if not self.has_null_block:
            return (False,) * len(self.lengths)
        return tuple(index == self.null_block_id for index in self.cache_indices)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _nonnegative_int(value, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(reason)
    return value


def _normalize_params(params) -> _ConvSpec:
    if not isinstance(params, dict):
        raise ModelingError("causal_conv_prefill_params_must_be_mapping")
    unknown = sorted(set(params) - _PARAM_KEYS)
    if unknown:
        raise ModelingError(f"causal_conv_prefill_unknown_parameter:{unknown[0]}")

    lengths_raw = params.get("lengths")
    if isinstance(lengths_raw, (str, bytes)) or not isinstance(lengths_raw, (list, tuple)):
        raise ModelingError("causal_conv_prefill_lengths_must_be_sequence")
    lengths = tuple(lengths_raw)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in lengths):
        raise ModelingError("causal_conv_prefill_lengths_must_be_nonnegative_ints")

    channels = params.get("channels")
    width = params.get("width")
    dtype_raw = params.get("dtype")
    if not isinstance(dtype_raw, str) or dtype_raw.strip().lower() not in _ALLOWED_DTYPES:
        raise ModelingError(f"causal_conv_prefill_dtype_unsupported:{dtype_raw}")
    dtype = _CANONICAL_DTYPES.get(dtype_raw.strip().lower(), dtype_raw.strip().lower())

    initial_raw = params.get("initial_state", False)
    if isinstance(initial_raw, bool):
        has_initial = (initial_raw,) * len(lengths)
    elif isinstance(initial_raw, (list, tuple)):
        if len(initial_raw) != len(lengths) or any(not isinstance(value, bool) for value in initial_raw):
            raise ModelingError("causal_conv_prefill_initial_state_length_or_type_mismatch")
        has_initial = tuple(initial_raw)
    else:
        raise ModelingError("causal_conv_prefill_initial_state_must_be_bool_or_per_sequence_flags")

    bias_raw = params.get("bias", False)
    if not isinstance(bias_raw, bool):
        raise ModelingError("causal_conv_prefill_bias_must_be_bool")

    activation_raw = params.get("activation", "silu")
    if isinstance(activation_raw, bool):
        activation_raw = "silu" if activation_raw else None
    if isinstance(activation_raw, str):
        activation_raw = activation_raw.strip().lower()
    if activation_raw is not None and activation_raw not in _ACTIVATIONS:
        raise ModelingError(f"causal_conv_prefill_activation_unsupported:{activation_raw}")
    activation = activation_raw

    implementation = params.get("implementation")
    if implementation != IMPLEMENTATION:
        raise ModelingError(f"causal_conv_prefill_unsupported_implementation:{implementation}")
    if params.get("is_apc_enabled") is not False:
        raise ModelingError("causal_conv_prefill_apc_not_selected")

    # Source wrapper default: ``null_block_id: int = NULL_BLOCK_ID`` (0), so
    # HAS_NULL_BLOCK is true when the caller does not override the default.
    # The source derives HAS_NULL_BLOCK from null_block_id (line 738), so an
    # explicit false flag corresponds to the None specialization.
    null_block_present = "null_block_id" in params
    null_block_raw = params.get("null_block_id", 0)
    if null_block_raw is not None:
        _nonnegative_int(null_block_raw, "causal_conv_prefill_null_block_id_must_be_nonnegative_int_or_none")
    if "has_null_block" in params:
        has_null_raw = params["has_null_block"]
        if not isinstance(has_null_raw, bool):
            raise ModelingError("causal_conv_prefill_has_null_block_must_be_bool")
        if has_null_raw:
            if null_block_raw is None:
                raise ModelingError("causal_conv_prefill_null_block_id_required")
        else:
            if null_block_present and null_block_raw is not None:
                raise ModelingError(
                    "causal_conv_prefill_null_block_flag_disagrees_with_null_block_id")
            null_block_raw = None
    else:
        has_null_raw = null_block_raw is not None

    cache_indices_raw = params.get("cache_indices")
    if cache_indices_raw is None:
        # Declared one-line-per-sequence assumption used by the selected plan.
        cache_indices = tuple(range(len(lengths)))
    else:
        if (isinstance(cache_indices_raw, (str, bytes))
                or not isinstance(cache_indices_raw, (list, tuple))
                or len(cache_indices_raw) != len(lengths)):
            raise ModelingError("causal_conv_prefill_cache_indices_must_be_nonnegative_ints_per_sequence")
        cache_indices = tuple(cache_indices_raw)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cache_indices):
        raise ModelingError("causal_conv_prefill_cache_indices_must_be_nonnegative_ints_per_sequence")

    block_align = params.get("block_size_to_align", 0)
    if (isinstance(block_align, bool) or not isinstance(block_align, int)
            or block_align < 0 or (block_align and block_align % BLOCK_M)):
        raise ModelingError("causal_conv_prefill_block_alignment_must_be_multiple_of_block_m")

    minimum_cache_lines = max(
        1, len(lengths), max(cache_indices, default=-1) + 1,
    )
    num_cache_lines = params.get("num_cache_lines", minimum_cache_lines)
    cache_index_stride = params.get("cache_index_stride", 1)
    return _ConvSpec(
        lengths=lengths, channels=channels, width=width, dtype=dtype,
        has_initial_state=has_initial, bias=bias_raw, activation=activation,
        block_size_to_align=block_align, num_cache_lines=num_cache_lines,
        cache_index_stride=cache_index_stride, has_null_block=has_null_raw,
        null_block_id=null_block_raw, cache_indices=cache_indices,
    )


def _spec_from_metadata(metadata) -> _ConvSpec:
    meta = dict(metadata)
    missing = [key for key in _CORE_KEYS if key not in meta]
    if missing:
        raise ModelingError(f"causal_conv_prefill_metadata_missing:{missing[0]}")
    lengths = meta["lengths"]
    if isinstance(lengths, (str, bytes)) or not isinstance(lengths, (list, tuple)):
        raise ModelingError("causal_conv_prefill_lengths_must_be_sequence")
    initial = meta["has_initial_state"]
    if isinstance(initial, (str, bytes)) or not isinstance(initial, (list, tuple)):
        raise ModelingError("causal_conv_prefill_initial_state_length_or_type_mismatch")
    cache_indices = meta["cache_indices"]
    if isinstance(cache_indices, (str, bytes)) or not isinstance(cache_indices, (list, tuple)):
        raise ModelingError("causal_conv_prefill_cache_indices_must_be_nonnegative_ints_per_sequence")
    return _ConvSpec(
        lengths=tuple(lengths), channels=meta["channels"], width=meta["width"],
        dtype=meta["dtype"], has_initial_state=tuple(bool(value) for value in initial),
        bias=meta["bias"], activation=meta["activation"],
        block_size_to_align=meta["block_size_to_align"],
        num_cache_lines=meta["num_cache_lines"],
        cache_index_stride=meta["cache_index_stride"],
        has_null_block=meta["has_null_block"],
        null_block_id=meta["null_block_id"],
        cache_indices=tuple(cache_indices),
    )


def _chunks_per_sequence(spec: _ConvSpec) -> tuple[int, ...]:
    return tuple(_ceil_div(length, BLOCK_M) for length in spec.lengths)


def _counts(spec: _ConvSpec):
    chunks_per_sequence = _chunks_per_sequence(spec)
    chunks = sum(chunks_per_sequence)
    blocks = spec.feature_blocks
    skipped = spec.skipped
    processed = tuple(
        bool(length) and not skip
        for length, skip in zip(spec.lengths, skipped))
    early_chunks = sum(
        count for count, skip in zip(chunks_per_sequence, skipped) if skip)
    processed_sequences = sum(processed)
    barrier_sequences = sum(
        spec.state_len <= length or has_initial
        for length, has_initial, active in zip(
            spec.lengths, spec.has_initial_state, processed)
        if active
    )
    return {
        "sequences": len(spec.lengths),
        "tokens": sum(spec.lengths),
        "chunks": chunks,
        "padded_token_slots": chunks * BLOCK_M,
        "feature_blocks": blocks,
        "padded_channels_per_chunk": blocks * BLOCK_N,
        "programs": chunks * blocks,
        "processed_programs": (chunks - early_chunks) * blocks,
        "early_return_programs": early_chunks * blocks,
        "processed_sequences": processed_sequences,
        "state_write_programs": processed_sequences * blocks,
        "barrier_instructions": barrier_sequences * blocks,
    }


def _dispatch(spec: _ConvSpec):
    return {
        "implementation": IMPLEMENTATION,
        "kernel_function": KERNEL_FUNCTION,
        "wrapper": WRAPPER_FUNCTION,
        "is_apc_enabled": False,
        "has_null_block": spec.has_null_block,
        "null_block_id": spec.null_block_id,
        "pad_slot_id": PAD_SLOT_ID,
        "has_bias": spec.bias,
        "kernel_width": spec.width,
        "silu_activation": spec.activation in _ACTIVATIONS,
        "state_len": spec.state_len,
        "np2_statelen": spec.np2_state_len,
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "num_stages": NUM_STAGES,
        "cta_window": OWNER_CTA_WINDOW,
        "layout": "channel_last",
        "cache_indices": spec.cache_indices,
        "cache_index_stride": spec.cache_index_stride,
        "block_size_to_align": spec.block_size_to_align,
        "num_cache_lines": spec.num_cache_lines,
        "initial_state_mode": "per_sequence_flags",
        "dtype": spec.dtype,
    }


def make_op(params: dict) -> OpSpec:
    """Build the selected fwd-kernel OpSpec; physical work is owned by build."""
    spec = _normalize_params(params)
    counts = _counts(spec)
    chunks_per_sequence = _chunks_per_sequence(spec)
    grid = (sum(chunks_per_sequence), spec.feature_blocks)
    meta = {
        "dispatch": _dispatch(spec),
        "source": {
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "lines": SOURCE_LINES,
            "line_map": dict(_SOURCE_LINES_DETAIL),
            "scope": "current_checkout_vllm_v0.25.1_triton_cuda_prefill",
        },
        "null_block_source": {
            "path": NULL_BLOCK_SOURCE_PATH,
            "sha256": NULL_BLOCK_SOURCE_SHA256,
            "line": NULL_BLOCK_SOURCE_LINE,
            "value": 0,
            "pad_slot_id": PAD_SLOT_ID,
            "pad_slot_line": PAD_SLOT_SOURCE_LINE,
        },
        "boundary": "device_kernel:_causal_conv1d_fwd_kernel",
        "tiling": {
            "block_m": BLOCK_M, "block_n": BLOCK_N, "num_stages": NUM_STAGES,
            "cta_window_per_owner": OWNER_CTA_WINDOW,
            "grid": grid,
            "grid_order": "source x=sequence-major chunk, y=channel block; "
                          "linear CTA id = chunk + total_chunks*feature_block",
            "cta_schedule": "round-robin over linear source-grid CTA id across "
                            "physical SM owners; lazy per-owner streams",
        },
        "counts": counts,
        "work_model": {
            "element_bytes": spec.element_bytes,
            "vector_ops_per_token_lane": spec.per_token_vector_ops,
            "activation_ops_per_lane": spec.activation_ops,
            "output_cast_ops_per_token_lane": spec.output_cast_ops,
            "bias_preload_ops_per_program_lane": 1.0 if spec.bias else 0.0,
            "state_select_ops_per_program_element": 1.0,
            "scalar_bytes_per_program": 20.0,
            "null_block_scalar_bytes_per_program": 20.0,
            "has_initial_state_bytes_per_chunk0_program": 1.0,
            "state_output_coord_bytes_when_state_len_le_seqlen": 4.0,
            "barrier_rule": "chunk0 and (state_len <= seqlen or has_initial_state)",
        },
        "lengths": spec.lengths,
        "channels": spec.channels,
        "width": spec.width,
        "dtype": spec.dtype,
        "has_initial_state": spec.has_initial_state,
        "bias": spec.bias,
        "activation": spec.activation,
        "block_size_to_align": spec.block_size_to_align,
        "num_cache_lines": spec.num_cache_lines,
        "cache_index_stride": spec.cache_index_stride,
        "has_null_block": spec.has_null_block,
        "null_block_id": spec.null_block_id,
        "cache_indices": spec.cache_indices,
        "cache_index_assumption": "explicit cache_indices or identity 0..batch-1; "
                                   "null_block_id lines are skipped when enabled",
    }
    return OpSpec(
        op_id="causal_conv_prefill",
        family="causal_conv1d",
        phase="prefill",
        stage=0,
        layer=0,
        requests=("case",),
        semantic_shape=(spec.channels, counts["tokens"]),
        local_shape=(spec.channels, counts["tokens"]),
        padded_shape=(spec.channels, counts["padded_token_slots"]),
        launch_shape=(spec.channels, counts["padded_token_slots"]),
        flops=0.0,
        hbm_bytes=0.0,
        mte2_bytes=0.0,
        mte3_bytes=0.0,
        vector_ops=0.0,
        deps=(),
        recipe=IMPLEMENTATION, source_boundary="kernel",
        coverage="decomposed",
        dtype=spec.dtype,
        layout="channel_last",
        returns=("final_conv_state",),
        metadata=tuple(sorted(meta.items())),
        notes=(
            "selected _causal_conv1d_fwd_kernel; update kernel is not a substitute",
            "APC disabled: only chunk offset 0 writes the final conv state",
            "HAS_NULL_BLOCK cache-index early returns are modeled explicitly",
            "each source-grid CTA's vector work is owned by one physical SM; memory is shared",
            "physical work is declared by build; wrapper host metadata is outside the device-kernel boundary",
            "tl.debug_barrier is counted structurally and conditionally, not priced",
        ),
    )


@dataclass(frozen=True)
class _TileWork:
    prep_hbm: float = 0.0
    conv_load_hbm: float = 0.0
    state_read_hbm: float = 0.0
    state_store_hbm: float = 0.0
    output_hbm: float = 0.0
    conv_vector: float = 0.0
    state_vector: float = 0.0
    barrier: bool = False
    state_rewrite: bool = False
    early_return: bool = False
    first: bool = False


def _tile_work(spec: _ConvSpec, length: int, chunk_offset: int,
               feature_block: int, has_initial: bool, skipped: bool) -> _TileWork:
    """Exact work for one Triton program; mirrors lines 74-465."""
    if skipped:
        # Lines 75-87 and 132-139: two program ids, two query_start_loc entries
        # and the cache-index scalar are loaded before the early return.  No
        # weight, state, conv or output traffic occurs.
        return _TileWork(prep_hbm=20.0, early_return=True)

    elem = spec.element_bytes
    state_len = spec.state_len
    np2 = spec.np2_state_len
    first = chunk_offset == 0
    seg = min(BLOCK_M, length - chunk_offset * BLOCK_M)
    active = min(BLOCK_N, spec.channels - feature_block * BLOCK_N)
    if seg <= 0 or active <= 0:
        raise ModelingError("causal_conv_prefill_invalid_program_geometry")

    # Weight and bias are reloaded by every program (397-409, 386-393).  The
    # initial state is read by every chunk-0 program when the flag is set.
    prep = active * spec.width * elem
    if spec.bias:
        prep += active * elem
    if first and has_initial:
        prep += state_len * active * elem
    # batch_ptr, token_chunk_offset_ptr, two query_start_loc entries and the
    # conv_state_indices entry are scalar int32 loads in every program (75-76,
    # 84-85, 132-134).  The chunk-0 program also reads has_initial_states.
    prep += 20.0
    if first:
        prep += 1.0

    conv_load = active * seg * elem
    if not first:
        # Lines 309-338: chunk_offset > 0 reads the preceding KERNEL_WIDTH-1
        # tokens from x instead of the conv state.
        conv_load += (spec.width - 1) * active * elem

    state_read = 0.0
    if first:
        if state_len <= length:
            # Lines 202-235 copy the last state_len tokens of x into the state.
            state_read += state_len * active * elem
            state_read += 4.0  # output cache-index coordinate, line 219-223
        else:
            # Lines 237-307 shift the old state and append the few x tokens.
            if has_initial:
                state_read += (state_len - length) * active * elem
            state_read += length * active * elem
    state_store = state_len * active * elem if first else 0.0
    # Lines 234/269/383: the barrier is conditional.  The S <= L path always
    # barriers before its store; the S > L path barriers only when the old
    # state is shifted and selected with tl.where.
    barrier = first and (state_len <= length or has_initial)
    state_rewrite = first and state_len > length and has_initial
    state_vector = float(BLOCK_N * np2) if state_rewrite else 0.0

    conv_vector = float(BLOCK_N * seg) * spec.per_token_vector_ops
    if spec.bias:
        # acc_preload conversion applies to the full padded lane vector.
        conv_vector += float(BLOCK_N)

    output = active * seg * elem
    return _TileWork(
        prep_hbm=prep, conv_load_hbm=conv_load, state_read_hbm=state_read,
        state_store_hbm=state_store, output_hbm=output,
        conv_vector=conv_vector, state_vector=state_vector,
        barrier=barrier, state_rewrite=state_rewrite, first=first,
    )


def _sequence_totals(spec: _ConvSpec, length: int, has_initial: bool, skipped: bool):
    """Closed-form sum over all feature blocks and chunks, O(1) in tokens."""
    chunks = _ceil_div(length, BLOCK_M)
    blocks = spec.feature_blocks
    if not chunks:
        return 0.0, 0.0, 0, 0, 0
    if skipped:
        # Every chunk/feature CTA early-returns after the 20 scalar bytes.
        return 20.0 * chunks * blocks, 0.0, blocks, chunks, 0
    padded_channels = blocks * BLOCK_N
    elem = spec.element_bytes
    state_len = spec.state_len

    prep = chunks * spec.width * spec.channels * elem
    if spec.bias:
        prep += chunks * spec.channels * elem
    if has_initial:
        prep += state_len * spec.channels * elem
    prep += 20.0 * chunks * blocks
    prep += 1.0 * blocks  # has_initial_states load in each chunk-0 program

    conv_load = spec.channels * length * elem
    if chunks > 1:
        conv_load += (chunks - 1) * spec.channels * (spec.width - 1) * elem

    state_read = 0.0
    if state_len <= length:
        state_read += state_len * spec.channels * elem
        state_read += 4.0 * blocks
    else:
        if has_initial:
            state_read += (state_len - length) * spec.channels * elem
        state_read += length * spec.channels * elem

    state_store = state_len * spec.channels * elem
    output = spec.channels * length * elem

    conv_vector = padded_channels * (
        length * spec.per_token_vector_ops + (chunks if spec.bias else 0.0)
    )
    state_vector = (
        float(padded_channels * spec.np2_state_len)
        if (state_len > length and has_initial) else 0.0
    )
    barriers = blocks if (state_len <= length or has_initial) else 0
    hbm = prep + conv_load + state_read + state_store + output
    return hbm, conv_vector + state_vector, blocks, chunks, barriers


def _total_work(spec: _ConvSpec):
    hbm = 0.0
    vector = 0.0
    barriers = 0
    skipped = spec.skipped
    for length, has_initial, skip in zip(
            spec.lengths, spec.has_initial_state, skipped):
        seq_hbm, seq_vector, _blocks, _chunks, seq_barriers = _sequence_totals(
            spec, length, has_initial, skip)
        hbm += seq_hbm
        vector += seq_vector
        barriers += seq_barriers
    return Work(resources={"hbm": hbm, "vector": vector}), barriers


def _require_owned_units(context: RecipeContext, owners: int, roles) -> None:
    """Reject a geometry that the shared unit partition cannot map exactly."""
    for role in roles:
        if role not in context.resource_names:
            continue
        units = context.resource_units.get(role, context.core_count)
        compatible = units >= 1 and (
            (units >= owners and units % owners == 0)
            or (units < owners and owners % units == 0))
        if not compatible:
            raise ModelingError(
                "causal_conv_prefill_owner_geometry_incompatible:"
                f"{role}:{owners}:{units}")


def _tile_stages(tile_work: _TileWork, hbm: str, vector: str, owned) -> tuple[Stage, ...]:
    """Build one CTA's stage DAG with vector service pinned to its owner."""
    if tile_work.early_return:
        # Global-memory scalar loads are shared; no CTA arithmetic is executed.
        return (Stage("scalar", {hbm: tile_work.prep_hbm}),)

    stages = [Stage("prep", {hbm: tile_work.prep_hbm})]
    if tile_work.first:
        # Source order 151-307: state read/shift, optional compiler barrier,
        # optional tl.where, then the state store.
        stages.append(Stage("state_read", {hbm: tile_work.state_read_hbm},
                            after=("prep",)))
        if tile_work.barrier:
            stages.append(Stage("barrier", after=("state_read",), placement=owned))
            if tile_work.state_rewrite:
                stages.append(Stage("state_rewrite", {vector: tile_work.state_vector},
                                    after=("barrier",), placement=owned))
                stages.append(Stage("state_store", {hbm: tile_work.state_store_hbm},
                                    after=("state_rewrite",)))
            else:
                stages.append(Stage("state_store", {hbm: tile_work.state_store_hbm},
                                    after=("barrier",)))
        else:
            # S > L and no initial state: no barrier and no tl.where.
            stages.append(Stage("state_store", {hbm: tile_work.state_store_hbm},
                                after=("state_read",)))
        conv_deps = ("prep", "state_store")
    else:
        conv_deps = ("prep",)
    stages.append(Stage("conv_load", {hbm: tile_work.conv_load_hbm}, after=conv_deps))
    stages.append(Stage("conv_compute", {vector: tile_work.conv_vector},
                        after=("conv_load",), placement=owned))
    stages.append(Stage("store", {hbm: tile_work.output_hbm},
                        after=("conv_compute",)))
    return tuple(stages)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Bind source work to lazy per-SM CTA streams over the source grid."""
    spec = _spec_from_metadata(op.metadata)
    work, _barriers = _total_work(spec)
    skipped = spec.skipped
    chunks_per_sequence = _chunks_per_sequence(spec)
    feature_blocks = spec.feature_blocks
    flat_ends = []
    total_chunks = 0
    for chunks in chunks_per_sequence:
        total_chunks += chunks
        flat_ends.append(total_chunks)
    total_programs = total_chunks * feature_blocks
    if total_programs < 0 or geometry.compute_units < 1:
        raise ModelingError("causal_conv_prefill_program_geometry_invalid")
    owners = max(1, int(geometry.compute_units))

    def program(context: RecipeContext):
        names = context.resource_names
        for role in ("hbm", "vector"):
            if role not in names:
                raise ModelingError(f"physical_resource_rate_missing:{role}")
        hbm = names["hbm"]
        vector = names["vector"]
        if int(context.core_count) != owners:
            raise ModelingError(
                "causal_conv_prefill_owner_count_mismatch:"
                f"{owners}:{context.core_count}")
        if total_programs == 0:
            return ParallelProgram(())
        _require_owned_units(context, owners, ("vector",))

        def source_program(index: int) -> _TileWork:
            # CUDA enumerates x fastest.  The source grid's x dimension is the
            # sequence-major flat chunk and y is the channel block.
            feature_block, flat_chunk = divmod(index, total_chunks)
            # bisect_right skips duplicate ends for zero-length sequences.
            sequence_index = bisect_right(flat_ends, flat_chunk)
            sequence_start = flat_ends[sequence_index - 1] if sequence_index else 0
            chunk_offset = flat_chunk - sequence_start
            length = spec.lengths[sequence_index]
            return _tile_work(
                spec, length, chunk_offset, feature_block,
                spec.has_initial_state[sequence_index],
                skipped[sequence_index],
            )

        def core_program(owner: int):
            count = (total_programs - 1 - owner) // owners + 1
            if count <= 0:
                return None
            owned = context.owned(owner, owners, ("vector",))

            def tile(index: int):
                tile_work = source_program(owner + index * owners)
                return _tile_stages(tile_work, hbm, vector, owned)

            return PipelineProgram(count, min(OWNER_CTA_WINDOW, count), tile)

        streams = tuple(
            stream for stream in (
                core_program(owner) for owner in range(min(owners, total_programs)))
            if stream is not None)
        return ParallelProgram(streams)

    return OperatorPlan(work, program)


__all__ = ["build", "make_op", "IMPLEMENTATION", "KERNEL_FUNCTION"]
