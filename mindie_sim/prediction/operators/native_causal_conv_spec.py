"""Native arch35 CausalConv1d FN geometry and host dispatch.

[Role] Adapt runMode=0 varlen parameters and reproduce CUTBS/CUTBSD tiling.
[Invariants] Launch width uses AIV count. Device-visible all-false flags still
  select snapshot workspace and SyncAll; CPU-visible all-false flags do not.
  Absent convStates clears the initial-state branch. Reserved workspace size
  and copied history payload remain distinct.
[Timeouts & Locks] O(batch + token tiles) metadata; no locks or external work.
[Side-Effect Policy] Pure descriptors; no device, measurement or filesystem I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...schema.operators import OpSpec
from ..mechanism import Geometry

IMPLEMENTATION = "causal_conv1d_arch35_fn"

__all__ = ["make_op", "host_tiling", "IMPLEMENTATION"]
SOURCE_REVISION = "a4a795857868729fbdbb0118c1e06c8655a7b31b"
SOURCE_IMAGE = "sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4"

MAX_WIDTH = 4
MAX_BLOCK_DIM = 4096
RING_SLOTS = 5
OUT_SLOTS = 2
CALC_SLOTS = 8
DIM_ALIGN_ELEMS = 16
MAX_DIM_TILE_SIZE = 4096
UB_RESERVED_BYTES = 512
FN_UB_RESERVED_BYTES = 512
DEFAULT_CORE_NUM = 64
DEFAULT_UB_BYTES = 253952
ASCENDC_RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024
INIT_STATE_SYNC_BYTES_PER_BLOCK = 32
PAIR_TOKENS = 2
# Bounded lazy issue window for the per-token rolling/generic schedule.  It is a
# scheduling window, not a source buffer size: deps still gate every stage.
TOKEN_PIPELINE_SLOTS = 6

# Source pipe.InitBuffer sizes: 5*4096*T + 2*4096*T + 8*4096*float.
UB_RESIDENT_BYTES = (RING_SLOTS * MAX_BLOCK_DIM * 2
                     + OUT_SLOTS * MAX_BLOCK_DIM * 2
                     + CALC_SLOTS * MAX_BLOCK_DIM * 4)

_METADATA_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float": "fp32"}


@dataclass(frozen=True)
class _Dtypes:
    query_start_loc: str
    cache_indices: str
    has_initial_state: str


@dataclass(frozen=True)
class _Config:
    lengths: tuple[int, ...]
    dim: int
    width: int
    dtype: str
    activation: str
    bias: bool
    metadata_placement: str
    dtypes: _Dtypes
    has_initial_state: tuple[bool, ...] | None
    cache_indices: tuple[int, ...] | None
    state_history_slots: int
    state_rows: int
    null_block_id: int | None
    pad_slot_id: int
    qsl: tuple[int, ...]

    @property
    def batch(self):
        return len(self.lengths)

    @property
    def cu_seqlen(self):
        return self.qsl[-1]

    @property
    def history_count(self):
        return self.width - 1

    @property
    def has_conv_states(self):
        return self.state_rows > 0 and self.state_history_slots > 0

    @property
    def has_init_flag(self):
        if not self.has_conv_states or self.has_initial_state is None:
            return 0
        if self.metadata_placement == "cpu":
            return 1 if any(self.has_initial_state) else 0
        return 1

    @property
    def has_init_workspace(self):
        return int(self.has_conv_states and self.has_init_flag != 0)

    def has_init(self, seq):
        # ResolveSeqHasInit(): HasConvStates() && hasInitialState && value != 0.
        # Validation forces tilingData.hasInitialState=0 when convStates is
        # absent, so an actual true flag cannot select the cache/snapshot path.
        if not self.has_conv_states or self.has_init_flag == 0:
            return False
        return bool(self.has_initial_state[seq]) if self.has_initial_state is not None else False

    def dtype_bytes(self):
        return 2 if self.dtype in {"bf16", "fp16"} else 4


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_down(value, align):
    return 0 if value <= 0 or align <= 0 else (value // align) * align


def _align_up(value, align):
    return 0 if value <= 0 or align <= 0 else _ceil_div(value, align) * align


def _int_param(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if float(value) != float(number):
        raise ValueError(f"{name} must be an integer")
    return number


def _bool_param(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a bool or 0/1")


def _normalize_dtype(value):
    text = str(value).strip().lower().replace("-", "_")
    text = _METADATA_ALIASES.get(text, text)
    if text not in {"bf16", "fp16"}:
        raise ValueError(f"native_causal_conv1d_fn supports bf16/fp16, got {value!r}")
    return text


def _normalize_activation(value):
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if text in {"none", "silu", "swish"}:
        return "silu" if text in {"silu", "swish"} else "none"
    raise ValueError(f"activation must be none/silu/swish, got {value!r}")


def _normalize_metadata_dtype(value, name, *, allow_bool=False):
    text = str(value).strip().lower()
    aliases = {"bool": "bool", "boolean": "bool", "int32": "int32",
               "int64": "int64", "i32": "int32", "i64": "int64"}
    text = aliases.get(text)
    if text is None or (text == "bool" and not allow_bool):
        raise ValueError(f"{name} metadata dtype must be int32{('/bool' if allow_bool else '')}")
    return text


def _parse_dtypes(params, placement):
    raw = params.get("metadata_dtypes") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("metadata_dtypes must be a mapping")
    default = {"query_start_loc": "int32", "cache_indices": "int32",
               "has_initial_state": "bool"} if placement == "device" else {
                   "query_start_loc": "int64", "cache_indices": "int64",
                   "has_initial_state": "int64"}
    dtypes = _Dtypes(
        _normalize_metadata_dtype(raw.get("query_start_loc", default["query_start_loc"]),
                                  "query_start_loc"),
        _normalize_metadata_dtype(raw.get("cache_indices", default["cache_indices"]),
                                  "cache_indices"),
        _normalize_metadata_dtype(raw.get("has_initial_state", default["has_initial_state"]),
                                  "has_initial_state", allow_bool=True),
    )
    if placement == "cpu" and any(value != "int64" for value in dtypes.__dict__.values()):
        raise ValueError("CPU-visible metadata is only accepted as int64")
    if placement == "device" and (dtypes.query_start_loc != "int32"
                                  or dtypes.cache_indices != "int32"):
        raise ValueError("device queryStartLoc/cacheIndices must be int32")
    if placement == "device" and dtypes.has_initial_state not in {"bool", "int32"}:
        raise ValueError("device hasInitialState must be bool or int32")
    return dtypes


def _dtype_width(name):
    return {"int32": 4, "int64": 8, "bool": 1}[name]


def _parse_params(params) -> _Config:
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    raw_lengths = params.get("lengths", params.get("seqlens"))
    if raw_lengths is None:
        total = params.get("total_tokens")
        if total is None:
            raise ValueError("lengths are required")
        raw_lengths = [total]
    if isinstance(raw_lengths, (str, bytes)):
        raise ValueError("lengths must be a sequence")
    lengths = tuple(_int_param(value, "lengths[]") for value in raw_lengths)
    if not lengths or any(value < 0 for value in lengths):
        raise ValueError("lengths must be non-empty and non-negative")
    total = sum(lengths)
    if total <= 0:
        raise ValueError("native causal_conv1d FN requires cuSeqlen > 0")
    if total > 2**31 - 1:
        raise ValueError("cuSeqlen exceeds the source int32 indexing limit")

    if "dim" not in params or "width" not in params:
        raise ValueError("dim and width are required")
    dim = _int_param(params["dim"], "dim")
    if dim <= 0 or dim % DIM_ALIGN_ELEMS:
        raise ValueError("dim must be positive and divisible by 16")
    width = _int_param(params["width"], "width")
    if width < 2 or width > MAX_WIDTH:
        raise ValueError("width must be in [2, 4]")

    dtype = _normalize_dtype(params.get("dtype", "bf16"))
    activation = _normalize_activation(params.get("activation", "silu"))
    bias = _bool_param(params.get("bias", False), "bias")

    input_mode = str(params.get("input_mode", "varlen")).lower()
    if input_mode not in {"varlen", "0"}:
        raise ValueError("selected source branch requires 2D varlen inputMode=0")
    run_mode = _int_param(params.get("run_mode", 0), "run_mode")
    if run_mode != 0:
        raise ValueError("selected source branch requires runMode=0 (fn/prefill)")
    head_num = _int_param(params.get("head_num", 0), "head_num")
    if head_num != 0:
        raise ValueError("head-major output is not used by the fixed cases")
    if params.get("num_accepted_tokens") not in (None, [], ()):
        raise ValueError("numAcceptedTokens belongs to the update branch")
    if _int_param(params.get("block_size_to_align", 0), "block_size_to_align") != 0:
        raise ValueError("block_size_to_align is not supported by the native source")

    placement = str(params.get("metadata_placement", "device")).lower()
    if placement not in {"cpu", "device"}:
        raise ValueError("metadata_placement must be cpu or device")
    dtypes = _parse_dtypes(params, placement)

    if "state_history_slots" in params:
        history = _int_param(params["state_history_slots"], "state_history_slots")
    elif "state_len" in params:
        history = _int_param(params["state_len"], "state_len")
    else:
        # convStates is an optional input.  An explicit zero-row descriptor
        # denotes its absence just as well as the two zero-valued fields.
        rows_hint = params.get("state_rows", params.get("num_cache_lines"))
        absent_hint = (rows_hint is not None
                       and _int_param(rows_hint, "state_rows") == 0)
        history = 0 if absent_hint else width - 1
    if history < 0:
        raise ValueError("state_history_slots must be non-negative")
    if history and history < width - 1:
        raise ValueError("convStates stateLen must be >= width-1")

    raw_cache = params.get("cache_indices")
    if raw_cache is not None:
        if isinstance(raw_cache, (str, bytes)):
            raise ValueError("cache_indices must be a sequence")
        cache = tuple(_int_param(value, "cache_indices[]") for value in raw_cache)
        if len(cache) != len(lengths):
            raise ValueError("cache_indices must have batch entries")
    else:
        cache = None

    null_raw = params.get("null_block_id", 0)
    null_block_id = None if null_raw is None else _int_param(null_raw, "null_block_id")
    pad_slot_id = _int_param(params.get("pad_slot_id", -1), "pad_slot_id")

    if "state_rows" in params:
        state_rows = _int_param(params["state_rows"], "state_rows")
    elif "num_cache_lines" in params:
        state_rows = _int_param(params["num_cache_lines"], "num_cache_lines")
    elif history and cache is not None:
        valid = [value for value in cache
                 if value != pad_slot_id
                 and not (null_block_id is not None and value == null_block_id)]
        state_rows = max(1, max(valid) + 1) if valid else max(1, len(lengths))
    elif history:
        state_rows = len(lengths)
    else:
        state_rows = 0
    if state_rows < 0:
        raise ValueError("state_rows must be non-negative")
    if history and state_rows == 0:
        raise ValueError("convStates rows must be positive when stateLen is positive")
    if not history and state_rows:
        # No usable history: HasConvStates() is false.
        state_rows = 0

    if cache is not None:
        for value in cache:
            skipped = (value == pad_slot_id
                       or (null_block_id is not None and value == null_block_id))
            if skipped:
                continue
            if state_rows and not 0 <= value < state_rows:
                raise ValueError(f"cacheIndices value {value} out of range [0,{state_rows})")
    if cache is None and state_rows and state_rows < len(lengths):
        raise ValueError("identity cache mapping requires num_cache_lines >= batch")

    raw_initial = params.get("has_initial_state")
    if raw_initial is not None:
        if isinstance(raw_initial, (str, bytes)):
            raise ValueError("has_initial_state must be a sequence")
        initial = tuple(_bool_param(value, "has_initial_state[]") for value in raw_initial)
        if len(initial) != len(lengths):
            raise ValueError("has_initial_state must have batch entries")
        if placement == "cpu" and any(value not in (0, 1) for value in initial):
            raise ValueError("CPU has_initial_state accepts only 0/1")
    else:
        initial = None

    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + length)
    if params.get("query_start_loc") is not None:
        supplied = tuple(_int_param(value, "query_start_loc[]")
                         for value in params["query_start_loc"])
        if supplied != tuple(qsl):
            raise ValueError("query_start_loc is inconsistent with lengths")

    return _Config(
        lengths=lengths, dim=dim, width=width, dtype=dtype,
        activation=activation, bias=bias, metadata_placement=placement,
        dtypes=dtypes, has_initial_state=initial, cache_indices=cache,
        state_history_slots=history, state_rows=state_rows,
        null_block_id=null_block_id, pad_slot_id=pad_slot_id, qsl=tuple(qsl),
    )


def _host_tiling(cfg: _Config, core_num: int, ub_bytes: int) -> dict:
    """Literal translation of causal_conv1d_tiling_planner.h lines 76-301."""
    invalid = {
        "valid": False, "execution_plan": "INVALID", "base_dim": 0,
        "base_dim_count": 0, "token_core_budget": 0, "token_block_size": 0,
        "token_block_count": 0, "token_blocks_per_core": 0,
        "token_core_tail_count": 0, "block_dim": 0, "grid_size": 0,
        "effective_grid_size": 0, "ub_limited_base_dim": 0,
        "core_count": int(core_num), "ub_bytes": int(ub_bytes),
        "has_initial_state": cfg.has_init_flag,
        "has_init_state_workspace": 0, "sync_workspace_bytes": 0,
        "snapshot_workspace_bytes": 0, "workspace_bytes": 0,
        "has_explicit_token_seq_ranges": 0,
    }
    if (core_num <= 0 or ub_bytes <= 0 or cfg.batch <= 0 or cfg.cu_seqlen <= 0
            or cfg.dim <= 0):
        return invalid

    if cfg.dim <= MAX_DIM_TILE_SIZE:
        execution_plan = "CUTBS"
        base_dim = cfg.dim
        base_dim_count = 1
        ub_limited = 0
    else:
        execution_plan = "CUTBSD"
        if ub_bytes <= FN_UB_RESERVED_BYTES:
            return invalid
        bytes_per_elem = (RING_SLOTS * 2) + (OUT_SLOTS * 2) + (CALC_SLOTS * 4)
        ub_limited = min(MAX_DIM_TILE_SIZE,
                         _align_down((int(ub_bytes) - FN_UB_RESERVED_BYTES)
                                     // bytes_per_elem, DIM_ALIGN_ELEMS))
        if ub_limited <= 0:
            return invalid
        base_dim = ub_limited
        base_dim_count = _ceil_div(cfg.dim, base_dim)
        if (1 < base_dim_count < core_num and core_num % base_dim_count):
            adjusted = base_dim_count
            while adjusted < core_num and core_num % adjusted:
                adjusted += 1
            if adjusted < core_num:
                candidate = _align_up(_ceil_div(cfg.dim, adjusted), DIM_ALIGN_ELEMS)
                if 0 < candidate <= ub_limited and candidate <= MAX_DIM_TILE_SIZE:
                    base_dim = candidate
                    base_dim_count = _ceil_div(cfg.dim, base_dim)

    token_core_budget = int(core_num)
    if execution_plan == "CUTBSD":
        token_core_budget = max(1, token_core_budget // base_dim_count)
    ideal = _ceil_div(cfg.cu_seqlen, token_core_budget)
    token_block_size = ideal if ideal > 0 else 1
    token_block_count = _ceil_div(cfg.cu_seqlen, token_block_size)
    token_blocks_per_core = _ceil_div(token_block_count, token_core_budget)
    token_core_tail = token_block_count - max(0, token_blocks_per_core - 1) * token_core_budget
    if token_core_tail <= 0:
        token_core_tail = token_core_budget
    mapping_block_dim = min(token_core_budget * base_dim_count, int(core_num))
    effective_grid = token_block_count * base_dim_count
    block_dim = min(effective_grid, mapping_block_dim)
    if block_dim <= 0:
        return invalid

    explicit = int(cfg.metadata_placement == "cpu"
                   and cfg.dtypes.query_start_loc == "int64"
                   and token_block_count <= 128)
    workspace = cfg.has_init_workspace
    sync_bytes = block_dim * INIT_STATE_SYNC_BYTES_PER_BLOCK if workspace else 0
    snapshot_bytes = (cfg.batch * cfg.history_count * cfg.dim * 2
                      if workspace else 0)
    workspace_bytes = (ASCENDC_RESERVED_WORKSPACE_SIZE + sync_bytes + snapshot_bytes
                       if workspace else 0)
    return {
        "valid": True,
        "execution_plan": execution_plan,
        "base_dim": base_dim,
        "base_dim_count": base_dim_count,
        "token_core_budget": token_core_budget,
        "token_block_size": token_block_size,
        "token_block_count": token_block_count,
        "token_blocks_per_core": token_blocks_per_core,
        "token_core_tail_count": token_core_tail,
        "block_dim": block_dim,
        "grid_size": token_block_count * base_dim_count,
        "effective_grid_size": effective_grid,
        "ub_limited_base_dim": ub_limited,
        "core_count": int(core_num),
        "ub_bytes": int(ub_bytes),
        "has_initial_state": cfg.has_init_flag,
        "has_init_state_workspace": workspace,
        "sync_workspace_bytes": sync_bytes,
        "snapshot_workspace_bytes": snapshot_bytes,
        "workspace_bytes": workspace_bytes,
        "has_explicit_token_seq_ranges": explicit,
    }


def _config_from_op(op: OpSpec) -> _Config:
    meta = dict(op.metadata)
    dtypes = _Dtypes(**dict(meta["metadata_dtypes"]))
    lengths = tuple(int(v) for v in meta["lengths"])
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + length)
    return _Config(
        lengths=lengths, dim=int(meta["dim"]),
        width=int(meta["width"]), dtype=str(meta["dtype"]),
        activation=str(meta["activation"]), bias=bool(meta["bias"]),
        metadata_placement=str(meta["metadata_placement"]), dtypes=dtypes,
        has_initial_state=(tuple(bool(v) for v in meta["has_initial_state"])
                           if meta.get("has_initial_state") is not None else None),
        cache_indices=(tuple(int(v) for v in meta["cache_indices"])
                       if meta.get("cache_indices") is not None else None),
        state_history_slots=int(meta["state_history_slots"]),
        state_rows=int(meta["state_rows"]),
        null_block_id=(None if meta.get("null_block_id") is None
                       else int(meta["null_block_id"])),
        pad_slot_id=int(meta["pad_slot_id"]),
        qsl=tuple(qsl),
    )


def make_op(params: dict) -> OpSpec:
    """Explicit geometry adapter for the native arch35 FN varlen branch."""
    cfg = _parse_params(params)
    host = _host_tiling(cfg, DEFAULT_CORE_NUM, DEFAULT_UB_BYTES)
    if not host["valid"]:
        raise ValueError("native causal_conv1d FN cannot tile this geometry")
    width_key = {2: 1, 3: 2, 4: 3}[cfg.width]
    fn_plan_key = 1 if host["execution_plan"] == "CUTBS" else 2
    dispatch = {
        "implementation": IMPLEMENTATION,
        "source_branch": "arch35_fn_varlen",
        "activation": cfg.activation,
        "activation_mode": 1 if cfg.activation == "silu" else 0,
        "bias": cfg.bias,
        "has_bias": cfg.bias,
        "run_mode": 0,
        "input_mode": "varlen",
        "execution_plan": host["execution_plan"],
        "fn_plan_key": fn_plan_key,
        "width": cfg.width,
        "width_key": width_key,
        "dtype": cfg.dtype,
        "metadata_placement": cfg.metadata_placement,
        "has_cache_indices": cfg.cache_indices is not None,
        "has_initial_state": bool(host["has_initial_state"]),
        "has_init_state_workspace": bool(host["has_init_state_workspace"]),
        "has_explicit_token_seq_ranges": bool(host["has_explicit_token_seq_ranges"]),
        "has_num_accepted_tokens": False,
        "head_num": 0,
        "has_conv_states": bool(cfg.has_conv_states),
        "state_history_slots": cfg.state_history_slots,
        "state_rows": cfg.state_rows,
        "rolling_fast_path": not cfg.bias,
        "generic_run_seq": cfg.bias,
        "rolling_math": "cann_default_ftz_true_intrinsic",
        "rolling_cost": "source_register_capacity_projection",
        "null_block_id": cfg.null_block_id,
        "pad_slot_id": cfg.pad_slot_id,
        "sync_all": bool(host["has_init_state_workspace"]),
        "one_native_launch": True,
    }
    metadata = {
        "implementation": IMPLEMENTATION,
        "source_revision": SOURCE_REVISION,
        "source_image": SOURCE_IMAGE,
        "lengths": cfg.lengths,
        "dim": cfg.dim,
        "width": cfg.width,
        "dtype": cfg.dtype,
        "activation": cfg.activation,
        "activation_mode": 1 if cfg.activation == "silu" else 0,
        "bias": cfg.bias,
        "has_bias": cfg.bias,
        "metadata_placement": cfg.metadata_placement,
        "metadata_dtypes": {
            "query_start_loc": cfg.dtypes.query_start_loc,
            "cache_indices": cfg.dtypes.cache_indices,
            "has_initial_state": cfg.dtypes.has_initial_state,
        },
        "has_initial_state": cfg.has_initial_state,
        "cache_indices": cfg.cache_indices,
        "state_history_slots": cfg.state_history_slots,
        "state_rows": cfg.state_rows,
        "null_block_id": cfg.null_block_id,
        "pad_slot_id": cfg.pad_slot_id,
        "dispatch": dispatch,
        "host_tiling": host,
        "default_core_num": DEFAULT_CORE_NUM,
        "default_ub_bytes": DEFAULT_UB_BYTES,
        "run_mode": 0,
        "head_num": 0,
        "source_kernel_launches": 1,
        "state_update": "sequence_tail",
        "workspace_enabled": bool(host["has_init_state_workspace"]),
        "workspace_bytes": host["workspace_bytes"],
        "sync_workspace_bytes": host["sync_workspace_bytes"],
        "snapshot_workspace_bytes": host["snapshot_workspace_bytes"],
        "reserved_workspace_bytes": (ASCENDC_RESERVED_WORKSPACE_SIZE
                                     if host["has_init_state_workspace"] else 0),
    }
    padded_tokens = host["token_block_count"] * host["token_block_size"]
    semantic = (cfg.cu_seqlen, cfg.dim)
    padded = (padded_tokens, cfg.dim)
    return OpSpec(
        op_id=f"native-causal-conv1d-fn:T{cfg.cu_seqlen}:D{cfg.dim}:W{cfg.width}",
        family="causal_conv1d_fn", phase="prefill", stage=0, layer=0,
        requests=(), semantic_shape=semantic, local_shape=semantic,
        padded_shape=padded, launch_shape=padded, dtype=cfg.dtype,
        recipe=IMPLEMENTATION, source_boundary="kernel", layout="TN", metadata=tuple(sorted(metadata.items())),
        notes=("source-selected arch35 runMode=0 varlen FN",),
    )


def _default_geometry():
    return Geometry(32, {"vector": DEFAULT_CORE_NUM}, {"ub": DEFAULT_UB_BYTES},
                    {"ub": DEFAULT_CORE_NUM})


def host_tiling(op_or_params, geometry=None, *, core_num=None, ub_bytes=None) -> dict:
    """Public host-tiling projection for independent comparison.

    ``geometry`` supplies AIV count and UB size.  With no geometry, the source
    hardware defaults (64 AIV, 253952 UB) are used; these are the values behind
    the coordinator's fixed expected_planner entries.
    """
    if isinstance(op_or_params, OpSpec):
        cfg = _config_from_op(op_or_params)
    else:
        cfg = _parse_params(op_or_params)
    if geometry is not None:
        core_num = geometry.resource_units.get("vector", geometry.compute_units)
        ub_bytes = geometry.buffers.get("ub", DEFAULT_UB_BYTES)
    return _host_tiling(cfg,
                        DEFAULT_CORE_NUM if core_num is None else int(core_num),
                        DEFAULT_UB_BYTES if ub_bytes is None else int(ub_bytes))
