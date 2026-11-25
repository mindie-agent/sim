"""Acceptance-bound RecoverSSM commit intent and physical state publication.

[Role]
  A recurrent builder emits a compact-record intent; after sampling/acceptance
  has been assembled this module materializes exactly one commit per
  request/layer state owner. The commit depends on the verify record, the real
  acceptance event, and the base state version so an unaccepted state version
  can never become visible.

[Invariants]
  * Integer commitment comes from ``RequestView.draft_advanced_tokens`` or a
    declared body input; a configured mean is never rounded into a commit.
  * GDN and upstream-exact KDA (GA-09) execute ``2 * (sum u_i + N_act) * E``
    reverse-accumulation matrix ops plus align-mode boundary work (exact
    ``gdn_recoverssm_commit.py:144-146`` shorter-prefix guard) and per-token
    scalar work tracked distinctly from matrix work.
  * Unique record bytes (``1,028 B/head`` GDN; ``1,024 / 1,536 B/head`` KDA)
    and physical tiled GDN reads ``T_acc * H_v * (4*Dv + N_v*(4*Dk + 4))`` are
    accounted cleanly without conflation.
  * State publication enforces strict monotonic ``published_version == base + 1``
    and rejects duplicate commits per ``(owner, scope, layer, version)``.
  * Emits schema-valid Triton ``KernelBinding`` and ``NativeCallBinding`` with
    family-specific ``kernel_id``, runtime ``arch_family``, and binding-derived
    ``readiness``.

[Timeouts & Locks]
  Pure record construction; no I/O, locks or Engine mutation.

[Side-Effect Policy]
  Returns replacement immutable operators only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from mindie_sim.prediction.native_binding import (
    NATIVE_CALL_METADATA_KEY,
    bind_native_call,
    declare_native_call,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)
from .recover_ssm_work import FP32_RECORD_BYTES, commit_tokens_for_view

REVERSE_FMA_OPS_PER_STATE_ELEMENT = 2
GDN_RECORD_SCALAR_FIELDS = 1
KDA_KG_SCALAR_OPS_PER_KEY_ELEM = 9
GDN_KEY_DECAY_SCALAR_OPS_PER_KEY_ELEM = 2
GDN_COMMIT_HEAD_TILE = 2
GDN_COMMIT_UB_MATRICES = 2
GDN_BOUNDARY_COMMIT_UB_MATRICES = 6
GDN_COMMIT_UB_HEADROOM_ROWS = 2
GDN_DEFAULT_VALUE_TILE = 64
GDN_GROUPED_DIM = 128
A3_UB_BYTES = 196_608
A5_UB_BYTES = 262_144
DEFAULT_NULL_BLOCK_ID = -1
ALIGN_DEFAULT_NULL_BLOCK_ID = 0
COMMIT_KERNEL_BACKEND = "Triton"
GDN_COMMIT_KERNEL_ID = "gdn_recoverssm_commit"
KDA_COMMIT_KERNEL_ID = "kda_recoverssm_commit"
KDA_VERIFY_KERNEL_NAME = "_kda_recoverssm_verify_kernel"
KDA_COMMIT_PLAN_KERNEL_NAME = "_prepare_commit_plan_kernel"
KDA_CONV_COMPACT_KERNEL_NAME = "_compact_conv_state_kernel"
GDN_VERIFY_KERNEL_NAME = "gdn_recoverssm_verify"
KDA_PLAN_INPUT_ARRAYS_BASE = 3
KDA_PLAN_INPUT_ARRAYS_ALIGN = 5
KDA_PLAN_OUTPUT_ARRAYS = 6
KDA_PLAN_OPS_PER_REQ = 16
KDA_CONV_DEFAULT_HISTORY_LEN = 3
KDA_CONV_ELEMENT_BYTES = 2
KDA_CONV_CHANNELS_PER_HEAD_FACTOR = 3

INTENT_STAGE = "commit_intent"
BODY_BOUNDARY = "declared_body_input"
ACCEPTANCE_BOUNDARY = "sampling_acceptance"


def _next_power_of_two(value: int) -> int:
    n = max(1, int(value))
    return 1 << (n - 1).bit_length()


def _is_a5_arch_family(arch_family: str) -> bool:
    low = str(arch_family or "").strip().lower()
    return any(tok in low for tok in ("3510", "35", "950", "a5"))


def _is_align_mode_active(metadata: Mapping[str, Any], runtime: Any) -> bool:
    if "align_mode" in metadata:
        return bool(metadata["align_mode"])
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    if "align_mode" in rt_meta:
        return bool(rt_meta["align_mode"])
    cache_mode = str(rt_meta.get("mamba_cache_mode") or "").strip().lower()
    if cache_mode:
        return cache_mode == "align"
    if runtime is not None and bool(getattr(runtime, "enable_prefix_caching", False)):
        return True
    return bool(rt_meta.get("enable_prefix_caching", False))


KDA_GROUPED_HEAD_COUNTS = frozenset({32, 64})


def resolve_commit_tile_geometry(
    *,
    family_kind: str,
    dk: int,
    dv: int,
    heads: int,
    arch_family: str,
    state_dtype: str = "fp32",
    align_mode: bool = False,
    ub_bytes: int | None = None,
) -> tuple[int, int, int]:
    """Exact upstream Triton ``(BV, v_tiles, head_groups)`` for GDN and KDA commit kernels.

    Both ``kda/recoverssm.py:1467`` and ``gdn_recoverssm_commit.py:412`` size UB
    from FP32 correction/replay data (4 B/elem). For KDA, pass ``align_mode=False``
    for the ordinary launch and ``align_mode=True`` exclusively for the separate
    boundary launch.
    """
    block_k = _next_power_of_two(dk)
    is_a5 = _is_a5_arch_family(arch_family)
    elem_bytes = FP32_RECORD_BYTES
    eff_ub = int(ub_bytes) if (ub_bytes is not None and int(ub_bytes) > 0) else (
        A5_UB_BYTES if is_a5 else A3_UB_BYTES
    )
    if family_kind == "kda":
        grouped = (
            is_a5
            and str(state_dtype).lower() == "fp32"
            and int(dk) == GDN_GROUPED_DIM
            and int(dv) == GDN_GROUPED_DIM
            and int(heads) in KDA_GROUPED_HEAD_COUNTS
        )
        value_tile = GDN_DEFAULT_VALUE_TILE
        if grouped:
            head_groups = max(1, int(heads) // GDN_COMMIT_HEAD_TILE)
        else:
            if is_a5:
                value_tile *= 2
            bytes_per_row = GDN_COMMIT_UB_MATRICES * block_k * elem_bytes
            max_rows = max(1, eff_ub // max(1, bytes_per_row) - GDN_COMMIT_UB_HEADROOM_ROWS)
            value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
            head_groups = max(1, int(heads))
        if align_mode:
            b_bytes = GDN_BOUNDARY_COMMIT_UB_MATRICES * block_k * elem_bytes
            if grouped:
                b_bytes *= GDN_COMMIT_HEAD_TILE
            b_max = max(1, eff_ub // max(1, b_bytes) - GDN_COMMIT_UB_HEADROOM_ROWS)
            value_tile = min(GDN_DEFAULT_VALUE_TILE, 1 << (b_max.bit_length() - 1))
        block_v = min(_next_power_of_two(dv), max(1, value_tile))
        v_tiles = max(1, math.ceil(max(1, int(dv)) / block_v))
        return block_v, v_tiles, head_groups
    grouped = (
        is_a5 and int(dk) == GDN_GROUPED_DIM and int(dv) == GDN_GROUPED_DIM
        and int(heads) >= GDN_COMMIT_HEAD_TILE
        and int(heads) % GDN_COMMIT_HEAD_TILE == 0
    )
    head_groups = max(1, int(heads) // GDN_COMMIT_HEAD_TILE) if grouped else max(1, int(heads))
    value_tile = GDN_DEFAULT_VALUE_TILE
    if not grouped and is_a5 and not align_mode:
        value_tile *= 2
    ub_matrices = GDN_BOUNDARY_COMMIT_UB_MATRICES if align_mode else GDN_COMMIT_UB_MATRICES
    bytes_per_row = ub_matrices * block_k * elem_bytes
    if grouped:
        bytes_per_row *= GDN_COMMIT_HEAD_TILE
    max_rows = max(1, eff_ub // max(1, bytes_per_row) - GDN_COMMIT_UB_HEADROOM_ROWS)
    value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
    block_v = min(_next_power_of_two(dv), max(1, value_tile))
    v_tiles = max(1, math.ceil(max(1, int(dv)) / block_v))
    return block_v, v_tiles, head_groups


def resolve_gdn_commit_tile_width(
    *, dk: int, dv: int, heads: int, arch_family: str, align_mode: bool = False,
    ub_bytes: int | None = None,
) -> tuple[int, int]:
    """Exact upstream ``gdn_recoverssm_commit.py:411-442`` value tile & tile count."""
    block_v, v_tiles, _ = resolve_commit_tile_geometry(
        family_kind="gdn", dk=dk, dv=dv, heads=heads, arch_family=arch_family,
        align_mode=align_mode, ub_bytes=ub_bytes,
    )
    return block_v, v_tiles


def compute_recover_ssm_commit_work(
    *,
    family_kind: str,
    accepted_per_request: Sequence[int],
    heads: int,
    dk: int,
    dv: int,
    state_dtype: str = "fp32",
    kg_dtype: str = "bf16",
    v_tiles: int = 1,
    boundary_ops: int = 0,
    boundary_write_bytes: float = 0.0,
    record_bytes_override: float | None = None,
    state_bytes_override: float | None = None,
) -> dict[str, Any]:
    """Single source of truth for G4 RecoverSSM commit work (shared with governed planner)."""
    active_counts = [int(a_i) for a_i in accepted_per_request if int(a_i) > 0]
    active_seqs = len(active_counts)
    total_tokens = sum(active_counts)
    state_elements = int(heads) * int(dk) * int(dv)
    if state_bytes_override is not None:
        per_seq_state = float(state_bytes_override)
    else:
        per_seq_state = float(state_elements) * float(dtype_bytes(state_dtype))
    if record_bytes_override is not None:
        unique_per_tok = float(record_bytes_override)
    elif family_kind == "gdn":
        unique_per_tok = float(int(heads) * FP32_RECORD_BYTES * (int(dv) + int(dk) + GDN_RECORD_SCALAR_FIELDS))
    else:
        unique_per_tok = float(int(heads)) * (
            float(int(dv)) * FP32_RECORD_BYTES + 2.0 * float(int(dk)) * float(dtype_bytes(kg_dtype))
        )
    unique_rec_read = float(total_tokens) * unique_per_tok
    eff_v_tiles = max(1, int(v_tiles))
    if record_bytes_override is not None and eff_v_tiles <= 1:
        phys_rec_read = unique_rec_read
    elif family_kind == "gdn":
        phys_rec_read = float(total_tokens) * float(int(heads)) * float(
            FP32_RECORD_BYTES * int(dv)
            + eff_v_tiles * FP32_RECORD_BYTES * (int(dk) + GDN_RECORD_SCALAR_FIELDS)
        )
    elif family_kind == "kda":
        phys_rec_read = float(total_tokens) * float(int(heads)) * (
            float(int(dv)) * FP32_RECORD_BYTES
            + float(eff_v_tiles) * 2.0 * float(int(dk)) * float(dtype_bytes(kg_dtype))
        )
    else:
        phys_rec_read = unique_rec_read
    matrix_ops = (
        REVERSE_FMA_OPS_PER_STATE_ELEMENT * (total_tokens + active_seqs) * state_elements
        if active_seqs > 0 else 0
    )
    scalar_coeff = (
        GDN_KEY_DECAY_SCALAR_OPS_PER_KEY_ELEM
        if family_kind == "gdn"
        else KDA_KG_SCALAR_OPS_PER_KEY_ELEM
    )
    scalar_aux_ops = total_tokens * int(heads) * int(dk) * scalar_coeff
    return {
        "active_sequences": active_seqs,
        "total_tokens": total_tokens,
        "state_elements": state_elements,
        "state_bytes_per_seq": per_seq_state,
        "record_bytes_per_token": unique_per_tok,
        "unique_record_read_bytes": unique_rec_read,
        "physical_tiled_record_read_bytes": phys_rec_read,
        "state_read_bytes": float(active_seqs) * per_seq_state,
        "state_write_bytes": float(active_seqs) * per_seq_state + float(boundary_write_bytes),
        "matrix_ops": matrix_ops,
        "boundary_ops": int(boundary_ops),
        "vector_ops": int(matrix_ops + int(boundary_ops)),
        "scalar_aux_ops": scalar_aux_ops,
    }


def _parse_version_number(raw: Any) -> tuple[int | None, str]:
    text = str(raw or "").strip()
    if not text:
        return 0, "empty"
    if text.startswith("v") and text[1:].isdigit():
        return int(text[1:]), "v_prefix"
    if text.isdigit():
        return int(text), "int"
    if ":v" in text:
        tail = text.rsplit(":v", 1)[-1]
        if tail.isdigit():
            return int(tail), "owner_v"
    return None, "opaque"


def _format_next_version(base: str, owner: str, base_num: int, style: str) -> str:
    nxt = base_num + 1
    if style == "v_prefix":
        return f"v{nxt}"
    if style == "int":
        return str(nxt)
    if style == "owner_v":
        prefix = base.rsplit(":v", 1)[0]
        return f"{prefix}:v{nxt}"
    return f"{owner}:v{nxt}"


def _advance_published_version(
    input_version: str, owner: str, total_tokens: int, explicit: Any = None,
    *, per_request_versions: Sequence[tuple[str, int]] = (),
    selected: Sequence[tuple[str, int]] = (),
) -> str:
    """Enforce strict single-step state-version monotonicity (published == base + int(active > 0))."""
    base = str(input_version or "").strip()
    if total_tokens <= 0:
        if explicit is not None and str(explicit).strip() and str(explicit).strip() != base:
            raise ModelingError(
                f"recover_ssm_non_monotonic_state_version:{owner}:zero_active:"
                f"base={base or 'v0'}:published={str(explicit).strip()}"
            )
        return base
    if per_request_versions and len({v for _, v in per_request_versions}) > 1:
        adv_by_req = {str(r): int(c) for r, c in selected}
        parts = [
            f"{req}:v{int(ver) + (1 if adv_by_req.get(str(req), 0) > 0 else 0)}"
            for req, ver in per_request_versions
        ]
        expected_het = f"{owner}:per_request[{','.join(parts)}]"
        if explicit is not None and str(explicit).strip() and str(explicit).strip() != expected_het:
            raise ModelingError(
                f"recover_ssm_non_monotonic_state_version:{owner}:heterogeneous:"
                f"published={str(explicit).strip()}:expected={expected_het}"
            )
        return expected_het
    base_num, style = _parse_version_number(base)
    if base_num is None:
        base_num = 0
    expected = _format_next_version(base, owner, base_num, style)
    if explicit is not None and str(explicit).strip():
        exp_text = str(explicit).strip()
        exp_num, _ = _parse_version_number(exp_text)
        if exp_num is None or exp_num <= base_num or exp_num != base_num + 1:
            raise ModelingError(
                f"recover_ssm_non_monotonic_state_version:{owner}:base={base or 'v0'}:"
                f"published={exp_text}:expected={expected}"
            )
        return exp_text
    return expected


def _resolve_arch_family(
    runtime: Any, metadata: Mapping[str, Any], *, machine: Any = None,
) -> str:
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    for candidate in (
        metadata.get("arch_family"),
        rt_meta.get("arch_family"),
        getattr(runtime, "arch_family", None),
        getattr(machine, "arch_family", None),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    raise ModelingError("recover_ssm_arch_family_required")


def _available_aivs(machine: Any) -> int | None:
    if machine is None:
        return None
    dies = getattr(machine, "dies", None)
    if dies:
        first = dies[0]
        if hasattr(first, "owner_count"):
            cnt = int(first.owner_count("vector") or 0)
            if cnt > 0:
                return cnt
    raw = getattr(machine, "aiv_count", None)
    return int(raw) if raw is not None and int(raw) > 0 else None


_TRACK_TO_READINESS: Mapping[str, str] = {
    "shipping": "shipping",
    "proposed": "proposed_pr",
    "target": "synthesized_target",
}
_VALID_READINESS: frozenset[str] = frozenset({
    "shipping", "proposed_pr", "synthesized_target", "tiler_extension",
})


def _resolve_readiness(runtime: Any, *, family_kind: str, arch_family: str = "") -> str:
    """Resolve KernelBinding readiness strictly from explicit metadata, StackTrack, or ProbedAbiRegistry."""
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    if bool(rt_meta.get("raw_eager", False)):
        raise ModelingError("recover_ssm_incompatible_with_eager_track")
    spine = rt_meta.get("single_spine")
    if isinstance(spine, Mapping):
        track_spec = spine.get("track_spec")
        if isinstance(track_spec, Mapping):
            if str(track_spec.get("track", "")).strip().lower() == "eager":
                raise ModelingError("recover_ssm_incompatible_with_eager_track")
        if str(spine.get("stack_track") or "").strip().lower() == "eager":
            raise ModelingError("recover_ssm_incompatible_with_eager_track")
    explicit = str(rt_meta.get("recover_ssm_readiness") or "").strip()
    if explicit:
        if explicit not in _VALID_READINESS:
            raise ModelingError(f"recover_ssm_invalid_readiness:{explicit!r}")
        return explicit
    if isinstance(spine, Mapping):
        track_spec = spine.get("track_spec")
        if isinstance(track_spec, Mapping):
            target_mech = "recoverssm_gdn" if family_kind == "gdn" else "recoverssm_kda"
            for binding in track_spec.get("active_bindings") or ():
                if isinstance(binding, Mapping) and str(binding.get("mechanism_id", "")) == target_mech:
                    state = str(binding.get("readiness") or binding.get("state") or "").strip()
                    if state in _VALID_READINESS:
                        return state
            track = str(track_spec.get("track", "")).strip().lower()
            if track in _VALID_READINESS:
                return track
            if track in _TRACK_TO_READINESS:
                return _TRACK_TO_READINESS[track]
        stack_track = str(spine.get("stack_track") or "").strip().lower()
        if stack_track in _VALID_READINESS:
            return stack_track
        if stack_track in _TRACK_TO_READINESS:
            return _TRACK_TO_READINESS[stack_track]
    arch = str(arch_family or rt_meta.get("arch_family") or "").strip().upper()
    if arch in {"A2", "A3", "A5"}:
        from mindie_sim.backends.abi_registry import default_abi_registry

        kernel_id = GDN_COMMIT_KERNEL_ID if family_kind == "gdn" else KDA_COMMIT_KERNEL_ID
        if default_abi_registry().is_shipping_available(kernel_id, arch):
            return "shipping"
    raise ModelingError(f"recover_ssm_readiness_unresolved:{family_kind}")


def _commit_kernel_binding(
    *, family_kind: str, runtime: Any, arch_family: str,
) -> KernelBinding:
    kernel_id = GDN_COMMIT_KERNEL_ID if family_kind == "gdn" else KDA_COMMIT_KERNEL_ID
    return KernelBinding(
        kernel_id=kernel_id,
        backend=COMMIT_KERNEL_BACKEND,
        arch_family=arch_family,
        submodule="recurrent_state",
        macro_module="attention",
        readiness=_resolve_readiness(runtime, family_kind=family_kind, arch_family=arch_family),  # type: ignore[arg-type]
        recipe_id="recover_ssm_commit",
    )


def _commit_owner_schedule(
    active_count: int,
    heads: int,
    dk: int,
    dv: int,
    total_tokens: int,
    read_bytes: float,
    write_bytes: float,
    vector_ops: float,
    state_dtype: str,
    *,
    num_layers: int = 1,
    v_tiles: int = 1,
    head_groups: int | None = None,
    available_aivs: int | None = None,
) -> OwnerScheduleSpec:
    """Physical Triton 1D flattened grid distributed across physical rank AIVs."""
    eff_head_groups = max(1, int(head_groups)) if head_groups is not None else max(1, int(heads))
    work_tiles = active_count * max(1, int(num_layers)) * eff_head_groups * max(1, int(v_tiles))
    has_cap = available_aivs is not None and int(available_aivs) > 0
    cap_aivs = int(available_aivs) if has_cap else 0
    owners = min(work_tiles, cap_aivs) if has_cap else work_tiles
    total_owners = cap_aivs if has_cap else max(1, owners)
    max_tiles = math.ceil(work_tiles / owners) if owners > 0 else 0
    per_head_state = int(dk * dv * dtype_bytes(state_dtype))
    ub_live = max(FP32_RECORD_BYTES, per_head_state // max(1, int(v_tiles)))
    denom = float(max(1, owners))
    phases: tuple[SerialPhaseSpec, ...] = ()
    if owners > 0:
        phases = (
            SerialPhaseSpec("state_record_load", "mte2", read_bytes / denom, "bytes"),
            SerialPhaseSpec("ssm_replay_step", "vec_alu", vector_ops / denom, "op",
                            dependent_chain_steps=max(1, total_tokens)),
            SerialPhaseSpec("state_publish", "mte3", write_bytes / denom, "bytes"),
        )
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="triton_1d_flattened_seq_layer_head_vtile",
        parallel_axis="active_seq_x_layer_x_head_x_vtile",
        work_tiles=work_tiles,
        launched_owners=owners,
        active_owners=owners,
        total_owners=total_owners,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=ub_live,
            ub_capacity_bytes=0,
            owner_mte2_bytes=(read_bytes / denom) if owners > 0 else 0.0,
            owner_mte3_bytes=(write_bytes / denom) if owners > 0 else 0.0,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="reuse",
        ),
        serial_phases=phases,
        dispatch_envelope_source="triton_1d_grid",
    )


def _advanced_tokens(view: Any) -> int:
    return commit_tokens_for_view(view)


def _bind_commit_native_call(candidate: OpSpec, *, kernel_id: str, plan_kernel: str) -> OpSpec:
    kernel_rows = (
        {"name": kernel_id, "role": "commit", "evidence": "source_declared"},
    ) + (({"name": plan_kernel, "role": "commit_plan", "evidence": "source_declared"},) if plan_kernel else ())
    declared = declare_native_call(
        candidate, api={"namespace": "triton", "entry": kernel_id},
        kernels=kernel_rows, execution_core="aiv",
    )
    bound_native = bind_native_call(declared)
    final_meta = dict(declared.metadata)
    if bound_native is not None:
        final_meta[NATIVE_CALL_METADATA_KEY] = bound_native.to_dict()
    return replace(declared, metadata=tuple(sorted(final_meta.items(), key=lambda pair: pair[0])))


def _lookup_block_table_row(
    raw_tables: Any, req_id: str, req_ordinal: int,
) -> Sequence[Any] | None:
    if isinstance(raw_tables, Mapping):
        row = raw_tables.get(req_id)
        return row if isinstance(row, (list, tuple)) else None
    if isinstance(raw_tables, (list, tuple)):
        if 0 <= req_ordinal < len(raw_tables):
            row = raw_tables[req_ordinal]
            if isinstance(row, (list, tuple)):
                return row
    return None


def _has_align_boundary(
    family_kind: str, *, b_idx: int, f_idx: int, j_i: int, u_i: int, null_id: int,
) -> bool:
    """KDA (`kda/recoverssm.py:406`) vs GDN (`gdn_recoverssm_commit.py:145-146`) boundary predicate."""
    if b_idx <= null_id or j_i <= 0:
        return False
    if family_kind == "kda":
        return True
    return not ((b_idx == f_idx) and (j_i == u_i))


def _evaluate_align_boundaries(
    active_selected: Sequence[tuple[str, int]],
    *,
    active_views: Sequence[Any] = (),
    metadata: Mapping[str, Any],
    runtime: Any,
    state_elements: int,
    state_bytes: float,
    align_mode: bool = False,
    family_kind: str = "gdn",
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], int, float, int]:
    """Evaluate exact upstream null-slot & align-boundary guards per family."""
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    raw_bounds = metadata.get("recover_ssm_boundaries")
    if raw_bounds is None:
        raw_bounds = rt_meta.get("recover_ssm_boundaries")
    raw_tables = metadata.get("mamba_block_table")
    if raw_tables is None:
        raw_tables = rt_meta.get("mamba_block_table")
    view_by_id = {str(getattr(v, "request_id", "")): v for v in active_views}
    ordinal_by_id = {str(getattr(v, "request_id", "")): idx for idx, v in enumerate(active_views)}
    block_size = int(
        metadata.get("mamba_block_size")
        or rt_meta.get("mamba_block_size")
        or getattr(runtime, "kv_block_size", 0)
        or 0
    )
    ordinary_selected: list[tuple[str, int]] = []
    boundary_selected: list[tuple[str, int]] = []
    boundary_ops = 0
    boundary_write = 0.0
    triggered_count = 0
    for idx, (req_id, u_i) in enumerate(active_selected):
        view = view_by_id.get(req_id)
        typed_bound = dict(getattr(view, "recover_ssm_boundary", ()) or ()) if view is not None else {}
        entry = typed_bound or (raw_bounds.get(req_id) if isinstance(raw_bounds, Mapping) else None)
        if isinstance(entry, Mapping):
            null_id = int(entry.get("null_block_id", DEFAULT_NULL_BLOCK_ID))
            src_raw = entry.get("source_state_idx", getattr(view, "state_slot", None))
            if src_raw is not None and int(src_raw) <= null_id:
                continue
            has_final_key = "final_state_idx" in entry and entry["final_state_idx"] is not None
            f_idx = int(entry["final_state_idx"]) if has_final_key else (null_id + 1)
            if has_final_key and f_idx <= null_id:
                continue
            b_idx = int(entry.get("boundary_state_idx", null_id))
            j_i = int(entry.get("boundary_tokens", 0) or 0)
            has_boundary = _has_align_boundary(
                family_kind, b_idx=b_idx, f_idx=f_idx, j_i=j_i, u_i=u_i, null_id=null_id,
            )
            if has_boundary:
                boundary_selected.append((req_id, u_i))
                boundary_ops += REVERSE_FMA_OPS_PER_STATE_ELEMENT * (j_i + 1) * state_elements
                boundary_write += float(state_bytes)
                triggered_count += 1
            else:
                ordinary_selected.append((req_id, u_i))
            continue
        null_id = int(rt_meta.get("null_block_id", ALIGN_DEFAULT_NULL_BLOCK_ID) or ALIGN_DEFAULT_NULL_BLOCK_ID)
        src_slot = getattr(view, "state_slot", None) if view is not None else None
        if src_slot is not None and int(src_slot) <= null_id:
            continue
        if align_mode and block_size > 0 and view is not None:
            num_computed = int(getattr(view, "context", 0) or 0)
            final_num = num_computed + int(u_i)
            req_ord = ordinal_by_id.get(req_id, idx)
            typed_tbl = tuple(getattr(view, "mamba_block_table", ()) or ())
            req_table = typed_tbl if typed_tbl else _lookup_block_table_row(raw_tables, req_id, req_ord)
            if req_table is not None:
                if not req_table:
                    continue
                final_col = min(max(final_num - 1, 0) // block_size, len(req_table) - 1)
                f_raw = req_table[final_col]
                f_idx = null_id if f_raw is None else int(f_raw)
                if f_idx <= null_id:
                    continue
                next_boundary = (num_computed // block_size + 1) * block_size
                crosses = final_num >= next_boundary
                boundary_col = next_boundary // block_size - 1
                j_i = next_boundary - num_computed
                if crosses and 0 <= boundary_col < len(req_table):
                    b_raw = req_table[boundary_col]
                    b_idx = null_id if b_raw is None else int(b_raw)
                else:
                    b_idx = null_id
                has_boundary = crosses and _has_align_boundary(
                    family_kind, b_idx=b_idx, f_idx=f_idx, j_i=j_i, u_i=u_i, null_id=null_id,
                )
                if has_boundary:
                    boundary_selected.append((req_id, u_i))
                    boundary_ops += REVERSE_FMA_OPS_PER_STATE_ELEMENT * (j_i + 1) * state_elements
                    boundary_write += float(state_bytes)
                    triggered_count += 1
                else:
                    ordinary_selected.append((req_id, u_i))
                continue
            next_boundary = (num_computed // block_size + 1) * block_size
            if final_num >= next_boundary:
                final_col = max(final_num - 1, 0) // block_size
                boundary_col = next_boundary // block_size - 1
                j_i = next_boundary - num_computed
                if (family_kind == "kda" or not (final_col == boundary_col and j_i == u_i)) and j_i > 0:
                    boundary_selected.append((req_id, u_i))
                    boundary_ops += REVERSE_FMA_OPS_PER_STATE_ELEMENT * (j_i + 1) * state_elements
                    boundary_write += float(state_bytes)
                    triggered_count += 1
                    continue
        ordinary_selected.append((req_id, u_i))
    return ordinary_selected, boundary_selected, boundary_ops, boundary_write, triggered_count


def _state_bytes(metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("state_bytes")
    if declared is not None:
        return float(declared)
    heads = int(metadata.get("heads", 1))
    dk = int(metadata.get("dk", 1))
    dv = int(metadata.get("dv", 1))
    dtype = str(metadata.get("state_dtype", "fp32"))
    return float(heads * dk * dv) * float(dtype_bytes(dtype))


def _decode_speculative(view: Any) -> bool:
    """A verify view is decode-only; a prefill multi-token query is not spec."""
    return (str(getattr(view, "phase", "")) == "decode"
            and int(getattr(view, "q_len", 0) or 0) > 1
            and not bool(getattr(view, "padding", False)))


def _owner(metadata: Mapping[str, Any], op: OpSpec) -> str:
    return str(metadata.get("state_owner") or
               metadata.get("recurrent_state_owner") or f"layer{op.layer}")


def _geometry(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("recover_ssm_geometry")
    return value if isinstance(value, Mapping) else {}


def _scope(metadata: Mapping[str, Any]) -> str:
    return str(_geometry(metadata).get("scope", metadata.get("scope", "")))


def _layer(metadata: Mapping[str, Any], op: OpSpec) -> int:
    declared = _geometry(metadata).get("layer", metadata.get("layer", op.layer))
    return int(declared if declared is not None else op.layer)


def is_recover_ssm_intent(op: OpSpec) -> bool:
    """Whether this operator is the deferred, uncharged commit intent."""
    metadata = dict(op.metadata)
    return bool(metadata.get("recover_ssm_intent")) or str(
        metadata.get("recover_ssm_stage", "")) == INTENT_STAGE


def _is_pending_commit(op: OpSpec) -> bool:
    metadata = dict(op.metadata)
    return bool(metadata.get("recover_ssm_pending_acceptance"))


def _resolve_v_tiles(
    metadata: Mapping[str, Any], geometry: Mapping[str, Any], *,
    family_kind: str, dk: int, dv: int, heads: int, arch_family: str,
    state_dtype: str = "fp32", align_mode: bool = False,
) -> tuple[int, int, int]:
    _, _, head_groups = resolve_commit_tile_geometry(
        family_kind=family_kind, dk=dk, dv=dv, heads=heads, arch_family=arch_family,
        state_dtype=state_dtype, align_mode=align_mode,
    )
    explicit = metadata.get("v_tiles", geometry.get("v_tiles"))
    if explicit is not None and int(explicit) > 0:
        tiles = int(explicit)
        return max(1, math.ceil(max(1, dv) / tiles)), tiles, head_groups
    v_block = metadata.get("gdn_v_tile_width", geometry.get("v_tile_width"))
    if v_block is not None and int(v_block) > 0:
        bv = int(v_block)
        return bv, max(1, math.ceil(max(1, dv) / bv)), head_groups
    if family_kind in {"gdn", "kda"}:
        return resolve_commit_tile_geometry(
            family_kind=family_kind, dk=dk, dv=dv, heads=heads, arch_family=arch_family,
            state_dtype=state_dtype, align_mode=align_mode,
        )
    return max(1, int(dv)), 1, max(1, int(heads))


def _request_state_lengths_and_transitions(
    active: Sequence[Any],
    sequence: Sequence[tuple[str, int]],
    *,
    owner: str,
    base_ver_num: int | None,
) -> tuple[tuple[int, ...], tuple[tuple[Any, ...], ...]]:
    """Build per-request query lengths and monotonic state-version transition records."""
    active_by_id = {str(getattr(view, "request_id", "")): view for view in active}
    req_lengths = tuple(
        max(1, int(getattr(active_by_id[req], "q_len", 1)))
        for req, _cnt in sequence if req in active_by_id
    )
    req_transitions = tuple(
        (
            req,
            getattr(active_by_id[req], "state_slot", None),
            int(getattr(active_by_id[req], "state_generation", 0) or 0),
            f"{owner}:v{int(getattr(active_by_id[req], 'state_version', base_ver_num or 0) or 0)}",
            f"{owner}:v{int(getattr(active_by_id[req], 'state_version', base_ver_num or 0) or 0) + (1 if cnt > 0 else 0)}",
            str(getattr(active_by_id[req], "previous_state_publication", "") or ""),
        )
        for req, cnt in sequence if req in active_by_id
    )
    return req_lengths, req_transitions


def _kda_split_work(
    *,
    ordinary_selected: Sequence[tuple[str, int]],
    boundary_selected: Sequence[tuple[str, int]],
    heads: int,
    dk: int,
    dv: int,
    state_dtype: str,
    kg_dtype: str,
    v_tiles_ord: int,
    v_tiles_bnd: int,
    boundary_ops: int,
    boundary_write: float,
    record_override: float | None,
    state_override: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute KDA ordinary (non-crossing) and boundary (crossing) work with distinct tiles."""
    ord_work = compute_recover_ssm_commit_work(
        family_kind="kda",
        accepted_per_request=tuple(cnt for _req, cnt in ordinary_selected),
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
        v_tiles=v_tiles_ord, boundary_ops=0, boundary_write_bytes=0.0,
        record_bytes_override=record_override, state_bytes_override=state_override,
    )
    bnd_work = compute_recover_ssm_commit_work(
        family_kind="kda",
        accepted_per_request=tuple(cnt for _req, cnt in boundary_selected),
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
        v_tiles=v_tiles_bnd, boundary_ops=boundary_ops, boundary_write_bytes=boundary_write,
        record_bytes_override=record_override, state_bytes_override=state_override,
    )
    return ord_work, bnd_work


def _build_shared_kda_plan_and_conv(
    *,
    stage: int,
    phase: str,
    requests: tuple[str, ...],
    plan_deps: tuple[str, ...],
    verify_deps: tuple[str, ...],
    num_layers: int,
    active_count: int,
    heads: int,
    dk: int,
    dv: int,
    align_mode: bool,
    arch_family: str,
    readiness: str,
    metadata: Mapping[str, Any],
    group_suffix: str = "",
) -> tuple[OpSpec, OpSpec]:
    """Emit stage-shared ``_prepare_commit_plan_kernel`` and ``_compact_conv_state_kernel``."""
    batch = max(1, len(requests))
    plan_read = float(
        batch * (KDA_PLAN_INPUT_ARRAYS_ALIGN if align_mode else KDA_PLAN_INPUT_ARRAYS_BASE) * FP32_RECORD_BYTES
    )
    plan_write = float(batch * KDA_PLAN_OUTPUT_ARRAYS * FP32_RECORD_BYTES)
    plan_kb = KernelBinding(
        kernel_id=KDA_COMMIT_PLAN_KERNEL_NAME, backend=COMMIT_KERNEL_BACKEND,
        arch_family=arch_family, submodule="recurrent_state", macro_module="attention",
        readiness=readiness, recipe_id="kda_prepare_commit_plan",  # type: ignore[arg-type]
    )
    plan_op = OpSpec(
        op_id=f"s{stage}.kda_prepare_commit_plan{group_suffix}", family="elementwise", phase=phase,
        stage=stage, layer=-1, requests=requests,
        semantic_shape=(batch,), local_shape=(batch,), padded_shape=(batch,), launch_shape=(batch,),
        flops=0.0, vector_ops=float(batch * KDA_PLAN_OPS_PER_REQ),
        hbm_bytes=plan_read + plan_write, mte2_bytes=plan_read, mte3_bytes=plan_write,
        deps=plan_deps, recipe="kda_prepare_commit_plan", dtype="int32",
        source_boundary="", kernel_binding=plan_kb,
        metadata=(
            ("align_mode", align_mode),
            ("native_kind", KDA_COMMIT_PLAN_KERNEL_NAME),
            ("num_layers", num_layers),
            ("recover_ssm_stage", "commit_plan"),
        ),
    )
    conv_dim = int(metadata.get("conv_dim") or (heads * (dk * 2 + dv)))
    conv_hist = int(metadata.get("conv_history_len") or KDA_CONV_DEFAULT_HISTORY_LEN)
    conv_elems = active_count * num_layers * conv_dim * conv_hist
    conv_bytes = float(conv_elems * KDA_CONV_ELEMENT_BYTES)
    conv_kb = KernelBinding(
        kernel_id=KDA_CONV_COMPACT_KERNEL_NAME, backend=COMMIT_KERNEL_BACKEND,
        arch_family=arch_family, submodule="recurrent_state", macro_module="attention",
        readiness=readiness, recipe_id="kda_compact_conv_state",  # type: ignore[arg-type]
    )
    conv_op = OpSpec(
        op_id=f"s{stage}.kda_compact_conv_state{group_suffix}", family="elementwise", phase=phase,
        stage=stage, layer=-1, requests=requests,
        semantic_shape=(active_count, num_layers, conv_dim, conv_hist),
        local_shape=(active_count, num_layers, conv_dim, conv_hist),
        padded_shape=(active_count, num_layers, conv_dim, conv_hist),
        launch_shape=(active_count, num_layers, conv_dim, conv_hist),
        flops=0.0, vector_ops=float(conv_elems),
        hbm_bytes=2.0 * conv_bytes, mte2_bytes=conv_bytes, mte3_bytes=conv_bytes,
        deps=tuple(dict.fromkeys((plan_op.op_id, *verify_deps))),
        recipe="kda_compact_conv_state", dtype="bf16",
        source_boundary="", kernel_binding=conv_kb,
        metadata=(
            ("align_mode", align_mode),
            ("conv_dim", conv_dim),
            ("conv_history_len", conv_hist),
            ("native_kind", KDA_CONV_COMPACT_KERNEL_NAME),
            ("num_layers", num_layers),
            ("recover_ssm_stage", "conv_compact"),
        ),
    )
    return plan_op, conv_op


def _build_shared_kda_physical_commit(
    *,
    op_id: str,
    commit_kind: str,
    stage: int,
    phase: str,
    requests: tuple[str, ...],
    deps: tuple[str, ...],
    selected_subset: Sequence[tuple[str, int]],
    active: Sequence[Any],
    work_per_layer: Mapping[str, Any],
    num_layers: int,
    covered_layers: tuple[int, ...],
    heads: int,
    dk: int,
    dv: int,
    state_dtype: str,
    kg_dtype: str,
    v_tile_width: int,
    v_tiles: int,
    head_groups: int,
    align_mode: bool,
    boundary_count: int,
    k_binding: KernelBinding,
    available_aivs: int | None,
    version_meta: Mapping[str, Any],
) -> OpSpec:
    """Build one cross-layer physical KDA commit kernel launch (ordinary or boundary)."""
    scale = float(max(1, num_layers))
    active_count = int(work_per_layer["active_sequences"])
    total_tokens = int(work_per_layer["total_tokens"])
    unique_rec = float(work_per_layer["unique_record_read_bytes"]) * scale
    phys_rec = float(work_per_layer["physical_tiled_record_read_bytes"]) * scale
    read = (float(work_per_layer["state_read_bytes"]) + float(work_per_layer["physical_tiled_record_read_bytes"])) * scale
    write = float(work_per_layer["state_write_bytes"]) * scale
    matrix_ops = int(work_per_layer["matrix_ops"]) * num_layers
    b_ops = int(work_per_layer["boundary_ops"]) * num_layers
    scalar_aux = int(work_per_layer["scalar_aux_ops"]) * num_layers
    vector = float(work_per_layer["vector_ops"]) * scale
    req_lengths = tuple(
        max(1, int(getattr(next((v for v in active if str(getattr(v, "request_id", "")) == r), None), "q_len", 1)))
        for r, _ in selected_subset
    )
    owner_schedule = _commit_owner_schedule(
        active_count, heads, dk, dv, total_tokens, read, write, vector,
        state_dtype, num_layers=num_layers, v_tiles=v_tiles,
        head_groups=head_groups, available_aivs=available_aivs,
    )
    meta: dict[str, Any] = {
        **version_meta,
        "recover_ssm_stage": "commit" if commit_kind == "ordinary" else "commit_boundary",
        "recover_ssm_commit_kind": commit_kind,
        "recover_ssm_pending_acceptance": False,
        "recover_ssm_intent": False,
        "recurrent_attention_kind": "kda",
        "recover_ssm_readiness": k_binding.readiness,
        "native_kind": k_binding.kernel_id,
        "verify_kernel_name": KDA_VERIFY_KERNEL_NAME,
        "commit_plan_kernel_name": KDA_COMMIT_PLAN_KERNEL_NAME,
        "launch_scope": "cross_layer_shared",
        "physical_launch_count": 1,
        "num_layers": num_layers,
        "covered_layers": covered_layers,
        "lengths": req_lengths or (max(1, total_tokens),),
        "sequence_lengths": req_lengths or (max(1, total_tokens),),
        "state_slots": active_count,
        "active_sequences": active_count,
        "accepted_per_request": tuple(cnt for _r, cnt in selected_subset),
        "query_tokens": sum(req_lengths),
        "committed_tokens": tuple(selected_subset),
        "record_bytes_per_token": float(work_per_layer["record_bytes_per_token"]),
        "unique_record_read_bytes": unique_rec,
        "physical_tiled_record_read_bytes": phys_rec,
        "gdn_v_tile_width": v_tile_width,
        "v_tiles": v_tiles,
        "align_mode": align_mode,
        "commit_matrix_ops": matrix_ops,
        "boundary_matrix_ops": b_ops,
        "scalar_aux_ops": scalar_aux,
        "boundary_write_bytes": float(work_per_layer["state_write_bytes"] - active_count * work_per_layer["state_bytes_per_seq"]) * scale,
        "align_boundary_triggered": boundary_count,
        "state_bytes": float(work_per_layer["state_bytes_per_seq"]),
        "heads": heads,
        "dk": dk,
        "dv": dv,
        "state_dtype": state_dtype,
        "record_kg_dtype": kg_dtype,
        "acceptance_boundary": ACCEPTANCE_BOUNDARY,
    }
    candidate = OpSpec(
        op_id=op_id, family="elementwise", phase=phase, stage=stage,
        layer=covered_layers[0] if len(covered_layers) == 1 else -1,
        requests=requests,
        semantic_shape=(max(1, active_count), num_layers, heads, dv),
        local_shape=(max(1, active_count), num_layers, heads, dv),
        padded_shape=(max(1, active_count), num_layers, heads, dv),
        launch_shape=(max(1, active_count), num_layers, heads, dv),
        flops=0.0, vector_ops=vector, hbm_bytes=read + write,
        mte2_bytes=read, mte3_bytes=write, deps=deps,
        recipe="recover_ssm_commit", dtype=state_dtype,
        source_boundary="kernel", kernel_binding=k_binding,
        owner_schedule=owner_schedule,
        metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])),
        notes=("recover_ssm_commit_after_acceptance", "kda_cross_layer_shared_launch"),
    )
    return _bind_commit_native_call(
        candidate, kernel_id=k_binding.kernel_id, plan_kernel=KDA_COMMIT_PLAN_KERNEL_NAME,
    )


def _collect_group_pub_records(
    group: Sequence[tuple[OpSpec, dict[str, Any], str, str, list[tuple[str, int]]]],
    *,
    active: Sequence[Any],
    events: Mapping[str, Any],
    bindings: Mapping[str, Any],
    valid_req_ids: frozenset[str],
) -> tuple[
    list[str],
    list[str],
    list[tuple[OpSpec, str, str, str, int | None, bool, tuple[tuple[Any, ...], ...], tuple[tuple[str, int], ...]]],
]:
    """Validate acceptance events and compute null-slot-filtered per-layer publish records."""
    plan_deps_list: list[str] = []
    verify_deps_list: list[str] = []
    records: list[tuple[OpSpec, str, str, str, int | None, bool, tuple[tuple[Any, ...], ...], tuple[tuple[str, int], ...]]] = []
    for op, meta, owner, version, selected in group:
        missing = [req for req, _ in selected if req not in events]
        if missing:
            raise ModelingError("recover_ssm_commit_requires_acceptance_event:" + ",".join(sorted(missing)))
        plan_deps_list.extend(str(events[req]) for req, _ in selected)
        binding = bindings.get(owner)
        if binding is not None:
            plan_deps_list.append(str(binding))
        for _req, pub in tuple(meta.get("previous_state_publications") or ()):
            if pub:
                plan_deps_list.append(f"state_input.{pub}")
        verify_deps_list.extend(op.deps)
        sequence = tuple((req, cnt if req in valid_req_ids else 0) for req, cnt in selected)
        total = sum(cnt for _r, cnt in sequence if cnt > 0)
        req_vers_meta = tuple(meta.get("request_state_versions") or ())
        is_het = bool(req_vers_meta and len({v for _, v in req_vers_meta}) > 1)
        pub_ver = _advance_published_version(
            version, owner, total, explicit=meta.get("published_state_version"),
            per_request_versions=req_vers_meta, selected=sequence,
        )
        base_ver_num, _ = _parse_version_number(version)
        _, req_transitions = _request_state_lengths_and_transitions(
            active, sequence, owner=owner, base_ver_num=base_ver_num,
        )
        records.append((op, owner, version, pub_ver, None if is_het else int(base_ver_num or 0), is_het, req_transitions, sequence))
    return plan_deps_list, verify_deps_list, records


def _emit_layer_publish_barriers(
    layer_pub_records: Sequence[tuple[OpSpec, str, str, str, int | None, bool, tuple[tuple[Any, ...], ...], tuple[tuple[str, int], ...]]],
    *,
    family_kind: str,
    phys_dep_tuple: tuple[str, ...],
) -> list[OpSpec]:
    """Emit zero-work per-layer logical state publication barriers after a shared commit."""
    out: list[OpSpec] = []
    for op, owner, in_v, pub_v, base_n, is_het, req_trans, seq in layer_pub_records:
        pub_meta = {
            "recover_ssm_stage": "state_publish",
            "recover_ssm_pending_acceptance": False,
            "recover_ssm_intent": False,
            "recurrent_attention_kind": family_kind,
            "state_owner": owner,
            "base_version": base_n,
            "input_state_version": in_v,
            "published_state_version": pub_v,
            "heterogeneous_state_versions": is_het,
            "request_state_transitions": req_trans,
            "committed_tokens": seq,
            "shared_commit_ops": phys_dep_tuple,
            "acceptance_boundary": ACCEPTANCE_BOUNDARY,
        }
        out.append(OpSpec(
            op_id=op.op_id, family="barrier", phase=op.phase,
            stage=op.stage, layer=op.layer, requests=op.requests,
            semantic_shape=(len(seq),), local_shape=(len(seq),),
            padded_shape=(len(seq),), launch_shape=(len(seq),),
            flops=0.0, vector_ops=0.0, hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0,
            deps=phys_dep_tuple, recipe="recover_ssm_state_publish", dtype="int32",
            source_boundary="",
            metadata=tuple(sorted(pub_meta.items(), key=lambda pair: pair[0])),
            notes=("recover_ssm_logical_layer_state_publish",),
        ))
    return out


def _materialize_shared_kda_group(
    group: Sequence[tuple[OpSpec, dict[str, Any], str, str, list[tuple[str, int]]]],
    *,
    active: Sequence[Any],
    events: Mapping[str, Any],
    bindings: Mapping[str, Any],
    runtime: Any,
    machine: Any,
    group_suffix: str = "",
) -> list[OpSpec]:
    """Upstream Option A (`recoverssm.py:1412-1568`): shared plan/conv/commit + per-layer publish barriers."""
    first_op, first_meta, first_owner, _first_ver, first_sel = group[0]
    stage, phase = int(first_op.stage), first_op.phase
    geometry = _geometry(first_meta)
    heads = int(geometry.get("heads", first_meta.get("heads", 1)))
    dk = int(geometry.get("dk", first_meta.get("dk", 1)))
    dv = int(geometry.get("dv", first_meta.get("dv", 1)))
    state_dtype = str(geometry.get("state_dtype", first_meta.get("state_dtype", "fp32")))
    kg_dtype = str(geometry.get("kg_dtype", first_meta.get("record_kg_dtype", "bf16")))
    arch_family = _resolve_arch_family(runtime, first_meta, machine=machine)
    align_mode = _is_align_mode_active(first_meta, runtime)
    available_aivs = _available_aivs(machine)
    readiness = _resolve_readiness(runtime, family_kind="kda")
    k_binding = _commit_kernel_binding(family_kind="kda", runtime=runtime, arch_family=arch_family)

    active_selected = [(r, c) for r, c in first_sel if c > 0]
    state = _state_bytes({**first_meta, "heads": heads, "dk": dk, "dv": dv, "state_dtype": state_dtype})
    state_elements = heads * dk * dv
    record_override_raw = first_meta.get("record_bytes_per_token")
    record_override = float(record_override_raw) if record_override_raw is not None else None
    state_override_raw = first_meta.get("state_bytes")
    state_override = float(state_override_raw) if state_override_raw is not None else None
    ord_sel, bnd_sel, bnd_ops, bnd_write, bnd_count = _evaluate_align_boundaries(
        active_selected, active_views=active, metadata=first_meta, runtime=runtime,
        state_elements=state_elements, state_bytes=state, align_mode=align_mode,
        family_kind="kda",
    )
    valid_ids = frozenset(r for r, _ in (*ord_sel, *bnd_sel))
    plan_deps_list, verify_deps_list, layer_pub_records = _collect_group_pub_records(
        group, active=active, events=events, bindings=bindings, valid_req_ids=valid_ids,
    )
    bv_ord, vt_ord, hg_ord = _resolve_v_tiles(
        first_meta, geometry, family_kind="kda", dk=dk, dv=dv, heads=heads,
        arch_family=arch_family, state_dtype=state_dtype, align_mode=False,
    )
    bv_bnd, vt_bnd, hg_bnd = _resolve_v_tiles(
        first_meta, geometry, family_kind="kda", dk=dk, dv=dv, heads=heads,
        arch_family=arch_family, state_dtype=state_dtype, align_mode=True,
    )
    ord_work, bnd_work = _kda_split_work(
        ordinary_selected=ord_sel, boundary_selected=bnd_sel,
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
        v_tiles_ord=vt_ord, v_tiles_bnd=vt_bnd, boundary_ops=bnd_ops,
        boundary_write=bnd_write, record_override=record_override, state_override=state_override,
    )
    num_layers = len(group)
    covered_layers = tuple(_layer(meta, op) for op, meta, *_ in group)
    reqs = tuple(dict.fromkeys(r for r, _ in first_sel))
    plan_deps = tuple(dict.fromkeys(plan_deps_list))
    verify_deps = tuple(dict.fromkeys(verify_deps_list))
    total_active = len(ord_sel) + len(bnd_sel)
    plan_op, conv_op = _build_shared_kda_plan_and_conv(
        stage=stage, phase=phase, requests=reqs, plan_deps=plan_deps,
        verify_deps=verify_deps, num_layers=num_layers, active_count=total_active,
        heads=heads, dk=dk, dv=dv, align_mode=align_mode, arch_family=arch_family,
        readiness=readiness, metadata=first_meta, group_suffix=group_suffix,
    )
    out_ops: list[OpSpec] = [plan_op, conv_op]
    commit_deps = tuple(dict.fromkeys((plan_op.op_id, conv_op.op_id, *verify_deps)))
    _, _, first_in_v, first_pub_v, first_base_n, first_het, first_trans, first_eff_seq = layer_pub_records[0]
    ver_meta = {
        "base_version": first_base_n,
        "input_state_version": first_in_v,
        "published_state_version": first_pub_v,
        "heterogeneous_state_versions": first_het,
        "request_state_transitions": first_trans,
        "state_owner": f"stage{stage}.kda{group_suffix}" if num_layers > 1 else first_owner,
    }
    physical_ids: list[str] = []
    # Upstream kda/recoverssm.py:1544-1569 launches ALIGN_MODE=False then ALIGN_MODE=True in align mode.
    if align_mode or ord_sel or not bnd_sel:
        ord_op = _build_shared_kda_physical_commit(
            op_id=f"s{stage}.recover_ssm_commit{group_suffix}", commit_kind="ordinary",
            stage=stage, phase=phase, requests=reqs, deps=commit_deps,
            selected_subset=ord_sel if ord_sel else list(first_eff_seq), active=active,
            work_per_layer=ord_work, num_layers=num_layers, covered_layers=covered_layers,
            heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
            v_tile_width=bv_ord, v_tiles=vt_ord, head_groups=hg_ord,
            align_mode=align_mode, boundary_count=0, k_binding=k_binding,
            available_aivs=available_aivs, version_meta=ver_meta,
        )
        out_ops.append(ord_op)
        physical_ids.append(ord_op.op_id)
    if align_mode and bnd_sel:
        bnd_op = _build_shared_kda_physical_commit(
            op_id=f"s{stage}.recover_ssm_commit_boundary{group_suffix}", commit_kind="boundary",
            stage=stage, phase=phase, requests=tuple(r for r, _ in bnd_sel),
            deps=commit_deps, selected_subset=bnd_sel, active=active,
            work_per_layer=bnd_work, num_layers=num_layers, covered_layers=covered_layers,
            heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
            v_tile_width=bv_bnd, v_tiles=vt_bnd, head_groups=hg_bnd,
            align_mode=True, boundary_count=bnd_count, k_binding=k_binding,
            available_aivs=available_aivs, version_meta=ver_meta,
        )
        out_ops.append(bnd_op)
        physical_ids.append(bnd_op.op_id)
    out_ops.extend(_emit_layer_publish_barriers(
        layer_pub_records, family_kind="kda", phys_dep_tuple=tuple(physical_ids),
    ))
    return out_ops


def _materialize_shared_gdn_group(
    group: Sequence[tuple[OpSpec, dict[str, Any], str, str, list[tuple[str, int]]]],
    *,
    active: Sequence[Any],
    events: Mapping[str, Any],
    bindings: Mapping[str, Any],
    runtime: Any,
    machine: Any,
    group_suffix: str = "",
) -> list[OpSpec]:
    """Upstream GDN (`gdn_recoverssm_commit.py:442`): 1 batched launch across num_layers + publish barriers."""
    first_op, first_meta, first_owner, _first_ver, first_sel = group[0]
    stage, phase = int(first_op.stage), first_op.phase
    geometry = _geometry(first_meta)
    heads = int(geometry.get("heads", first_meta.get("heads", 1)))
    dk = int(geometry.get("dk", first_meta.get("dk", 1)))
    dv = int(geometry.get("dv", first_meta.get("dv", 1)))
    state_dtype = str(geometry.get("state_dtype", first_meta.get("state_dtype", "fp32")))
    kg_dtype = str(geometry.get("kg_dtype", first_meta.get("record_kg_dtype", "bf16")))
    arch_family = _resolve_arch_family(runtime, first_meta, machine=machine)
    align_mode = _is_align_mode_active(first_meta, runtime)
    available_aivs = _available_aivs(machine)
    k_binding = _commit_kernel_binding(family_kind="gdn", runtime=runtime, arch_family=arch_family)

    active_selected = [(r, c) for r, c in first_sel if c > 0]
    state = _state_bytes({**first_meta, "heads": heads, "dk": dk, "dv": dv, "state_dtype": state_dtype})
    state_elements = heads * dk * dv
    record_override_raw = first_meta.get("record_bytes_per_token")
    record_override = float(record_override_raw) if record_override_raw is not None else None
    state_override_raw = first_meta.get("state_bytes")
    state_override = float(state_override_raw) if state_override_raw is not None else None
    ord_sel, bnd_sel, bnd_ops, bnd_write, bnd_count = _evaluate_align_boundaries(
        active_selected, active_views=active, metadata=first_meta, runtime=runtime,
        state_elements=state_elements, state_bytes=state, align_mode=align_mode,
        family_kind="gdn",
    )
    valid_ids = frozenset(r for r, _ in (*ord_sel, *bnd_sel))
    plan_deps_list, verify_deps_list, layer_pub_records = _collect_group_pub_records(
        group, active=active, events=events, bindings=bindings, valid_req_ids=valid_ids,
    )
    (
        v_tile_width, v_tiles, head_groups, active_count, record,
        unique_rec, phys_rec, read_l, write_l, matrix_l, scalar_l, vector_l,
    ) = _compute_single_layer_commit_work(
        family_kind="gdn", metadata=first_meta, geometry=geometry,
        ord_sel=ord_sel, bnd_sel=bnd_sel, heads=heads, dk=dk, dv=dv,
        state_dtype=state_dtype, kg_dtype=kg_dtype, arch_family=arch_family,
        align_mode=align_mode, boundary_ops=bnd_ops, boundary_write=bnd_write,
        record_override=record_override, state_override=state_override,
    )
    num_layers = len(group)
    scale = float(num_layers)
    covered_layers = tuple(_layer(meta, op) for op, meta, *_ in group)
    reqs = tuple(dict.fromkeys(r for r, _ in first_sel))
    commit_deps = tuple(dict.fromkeys((*plan_deps_list, *verify_deps_list)))
    _, _, first_in_v, first_pub_v, first_base_n, first_het, first_trans, first_eff_seq = layer_pub_records[0]
    req_lengths, _ = _request_state_lengths_and_transitions(active, first_eff_seq, owner=first_owner, base_ver_num=first_base_n)
    total_tokens = sum(c for _, c in first_eff_seq if c > 0)
    read, write, vector = read_l * scale, write_l * scale, vector_l * scale
    owner_schedule = _commit_owner_schedule(
        active_count, heads, dk, dv, total_tokens, read, write, vector, state_dtype,
        num_layers=num_layers, v_tiles=v_tiles, head_groups=head_groups,
        available_aivs=available_aivs,
    )
    meta: dict[str, Any] = {
        "recover_ssm_stage": "commit",
        "recover_ssm_pending_acceptance": False,
        "recover_ssm_intent": False,
        "recover_ssm_readiness": k_binding.readiness,
        "recurrent_attention_kind": "gdn",
        "native_kind": k_binding.kernel_id,
        "verify_kernel_name": GDN_VERIFY_KERNEL_NAME,
        "commit_plan_kernel_name": "",
        "launch_scope": "cross_layer_shared",
        "physical_launch_count": 1,
        "num_layers": num_layers,
        "covered_layers": covered_layers,
        "lengths": req_lengths or (max(1, total_tokens),),
        "sequence_lengths": req_lengths or (max(1, total_tokens),),
        "state_slots": active_count,
        "active_sequences": active_count,
        "accepted_per_request": tuple(cnt for _r, cnt in first_eff_seq),
        "query_tokens": sum(req_lengths),
        "committed_tokens": first_eff_seq,
        "record_bytes_per_token": record,
        "unique_record_read_bytes": unique_rec * scale,
        "physical_tiled_record_read_bytes": phys_rec * scale,
        "gdn_v_tile_width": v_tile_width,
        "v_tiles": v_tiles,
        "align_mode": align_mode,
        "commit_matrix_ops": matrix_l * num_layers,
        "boundary_matrix_ops": bnd_ops * num_layers,
        "scalar_aux_ops": scalar_l * num_layers,
        "boundary_write_bytes": bnd_write * scale,
        "align_boundary_triggered": bnd_count,
        "state_bytes": state,
        "heads": heads,
        "dk": dk,
        "dv": dv,
        "state_dtype": state_dtype,
        "record_kg_dtype": kg_dtype,
        "base_version": first_base_n,
        "heterogeneous_state_versions": first_het,
        "input_state_version": first_in_v,
        "published_state_version": first_pub_v,
        "request_state_transitions": first_trans,
        "state_owner": f"stage{stage}.gdn{group_suffix}" if num_layers > 1 else first_owner,
        "acceptance_boundary": ACCEPTANCE_BOUNDARY,
    }
    commit_op = _bind_commit_native_call(
        OpSpec(
            op_id=f"s{stage}.recover_ssm_commit{group_suffix}", family="elementwise",
            phase=phase, stage=stage,
            layer=covered_layers[0] if len(covered_layers) == 1 else -1,
            requests=reqs,
            semantic_shape=(max(1, active_count), num_layers, heads, dv),
            local_shape=(max(1, active_count), num_layers, heads, dv),
            padded_shape=(max(1, active_count), num_layers, heads, dv),
            launch_shape=(max(1, active_count), num_layers, heads, dv),
            flops=0.0, vector_ops=vector, hbm_bytes=read + write,
            mte2_bytes=read, mte3_bytes=write, deps=commit_deps,
            recipe="recover_ssm_commit", dtype=state_dtype,
            source_boundary="kernel", kernel_binding=k_binding,
            owner_schedule=owner_schedule,
            metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])),
            notes=("recover_ssm_commit_after_acceptance", "gdn_cross_layer_shared_launch"),
        ),
        kernel_id=k_binding.kernel_id, plan_kernel="",
    )
    return [commit_op, *_emit_layer_publish_barriers(
        layer_pub_records, family_kind="gdn", phys_dep_tuple=(commit_op.op_id,),
    )]


def _compute_single_layer_commit_work(
    *,
    family_kind: str,
    metadata: Mapping[str, Any],
    geometry: Mapping[str, Any],
    ord_sel: Sequence[tuple[str, int]],
    bnd_sel: Sequence[tuple[str, int]],
    heads: int,
    dk: int,
    dv: int,
    state_dtype: str,
    kg_dtype: str,
    arch_family: str,
    align_mode: bool,
    boundary_ops: int,
    boundary_write: float,
    record_override: float | None,
    state_override: float | None,
) -> tuple[int, int, int, int, float, float, float, float, float, int, int, float]:
    """Compute tile geometry and exact work totals for a single-layer commit op."""
    if family_kind == "kda":
        bv_ord, vt_ord, head_groups = _resolve_v_tiles(
            metadata, geometry, family_kind="kda", dk=dk, dv=dv, heads=heads,
            arch_family=arch_family, state_dtype=state_dtype, align_mode=False,
        )
        bv_bnd, vt_bnd, _ = _resolve_v_tiles(
            metadata, geometry, family_kind="kda", dk=dk, dv=dv, heads=heads,
            arch_family=arch_family, state_dtype=state_dtype, align_mode=True,
        )
        ord_w, bnd_w = _kda_split_work(
            ordinary_selected=ord_sel, boundary_selected=bnd_sel,
            heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
            v_tiles_ord=vt_ord, v_tiles_bnd=vt_bnd, boundary_ops=boundary_ops,
            boundary_write=boundary_write, record_override=record_override,
            state_override=state_override,
        )
        v_tile_width = bv_bnd if (bnd_sel and not ord_sel) else bv_ord
        v_tiles = vt_bnd if (bnd_sel and not ord_sel) else vt_ord
        active_count = int(ord_w["active_sequences"]) + int(bnd_w["active_sequences"])
        record = float(ord_w["record_bytes_per_token"])
        unique_rec_read = float(ord_w["unique_record_read_bytes"]) + float(bnd_w["unique_record_read_bytes"])
        phys_rec_read = float(ord_w["physical_tiled_record_read_bytes"]) + float(bnd_w["physical_tiled_record_read_bytes"])
        read = float(ord_w["state_read_bytes"]) + float(bnd_w["state_read_bytes"]) + phys_rec_read
        write = float(ord_w["state_write_bytes"]) + float(bnd_w["state_write_bytes"])
        matrix_ops = int(ord_w["matrix_ops"]) + int(bnd_w["matrix_ops"])
        scalar_aux_ops = int(ord_w["scalar_aux_ops"]) + int(bnd_w["scalar_aux_ops"])
        vector = float(ord_w["vector_ops"]) + float(bnd_w["vector_ops"])
        return (
            v_tile_width, v_tiles, head_groups, active_count, record,
            unique_rec_read, phys_rec_read, read, write, matrix_ops, scalar_aux_ops, vector,
        )
    v_tile_width, v_tiles, head_groups = _resolve_v_tiles(
        metadata, geometry, family_kind=family_kind, dk=dk, dv=dv, heads=heads,
        arch_family=arch_family, state_dtype=state_dtype, align_mode=align_mode,
    )
    eff_selected = list(ord_sel) + list(bnd_sel)
    work = compute_recover_ssm_commit_work(
        family_kind=family_kind,
        accepted_per_request=tuple(count for _req, count in eff_selected),
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, kg_dtype=kg_dtype,
        v_tiles=v_tiles, boundary_ops=boundary_ops, boundary_write_bytes=boundary_write,
        record_bytes_override=record_override, state_bytes_override=state_override,
    )
    return (
        v_tile_width, v_tiles, head_groups, int(work["active_sequences"]),
        float(work["record_bytes_per_token"]), float(work["unique_record_read_bytes"]),
        float(work["physical_tiled_record_read_bytes"]),
        float(work["state_read_bytes"]) + float(work["physical_tiled_record_read_bytes"]),
        float(work["state_write_bytes"]), int(work["matrix_ops"]),
        int(work["scalar_aux_ops"]), float(work["vector_ops"]),
    )


def _materialize_commit_op(
    op: OpSpec,
    *,
    metadata: dict[str, Any],
    owner: str,
    version: str,
    selected: list[tuple[str, int]],
    active: Sequence[Any],
    events: Mapping[str, Any],
    bindings: Mapping[str, Any],
    runtime: Any,
    machine: Any = None,
) -> OpSpec:
    """Materialize one acceptance-bound RecoverSSM commit OpSpec with Triton/native bindings."""
    dependencies = list(op.deps)
    missing = [request for request, _count in selected if request not in events]
    if missing:
        raise ModelingError(
            "recover_ssm_commit_requires_acceptance_event:" + ",".join(sorted(missing))
        )
    dependencies.extend(str(events[request]) for request, _ in selected)
    binding = bindings.get(owner)
    if binding is not None:
        dependencies.append(str(binding))
    active_selected = [(request, count) for request, count in selected if count > 0]
    total = sum(count for _request, count in active_selected)
    geometry = _geometry(metadata)
    if geometry:
        metadata["heads"] = int(geometry.get("heads", 1))
        metadata["dk"] = int(geometry.get("dk", 1))
        metadata["dv"] = int(geometry.get("dv", 1))
        metadata["state_dtype"] = str(geometry.get("state_dtype", "fp32"))
        metadata["record_kg_dtype"] = str(geometry.get("kg_dtype", "bf16"))
    family_kind = str(
        metadata.get("recurrent_attention_kind") or geometry.get("attention_kind") or "kda"
    ).strip().lower()
    heads = int(metadata.get("heads", 1))
    dk = int(metadata.get("dk", 1))
    dv = int(metadata.get("dv", 1))
    state_dtype = str(metadata.get("state_dtype", "fp32"))
    kg_dtype = str(metadata.get("record_kg_dtype", "bf16"))
    arch_family = _resolve_arch_family(runtime, metadata, machine=machine)
    align_mode = _is_align_mode_active(metadata, runtime)
    record_override_raw = metadata.get("record_bytes_per_token")
    record_override = float(record_override_raw) if record_override_raw is not None else None
    state_override_raw = metadata.get("state_bytes")
    state_override = float(state_override_raw) if state_override_raw is not None else None
    state = _state_bytes(metadata)
    state_elements = heads * dk * dv
    ord_sel, bnd_sel, boundary_ops, boundary_write, boundary_count = _evaluate_align_boundaries(
        active_selected, active_views=active, metadata=metadata, runtime=runtime,
        state_elements=state_elements, state_bytes=state, align_mode=align_mode,
        family_kind=family_kind,
    )
    valid_ids = frozenset(r for r, _ in (*ord_sel, *bnd_sel))
    (
        v_tile_width, v_tiles, head_groups, active_count, record,
        unique_rec_read, phys_rec_read, read, write, matrix_ops, scalar_aux_ops, vector,
    ) = _compute_single_layer_commit_work(
        family_kind=family_kind, metadata=metadata, geometry=geometry,
        ord_sel=ord_sel, bnd_sel=bnd_sel, heads=heads, dk=dk, dv=dv,
        state_dtype=state_dtype, kg_dtype=kg_dtype, arch_family=arch_family,
        align_mode=align_mode, boundary_ops=boundary_ops, boundary_write=boundary_write,
        record_override=record_override, state_override=state_override,
    )
    sequence = tuple((request, count if request in valid_ids else 0) for request, count in selected)
    eff_total = sum(count for _request, count in sequence if count > 0)
    req_vers_meta = tuple(metadata.get("request_state_versions") or ())
    is_het = bool(req_vers_meta and len({v for _, v in req_vers_meta}) > 1)
    published_ver = _advance_published_version(
        version, owner, eff_total, explicit=metadata.get("published_state_version"),
        per_request_versions=req_vers_meta, selected=sequence,
    )
    prev_pubs = tuple(metadata.get("previous_state_publications") or ())
    for _req, pub in prev_pubs:
        if pub:
            dependencies.append(f"state_input.{pub}")
    base_ver_num, _ = _parse_version_number(version)
    k_binding = _commit_kernel_binding(family_kind=family_kind, runtime=runtime, arch_family=arch_family)
    verify_kernel = GDN_VERIFY_KERNEL_NAME if family_kind == "gdn" else KDA_VERIFY_KERNEL_NAME
    plan_kernel = "" if family_kind == "gdn" else KDA_COMMIT_PLAN_KERNEL_NAME
    req_lengths, req_transitions = _request_state_lengths_and_transitions(
        active, sequence, owner=owner, base_ver_num=base_ver_num,
    )
    metadata.update({
        "recover_ssm_stage": "commit",
        "recover_ssm_pending_acceptance": False,
        "recover_ssm_intent": False,
        "recover_ssm_readiness": k_binding.readiness,
        "recurrent_attention_kind": family_kind,
        "native_kind": k_binding.kernel_id,
        "verify_kernel_name": verify_kernel,
        "commit_plan_kernel_name": plan_kernel,
        "lengths": req_lengths or (max(1, eff_total),),
        "sequence_lengths": req_lengths or (max(1, eff_total),),
        "state_slots": active_count,
        "active_sequences": active_count,
        "accepted_per_request": tuple(count for _request, count in sequence),
        "query_tokens": sum(req_lengths),
        "committed_tokens": sequence,
        "record_bytes_per_token": record,
        "unique_record_read_bytes": unique_rec_read,
        "physical_tiled_record_read_bytes": phys_rec_read,
        "gdn_v_tile_width": v_tile_width,
        "v_tiles": v_tiles,
        "align_mode": align_mode,
        "commit_matrix_ops": matrix_ops,
        "boundary_matrix_ops": boundary_ops,
        "scalar_aux_ops": scalar_aux_ops,
        "boundary_write_bytes": boundary_write,
        "align_boundary_triggered": boundary_count,
        "state_bytes": state,
        "heads": heads,
        "dk": dk,
        "dv": dv,
        "base_version": None if is_het else int(base_ver_num or 0),
        "heterogeneous_state_versions": is_het,
        "input_state_version": version,
        "published_state_version": published_ver,
        "request_state_transitions": req_transitions,
        "acceptance_boundary": ACCEPTANCE_BOUNDARY,
        "state_owner": owner,
    })
    owner_schedule = _commit_owner_schedule(
        active_count, heads, dk, dv, total, read, write, vector,
        state_dtype, v_tiles=v_tiles, head_groups=head_groups,
        available_aivs=_available_aivs(machine),
    )
    candidate = replace(
        op, deps=tuple(dict.fromkeys(dependencies)), recipe="recover_ssm_commit",
        source_boundary="kernel", vector_ops=vector, hbm_bytes=read + write,
        mte2_bytes=read, mte3_bytes=write, kernel_binding=k_binding,
        owner_schedule=owner_schedule,
        metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[0])),
        notes=tuple(op.notes) + ("recover_ssm_commit_after_acceptance",),
    )
    return _bind_commit_native_call(candidate, kernel_id=k_binding.kernel_id, plan_kernel=plan_kernel)


def bind_recover_ssm_commits(
    ops: Sequence[OpSpec], *, views: Sequence[Any],
    acceptance_events: Mapping[str, Any] | None,
    state_bindings: Mapping[str, Any] | None, runtime: Any,
    machine: Any = None,
    shared_kda_launch: bool | None = None,
) -> tuple[OpSpec, ...]:
    """Materialize acceptance-bound commits for every declared state owner."""
    active = tuple(view for view in views if _decode_speculative(view))
    if not active:
        return tuple(ops)
    events = dict(acceptance_events or {})
    bindings = dict(state_bindings or {})
    committed_by_request = {
        str(getattr(view, "request_id", "")): commit_tokens_for_view(view, runtime)
        for view in active
    }
    active_by_id = {str(getattr(v, "request_id", "")): v for v in active}
    emitted: set[tuple[str, str, int, str]] = set()
    prepared: list[tuple[int, OpSpec, dict[str, Any] | None, str, str, list[tuple[str, int]]]] = []
    kda_groups: dict[tuple[int, str], list[int]] = {}
    for idx, op in enumerate(ops):
        if not (is_recover_ssm_intent(op) or _is_pending_commit(op)):
            prepared.append((idx, op, None, "", "", []))
            continue
        metadata = dict(op.metadata)
        owner = _owner(metadata, op)
        request_ids = tuple(str(item) for item in op.requests)
        selected = [
            (request, committed_by_request[request]) for request in request_ids
            if request in committed_by_request
        ]
        if not selected:
            prepared.append((idx, op, None, "", "", []))
            continue
        version = str(metadata.get("input_state_version", metadata.get("state_version", "")) or "").strip()
        if not version:
            base_v = max(
                (int(getattr(active_by_id[req], "state_version", 0) or 0) for req, _ in selected if req in active_by_id),
                default=0,
            )
            version = f"{owner}:v{base_v}"
        key = (owner, _scope(metadata), _layer(metadata, op), version)
        if key in emitted:
            raise ModelingError(f"recover_ssm_duplicate_commit:{owner}:{version}")
        emitted.add(key)
        prepared.append((idx, op, metadata, owner, version, selected))
        geometry = _geometry(metadata)
        fam = str(metadata.get("recurrent_attention_kind") or geometry.get("attention_kind") or "kda").strip().lower()
        if fam in {"kda", "gdn"}:
            g_heads = int(geometry.get("heads", metadata.get("heads", 1)))
            g_dk = int(geometry.get("dk", metadata.get("dk", 1)))
            g_dv = int(geometry.get("dv", metadata.get("dv", 1)))
            g_sdtype = str(geometry.get("state_dtype", metadata.get("state_dtype", "fp32")))
            g_kg = str(geometry.get("kg_dtype", metadata.get("record_kg_dtype", "bf16")))
            kda_groups.setdefault(
                (int(op.stage), _scope(metadata), fam, g_heads, g_dk, g_dv, g_sdtype, g_kg), []
            ).append(idx)

    stage_group_counts: dict[tuple[int, str], int] = {}
    for gk in kda_groups:
        stage_group_counts[(gk[0], gk[1])] = stage_group_counts.get((gk[0], gk[1]), 0) + 1
    shared_indices: dict[int, tuple[Any, ...]] = {}
    for grp_key, idxs in kda_groups.items():
        grp_fam = grp_key[2]
        use_shared = (
            (shared_kda_launch is True and grp_fam == "kda")
            or (shared_kda_launch is not False and len(idxs) > 1)
        )
        if use_shared:
            for i in idxs:
                shared_indices[i] = grp_key

    result: list[OpSpec] = []
    emitted_groups: set[tuple[Any, ...]] = set()
    stage_group_ordinal: dict[tuple[int, str], int] = {}
    for idx, op, metadata, owner, version, selected in prepared:
        if metadata is None:
            result.append(op)
            continue
        if idx in shared_indices:
            grp_key = shared_indices[idx]
            if grp_key in emitted_groups:
                continue
            emitted_groups.add(grp_key)
            ss_key = (grp_key[0], grp_key[1])
            ord_idx = stage_group_ordinal.get(ss_key, 0)
            stage_group_ordinal[ss_key] = ord_idx + 1
            suffix = f".g{ord_idx}" if stage_group_counts.get(ss_key, 1) > 1 else ""
            group_entries = [
                (prepared[i][1], prepared[i][2], prepared[i][3], prepared[i][4], prepared[i][5])
                for i in kda_groups[grp_key]
            ]
            if grp_key[2] == "gdn":
                result.extend(_materialize_shared_gdn_group(
                    group_entries, active=active, events=events,
                    bindings=bindings, runtime=runtime, machine=machine,
                    group_suffix=suffix,
                ))
            else:
                result.extend(_materialize_shared_kda_group(
                    group_entries, active=active, events=events,
                    bindings=bindings, runtime=runtime, machine=machine,
                    group_suffix=suffix,
                ))
            continue
        result.append(_materialize_commit_op(
            op, metadata=metadata, owner=owner, version=version,
            selected=selected, active=active, events=events,
            bindings=bindings, runtime=runtime, machine=machine,
        ))
    return tuple(result)


def bind_declared_body_commits(
    ops: Sequence[OpSpec], *, views: Sequence[Any],
) -> tuple[OpSpec, ...]:
    """Body-only assembly: require the declared integer advancement explicitly."""
    active = tuple(view for view in views if _decode_speculative(view))
    for view in active:
        _advanced_tokens(view)
    rewritten = []
    for op in ops:
        if is_recover_ssm_intent(op) or _is_pending_commit(op):
            metadata = dict(op.metadata)
            metadata["acceptance_boundary"] = BODY_BOUNDARY
            rewritten.append(replace(
                op, metadata=tuple(sorted(metadata.items(),
                                          key=lambda pair: pair[0]))))
        else:
            rewritten.append(op)
    return tuple(rewritten)
