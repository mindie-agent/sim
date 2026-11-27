"""Two-Layer Operator SDK and Simulation-vs-Modeling honesty boundary.

[Role] Expose the canonical two-layer operator workflow and enforce the strict
  separation between Layer-2 Simulation Mode (``allow_unbound_formula=False``)
  and Layer-1 Formula-Driven Hardware Modeling Mode (``allow_unbound_formula=True``):
  - Layer 1 (Modeling Mode ONLY): ``FormulaTensorSpec``, ``FormulaKernelSpec``,
    ``FormulaKernelEvaluation``, ``evaluate_formula_kernel`` /
    ``simulate_formula_kernel``, and ``model_target_ceiling``;
  - Layer 2 (Simulation Mode Binding): ``bind_kernel_implementation`` promoting
    a formula specification or custom operator into a real ``Extension`` /
    ``Provider`` bound to a concrete implementation, recipe, or ``aclnn`` call.
[Invariants]
  * Simulation Mode (``simulate_operator``, ``simulate_native_call``,
    ``simulate_aclnn_call``, ``simulate_graph``, ``simulate_model``,
    ``simulate_service``) hard-locks ``allow_unbound_formula=False`` and fails
    fast with ``CapabilityGap(code="unbound_operator_implementation")`` whenever
    an unbound ``FormulaKernelSpec`` or unbound formula ``OpSpec`` is passed
    without a bound Layer-2 provider.
  * Layer-1 formula projections ALWAYS stamp ``binding_state="unbound_formula_projection"``
    and ``has_unbound_formula_ops=True``, and compute latencies purely from
    hardware ``datapath_for_die`` SRAM/pipe limits (zero heuristic multipliers).
  * ``bind_kernel_implementation`` rejects empty/placeholder bindings so Layer 2
    cannot be bypassed without an executable implementation, recipe, or ABI call.
[Timeouts & Locks] Pure bounded in-memory arithmetic; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Pure functional builders and evaluations; zero global
  registry mutation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import math
import re
from typing import Any

from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.hardware import Device, Hardware, parse_hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.contracts import (
    FormulaKernelEvaluation,
    FormulaKernelSpec,
    FormulaTensorSpec,
)
from mindie_sim.templates.fuse_region import (
    evaluate_formula_kernel as _canonical_evaluate_formula_kernel,
)

from .contracts import Extension, Provider
from .environment import ExecutionEnvironment
from .errors import CapabilityGap, ExtensionError

BINDING_STATE_BOUND = "bound_implementation"
BINDING_STATE_UNBOUND_FORMULA = "unbound_formula_projection"
MODE_MODELING = "modeling"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FULL_HARDWARE_KEYS = ("device", "devices", "dies", "links")
_MATRIX_SHAPE_DIMS = 3
_GENERIC_MATRIX_PROGRAM = "matmul"
_GENERIC_GROUPED_RECIPES = frozenset({"balanced_grouped_gate_up", "balanced_grouped_down"})
_MATRIX_SHAPE_OVERRIDE_KEYS = frozenset({"m", "n", "k", "x_shape", "weight_shape", "tensors"})
_VECTOR_SHAPE_OVERRIDE_KEYS = frozenset({"rows", "columns", "cols", "hidden", "width", "elements", "x_shape", "shape", "tensors"})
_KERNEL_IDENTITY_META_KEYS = (
    "bound_kernel_name", "formula_kernel", "kernel_name", "kernel",
    "api", "aclnn_symbol", "aclnn_api",
)
_BUILDER_PROTOCOL_KEYS = frozenset({
    "include_submission", "input_ready_ms", "invocations", "max_operators",
    "mode", "recipes", "replica", "submission", "trace_limit",
    "profile", "service_id", "external_inputs", "execution", "extensions",
})
_ATTN_GEOMETRY_ARG_KEYS = (
    "batch", "heads", "kv_heads", "query_tokens", "key_tokens",
    "head_dim", "v_head_dim", "query_dtype", "key_dtype",
    "value_dtype", "output_dtype", "block_size",
)
_ATTN_GEOMETRY_ATTR_KEYS = ("causal", "sparse_mode", "input_layout")


def _resolve_hardware_target(
    hardware: Any,
) -> Case | Hardware | Device | DatapathCapabilitySpec:
    """Resolve ``ExecutionEnvironment`` / SKU string / mapping into canonical hardware input."""
    if isinstance(hardware, (Case, Hardware, Device, DatapathCapabilitySpec)):
        return hardware
    if isinstance(hardware, ExecutionEnvironment):
        return hardware.machine
    if isinstance(hardware, Mapping):
        keys = hardware.keys()
        if {"machine", "model", "workload"} <= keys or {"hardware", "model"} <= keys:
            from mindie_sim.prediction.structure import parse_case
            return parse_case(dict(hardware))
        hardware = hardware.get("hardware", hardware.get("machine", hardware))
        if isinstance(hardware, Mapping) and any(k in hardware for k in _FULL_HARDWARE_KEYS):
            return parse_hardware(dict(hardware))
    if isinstance(hardware, (str, Mapping)):
        # SKU strings / shorthand mappings reuse compose's single machine materializer.
        from mindie_sim.compose import _COMPOSE_OPTION_GROUPS, _materialize_machine
        options = dict.fromkeys(_COMPOSE_OPTION_GROUPS["machine_options"])
        return parse_hardware(_materialize_machine(hardware, required_devices=1, **options)[2])
    raise ExtensionError("formula_kernel_hardware_invalid:" + type(hardware).__name__)


def evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Evaluate a Layer-1 ``FormulaKernelSpec`` directly against hardware SRAM & pipes."""
    if not isinstance(spec, FormulaKernelSpec):
        raise ExtensionError("formula_kernel_requires_FormulaKernelSpec")
    target = _resolve_hardware_target(hardware)
    return _canonical_evaluate_formula_kernel(
        spec, target, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def simulate_formula_kernel(
    spec_or_hardware: Any,
    hardware_or_spec: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Layer-1 Modeling Mode entrypoint accepting ``(spec, hardware)`` or ``(hardware, spec)``."""
    if isinstance(spec_or_hardware, FormulaKernelSpec):
        spec, hardware = spec_or_hardware, hardware_or_spec
    elif isinstance(hardware_or_spec, FormulaKernelSpec):
        hardware, spec = spec_or_hardware, hardware_or_spec
    else:
        raise ExtensionError("simulate_formula_kernel_requires_FormulaKernelSpec")
    return evaluate_formula_kernel(
        spec, hardware, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def _is_rank_mapping(ops: Any) -> bool:
    return (
        isinstance(ops, Mapping)
        and bool(ops)
        and all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ops)
    )


def _iter_ops_items(ops: Any) -> tuple[Any, ...]:
    if ops is None:
        return ()
    if _is_rank_mapping(ops):
        # Rank-indexed graph sets (``{rank: ops}``) are inspected op by op.
        return tuple(op for group in ops.values() for op in _iter_ops_items(group))
    if isinstance(ops, (FormulaKernelSpec, OpSpec, Mapping)):
        return (ops,)
    if isinstance(ops, Sequence) and not isinstance(ops, (str, bytes, bytearray)):
        return tuple(ops)
    return (ops,)


def _is_unbound_op_item(item: Any) -> str | None:
    """Return offending kernel/op name if ``item`` represents an unbound Layer-1 formula op."""
    if isinstance(item, FormulaKernelSpec):
        return item.name or "formula_kernel"
    if isinstance(item, OpSpec):
        meta = dict(item.metadata)
        if (
            str(meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(meta.get("unbound_formula_op"))
        ):
            return item.op_id or item.family
        return None
    if isinstance(item, Mapping):
        meta = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        if (
            str(item.get("binding_state") or meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(item.get("unbound_formula_op") or meta.get("unbound_formula_op"))
        ):
            return str(item.get("op_id") or item.get("name") or "unbound_op")
    return None


def _recipe_is_concrete(rec: Any) -> bool:
    return (
        rec is not None
        and getattr(rec, "binding_state", BINDING_STATE_BOUND) != BINDING_STATE_UNBOUND_FORMULA
    )


def _book_selects_explicit_recipe(op: OpSpec, book: Any) -> bool:
    """True iff ``RecipeBook.select(op)`` chooses an explicit concrete caller binding."""
    caller_recipes = frozenset(getattr(book, "_caller_recipes", ()))
    if op.recipe and op.recipe in caller_recipes:
        return _recipe_is_concrete(book.recipes.get(op.recipe))
    meta = dict(op.metadata)
    prog_kind = meta.get("program_kind")
    caller_programs = frozenset(getattr(book, "_caller_programs", ()))
    if (
        isinstance(prog_kind, str)
        and prog_kind in caller_programs
        and (prog_kind == op.family or prog_kind != _GENERIC_MATRIX_PROGRAM)
    ):
        return _recipe_is_concrete(book.programs.get(prog_kind))
    explicit_families = frozenset(getattr(book, "_explicit_families", ()))
    if op.family and op.family in explicit_families:
        if op.family != "grouped_gemm" or op.recipe in _GENERIC_GROUPED_RECIPES:
            return _recipe_is_concrete(book.families.get(op.family))
    return False


def _item_has_verified_binding(item: Any, recipes: Any = None, native_bindings: Any = None) -> bool:
    """Return True only when ``item`` carries a verified concrete binding or recipe."""
    if isinstance(item, FormulaKernelSpec):
        return False
    op_id = item.op_id if isinstance(item, OpSpec) else (
        str(item.get("op_id") or item.get("name") or "") if isinstance(item, Mapping) else ""
    )
    if op_id and isinstance(native_bindings, Mapping) and op_id in native_bindings:
        binding = native_bindings[op_id]
        return binding is not None and bool(getattr(binding, "bound", True))
    if op_id and isinstance(recipes, Mapping) and op_id in recipes:
        if _recipe_is_concrete(recipes[op_id]):
            return True
    if isinstance(item, OpSpec):
        if recipes is not None and not isinstance(recipes, Mapping):
            if _book_selects_explicit_recipe(item, recipes):
                return True
        meta = dict(item.metadata)
        if str(meta.get("binding_state") or "") == BINDING_STATE_BOUND and not bool(meta.get("unbound_formula_op")):
            return True
    return False


def enforce_simulation_binding_guard(
    ops: Any,
    *,
    capability: str,
    entry: str,
    allow_unbound_formula: bool = False,
    has_bound_provider: bool = False,
    recipes: Any = None,
    native_bindings: Any = None,
) -> None:
    """Hard-lock Simulation Mode (``allow_unbound_formula=False``) against unbound Layer-1 ops.

    Never trusts ``has_bound_provider`` alone; every op must be genuinely bound or carry
    an explicit non-unbound ``NativeCallBinding`` / concrete recipe.
    """
    del has_bound_provider
    if allow_unbound_formula:
        raise ExtensionError(f"simulation_mode_forbids_unbound_formula_ops:{entry}")
    for item in _iter_ops_items(ops):
        offender = _is_unbound_op_item(item)
        if offender is not None and not _item_has_verified_binding(item, recipes=recipes, native_bindings=native_bindings):
            raise CapabilityGap(
                capability=capability,
                entry=entry,
                kind="unmodeled_capability",
                code="unbound_operator_implementation",
                message=(
                    f"unbound_operator_implementation:{offender}: Simulation Mode "
                    "(allow_unbound_formula=False) requires a Layer-2 bound implementation "
                    "registered via bind_kernel_implementation(); for Layer-1 formula-driven "
                    "ceiling projection use simulate_formula_kernel() or model_target_ceiling()."
                ),
                affected_results=("operator_sum_ms", "duration_ms", "timeline"),
                minimal_replay_input={"entry": entry, "unbound_op": offender},
                fallback="not_attempted",
            )


def _strip_unbound_metadata(op: OpSpec, *, kernel_name: str) -> OpSpec:
    meta = dict(op.metadata)
    meta.pop("unbound_formula_op", None)
    meta["binding_state"] = BINDING_STATE_BOUND
    meta.setdefault("bound_kernel_name", kernel_name)
    return replace(op, metadata=tuple(sorted(meta.items())))


def _normalize_recipe_book(recipe: Any, *, slug: str) -> Any:
    """Normalize a concrete recipe, recipe mapping, or ``RecipeBook``."""
    if recipe is None:
        return None
    from mindie_sim.prediction.program import RecipeBook
    if isinstance(recipe, RecipeBook):
        return recipe
    if isinstance(recipe, Mapping):
        if any(key in recipe for key in ("recipes", "families", "programs", "lowerings")):
            return RecipeBook.from_mapping(recipe)
        return RecipeBook(recipes=dict(recipe))
    if callable(getattr(recipe, "lower", None)) or callable(getattr(recipe, "prepare", None)):
        return RecipeBook(recipes={slug: recipe})
    raise ExtensionError(f"bind_kernel_implementation_recipe_invalid:{slug}")


def _merge_recipe_books(*books: Any) -> Any:
    """Combine ``RecipeBook`` instances in lowest-to-highest precedence order."""
    active = [book for book in books if book is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    from mindie_sim.prediction.program import RecipeBook
    first = active[0]
    merged = RecipeBook(
        recipes={},
        families={},
        programs={},
        default=getattr(first, "default", None),
        family_registry=getattr(first, "_family_registry", None),
    )
    base_recipes = set(merged.recipes)
    base_families = set(merged.families)
    base_programs = set(merged.programs)
    recipes_map = dict(first.recipes)
    families_map = dict(first.families)
    programs_map = dict(first.programs)
    explicit_recipes = set(getattr(first, "_explicit_recipes", ()))
    caller_recipes = set(getattr(first, "_caller_recipes", ()))
    explicit_families = set(getattr(first, "_explicit_families", ()))
    caller_programs = set(getattr(first, "_caller_programs", ()))
    family_kinds = dict(getattr(first, "_family_kinds", {}))
    default = getattr(first, "default", None)
    for book in active[1:]:
        b_explicit_recipes = set(getattr(book, "_explicit_recipes", ()))
        b_caller_recipes = set(getattr(book, "_caller_recipes", ()))
        b_explicit_families = set(getattr(book, "_explicit_families", ()))
        b_caller_programs = set(getattr(book, "_caller_programs", ()))
        for key, val in book.recipes.items():
            if key in b_caller_recipes:
                recipes_map[key] = val
            elif key not in caller_recipes and (
                key not in base_recipes
                or (key in b_explicit_recipes and key not in explicit_recipes)
                or key not in recipes_map
            ):
                recipes_map[key] = val
        for key, val in book.families.items():
            if key in b_explicit_families:
                families_map[key] = val
            elif key not in explicit_families and (key not in base_families or key not in families_map):
                families_map[key] = val
        for key, val in book.programs.items():
            if key in b_caller_programs:
                programs_map[key] = val
            elif key not in caller_programs and (key not in base_programs or key not in programs_map):
                programs_map[key] = val
        explicit_recipes.update(b_explicit_recipes)
        caller_recipes.update(b_caller_recipes)
        explicit_families.update(b_explicit_families)
        caller_programs.update(b_caller_programs)
        family_kinds.update(getattr(book, "_family_kinds", {}))
        if getattr(book, "default", None) is not None:
            default = book.default
    merged.recipes = recipes_map
    merged.families = families_map
    merged.programs = programs_map
    merged._explicit_recipes = frozenset(explicit_recipes)
    merged._caller_recipes = frozenset(caller_recipes)
    merged._explicit_families = frozenset(explicit_families)
    merged._caller_programs = frozenset(caller_programs)
    merged._family_kinds = family_kinds
    merged.default = default
    return merged


def _canonical_aclnn_entry(symbol: Any) -> str | None:
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    cleaned = symbol.strip()
    if cleaned.startswith("aclnn."):
        cleaned = cleaned.split(".", 1)[1]
    from mindie_sim.prediction.call_builders import ACLNN_ABI_ALIASES, ACLNN_ABI_REGISTRY
    if cleaned in ACLNN_ABI_REGISTRY:
        return cleaned
    return ACLNN_ABI_ALIASES.get(cleaned)


def _op_aclnn_symbols(op: OpSpec) -> tuple[str, ...]:
    meta = dict(op.metadata)
    found: list[str] = []
    for key in ("aclnn_symbol", "api", "aclnn_api"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            found.append(val.strip())
    for nc_key in ("native_call", "native_call_binding"):
        native_call = meta.get(nc_key)
        if not isinstance(native_call, Mapping):
            continue
        api_map = native_call.get("api")
        if not isinstance(api_map, Mapping):
            continue
        for key in ("aclnn_symbol", "entry"):
            val = api_map.get(key)
            if isinstance(val, str) and val.strip():
                found.append(val.strip())
    return tuple(dict.fromkeys(found))


def _formula_matches_kernel(
    item: FormulaKernelSpec,
    *,
    kernel_name: str,
    abi_signature_id: str,
    resolved_entry: str | None = None,
) -> bool:
    if abi_signature_id:
        return item.resolved_abi_signature() == abi_signature_id
    if item.name == kernel_name:
        return True
    return resolved_entry is not None and _canonical_aclnn_entry(item.name) == resolved_entry


def _op_matches_kernel_identity(
    op: OpSpec, *, kernel_name: str, slug: str, abi_signature_id: str,
) -> bool:
    meta = dict(op.metadata)
    op_abi = str(meta.get("abi_signature_id") or "").strip()
    if abi_signature_id and op_abi:
        return op_abi == abi_signature_id
    if op.op_id in {kernel_name, slug} or op.recipe in {kernel_name, slug}:
        return True
    target_entry = _canonical_aclnn_entry(kernel_name)
    for key in _KERNEL_IDENTITY_META_KEYS:
        val = meta.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        if val.strip() in {kernel_name, slug}:
            return True
        if target_entry is not None and _canonical_aclnn_entry(val) == target_entry:
            return True
    return False


def _match_recipe_for_op(
    op: OpSpec, book: Any, *, kernel_name: str, slug: str, abi_signature_id: str,
) -> str | None:
    """Return concrete recipe ID (or current recipe for family/program match), else ``None``."""
    caller_recipes = frozenset(getattr(book, "_caller_recipes", ()))
    if op.recipe and op.recipe in caller_recipes and _recipe_is_concrete(book.recipes.get(op.recipe)):
        return op.recipe
    if _book_selects_explicit_recipe(op, book):
        return op.recipe
    if _op_matches_kernel_identity(
        op, kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
    ):
        for candidate in (op.op_id, kernel_name, slug):
            if candidate and candidate in caller_recipes and _recipe_is_concrete(book.recipes.get(candidate)):
                return candidate
        if len(caller_recipes) == 1:
            only = next(iter(caller_recipes))
            if _recipe_is_concrete(book.recipes.get(only)):
                return only
        if not caller_recipes and _recipe_is_concrete(getattr(book, "default", None)):
            return op.recipe
    return None


def _bind_formula_atomics(
    atomic_ops: Sequence[OpSpec],
    book: Any,
    *,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> tuple[OpSpec, ...] | None:
    caller_recipes = frozenset(getattr(book, "_caller_recipes", ()))
    fallback_recipe: str | None = None
    for candidate in (kernel_name, slug):
        if candidate in caller_recipes and _recipe_is_concrete(book.recipes.get(candidate)):
            fallback_recipe = candidate
            break
    if fallback_recipe is None and len(caller_recipes) == 1:
        only = next(iter(caller_recipes))
        if _recipe_is_concrete(book.recipes.get(only)):
            fallback_recipe = only
    has_default = not caller_recipes and _recipe_is_concrete(getattr(book, "default", None))
    bound: list[OpSpec] = []
    for atomic in atomic_ops:
        matched = _match_recipe_for_op(
            atomic, book,
            kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
        )
        if matched is None:
            if fallback_recipe is not None:
                matched = fallback_recipe
            elif has_default:
                matched = atomic.recipe
            else:
                return None
        updated = replace(atomic, recipe=matched) if matched and matched != atomic.recipe else atomic
        bound.append(_strip_unbound_metadata(updated, kernel_name=kernel_name))
    return tuple(bound)


def _bind_recipe_group(
    items: tuple[Any, ...],
    book: Any,
    *,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> tuple[Any, ...]:
    converted: list[Any] = []
    for item in items:
        if isinstance(item, FormulaKernelSpec):
            if (
                book is not None
                and item.atomic_ops
                and _formula_matches_kernel(
                    item, kernel_name=kernel_name, abi_signature_id=abi_signature_id,
                )
            ):
                bound_atomics = _bind_formula_atomics(
                    item.atomic_ops, book,
                    kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                )
                if bound_atomics is not None:
                    converted.extend(bound_atomics)
                    continue
            converted.append(item)
            continue
        if isinstance(item, OpSpec) and book is not None:
            matched = _match_recipe_for_op(
                item, book,
                kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
            )
            if matched is not None:
                updated = replace(item, recipe=matched) if matched and matched != item.recipe else item
                converted.append(_strip_unbound_metadata(updated, kernel_name=kernel_name))
                continue
        converted.append(item)
    return tuple(converted)


def _item_matches_aclnn(
    item: Any,
    *,
    target_symbol: str,
    resolved_entry: str,
    target_recipe: str,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> bool:
    if isinstance(item, OpSpec):
        symbols = _op_aclnn_symbols(item)
        if symbols:
            return any(
                sym.removeprefix("aclnn.") == target_symbol.removeprefix("aclnn.")
                or _canonical_aclnn_entry(sym) == resolved_entry
                for sym in symbols
            )
        if item.recipe and item.recipe == target_recipe:
            return True
        return _op_matches_kernel_identity(
            item, kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
        )
    if isinstance(item, FormulaKernelSpec):
        return _formula_matches_kernel(
            item, kernel_name=kernel_name, abi_signature_id=abi_signature_id,
            resolved_entry=resolved_entry,
        )
    if isinstance(item, Mapping):
        raw_sym = (
            item.get("api") or item.get("symbol")
            or item.get("aclnn_symbol") or item.get("entry")
        )
        if isinstance(raw_sym, str) and raw_sym.strip():
            return (
                raw_sym.strip().removeprefix("aclnn.") == target_symbol.removeprefix("aclnn.")
                or _canonical_aclnn_entry(raw_sym) == resolved_entry
            )
        name = str(item.get("op_id") or item.get("name") or "")
        return bool(name) and name in {kernel_name, slug}
    return False


def _is_exact_aclnn_op(
    op: OpSpec, *, resolved_entry: str, target_recipe: str,
) -> bool:
    from mindie_sim.prediction.call_builders import ACLNN_ABI_SCHEMA
    meta = dict(op.metadata)
    if meta.get("aclnn_abi_schema") != ACLNN_ABI_SCHEMA:
        return False
    if op.recipe != target_recipe:
        return False
    symbols = _op_aclnn_symbols(op)
    return bool(symbols) and any(
        _canonical_aclnn_entry(sym) == resolved_entry for sym in symbols
    )


def _extract_op_native_evidence(meta: Mapping[str, Any], args: dict[str, Any]) -> None:
    for nc_key in ("native_call", "native_call_binding"):
        native_call = meta.get(nc_key)
        if not isinstance(native_call, Mapping):
            continue
        if "source" not in args and isinstance(native_call.get("source"), Mapping):
            args["source"] = dict(native_call["source"])
        if "kernels" not in args and native_call.get("kernels") is not None:
            args["kernels"] = native_call["kernels"]
    if "source" not in args and isinstance(meta.get("source"), Mapping):
        args["source"] = dict(meta["source"])


def _populate_matrix_aclnn_args(
    op: OpSpec, meta: Mapping[str, Any], args: dict[str, Any],
    *, kind: str, override_keys: frozenset[str],
) -> None:
    skip_shape = bool(override_keys & _MATRIX_SHAPE_OVERRIDE_KEYS)
    geom = meta.get("aclnn_geometry")
    if isinstance(geom, Mapping) and not skip_shape:
        groups = geom.get("matrix_groups")
        m_val, n_val, k_val = geom.get("m"), geom.get("n"), geom.get("k")
        if (
            kind == "matrix"
            and isinstance(groups, int) and groups > 1
            and isinstance(m_val, int) and isinstance(n_val, int) and isinstance(k_val, int)
            and m_val % groups == 0
            and "tensors" not in args
        ):
            args["x_shape"] = (groups, m_val // groups, k_val)
            args["weight_shape"] = (groups, k_val, n_val)
        else:
            for key in ("m", "n", "k"):
                if geom.get(key) is not None:
                    args[key] = geom[key]
        if kind == "grouped":
            if geom.get("group_type") is not None:
                args["group_type"] = geom["group_type"]
            if meta.get("aclnn_group_list") is not None:
                args["expert_rows"] = list(meta["aclnn_group_list"])
            elif groups is not None:
                args["groups"] = groups
    elif not skip_shape and "tensors" not in args and len(op.local_shape) == _MATRIX_SHAPE_DIMS:
        args["m"], args["n"], args["k"] = op.local_shape
        if kind == "grouped":
            if meta.get("group_type") is not None:
                args["group_type"] = meta["group_type"]
            if meta.get("aclnn_group_list") is not None:
                args["expert_rows"] = list(meta["aclnn_group_list"])
            elif meta.get("matrix_groups") is not None:
                args["groups"] = meta["matrix_groups"]
    abi = op.precision_abi
    act_dtype = (abi.input_dtype if abi else "") or meta.get("activation_format") or op.dtype
    wt_dtype = (abi.weight_dtype if abi else "") or meta.get("weight_format")
    out_dtype = (abi.output_dtype if abi else "") or meta.get("output_format")
    if act_dtype:
        args["activation_dtype"] = act_dtype
    if wt_dtype:
        args["weight_dtype"] = wt_dtype
    if out_dtype:
        args["output_dtype"] = out_dtype


def _aclnn_args_from_op(
    op: OpSpec, *, symbol: str, kind: str,
    override_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Extract actual shapes, dtypes, attrs, and source evidence from one matched ``OpSpec``."""
    meta = dict(op.metadata)
    args: dict[str, Any] = {"api": symbol}
    if op.phase:
        args["phase"] = op.phase
    if op.dtype:
        args["dtype"] = op.dtype
    if isinstance(meta.get("tensors"), Mapping):
        args["tensors"] = dict(meta["tensors"])
    _extract_op_native_evidence(meta, args)
    attrs: dict[str, Any] = {}
    if isinstance(meta.get("attrs"), Mapping):
        attrs.update(meta["attrs"])
    if kind in {"matrix", "grouped"}:
        _populate_matrix_aclnn_args(
            op, meta, args, kind=kind, override_keys=override_keys,
        )
    elif kind == "attention":
        attn_geom = meta.get("attention_geometry")
        if isinstance(attn_geom, Mapping):
            for key in _ATTN_GEOMETRY_ARG_KEYS:
                if attn_geom.get(key) is not None:
                    args[key] = attn_geom[key]
            for key in _ATTN_GEOMETRY_ATTR_KEYS:
                if attn_geom.get(key) is not None:
                    attrs.setdefault(key, attn_geom[key])
        elif "tensors" not in args and isinstance(meta.get("attention_tensor_view"), Mapping):
            view = meta["attention_tensor_view"]
            tensors = {}
            for role, skey, dkey in (
                ("query", "query_shape", "query_dtype"),
                ("key", "key_shape", "key_dtype"),
                ("value", "value_shape", "value_dtype"),
            ):
                if view.get(skey):
                    tensors[role] = {"shape": list(view[skey]), "dtype": view.get(dkey) or op.dtype or "bf16"}
            if tensors:
                args["tensors"] = tensors
            if view.get("output_dtype"):
                args["output_dtype"] = view["output_dtype"]
            if "causal" in view:
                attrs.setdefault("causal", view["causal"])
    else:
        if meta.get("input_format"):
            args["dtype"] = meta["input_format"]
        if meta.get("output_format"):
            args["output_dtype"] = meta["output_format"]
        if not (override_keys & _VECTOR_SHAPE_OVERRIDE_KEYS) and "tensors" not in args and op.local_shape:
            if len(op.local_shape) == 1:
                args["elements"] = op.local_shape[0]
            else:
                rows = math.prod(op.local_shape[:-1])
                cols = op.local_shape[-1]
                gamma_dtype = str(meta.get("norm_gamma_dtype") or "")
                act_dtype = str(args.get("dtype") or op.dtype or "bf16")
                if gamma_dtype and gamma_dtype != act_dtype:
                    args["tensors"] = {
                        "x": {"shape": [rows, cols], "dtype": act_dtype},
                        "gamma": {"shape": [cols], "dtype": gamma_dtype},
                    }
                else:
                    args["rows"] = rows
                    args["columns"] = cols
    if attrs:
        args["attrs"] = attrs
    return args


def _aclnn_args_from_item(
    item: Any, *, symbol: str, kind: str, call_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    override_keys = frozenset(k for k in call_kwargs if k not in _BUILDER_PROTOCOL_KEYS)
    if isinstance(item, OpSpec):
        args = _aclnn_args_from_op(item, symbol=symbol, kind=kind, override_keys=override_keys)
    elif isinstance(item, FormulaKernelSpec):
        args = {"api": symbol}
        tensors = {
            tensor.name: {"shape": list(tensor.shape), "dtype": tensor.dtype}
            for tensor in (*item.inputs, *item.weights, *item.outputs)
            if tensor.shape
        }
        if tensors:
            args["tensors"] = tensors
        if item.inputs and item.inputs[0].dtype:
            args["dtype"] = item.inputs[0].dtype
        if item.outputs and item.outputs[0].dtype:
            args["output_dtype"] = item.outputs[0].dtype
    elif isinstance(item, Mapping):
        args = {k: v for k, v in item.items() if k not in _BUILDER_PROTOCOL_KEYS}
        args["api"] = symbol
    else:
        args = {"api": symbol}
    for key, value in call_kwargs.items():
        if key in _BUILDER_PROTOCOL_KEYS:
            continue
        if key in {"attrs", "tensors"} and isinstance(value, Mapping) and isinstance(args.get(key), Mapping):
            merged_map = dict(args[key])
            merged_map.update(value)
            args[key] = merged_map
        else:
            args[key] = value
    return args


def _map_ops_structure(
    ops: Any, transform_items: Callable[[tuple[Any, ...]], tuple[Any, ...]],
) -> Any:
    """Apply ``transform_items`` while preserving rank-indexed ``{replica: ops}`` maps."""
    if _is_rank_mapping(ops):
        return {
            rank: transform_items(_iter_ops_items(group))
            for rank, group in ops.items()
        }
    return transform_items(_iter_ops_items(ops))


def _prepare_impl_sequence(
    raw_items: tuple[Any, ...],
    merged_book: Any,
    *,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> tuple[Any, ...]:
    bound_ops: list[Any] = []
    for item in raw_items:
        if not isinstance(item, OpSpec):
            bound_ops.append(item)
            continue
        if _is_unbound_op_item(item) is not None:
            matched = (
                _match_recipe_for_op(
                    item, merged_book,
                    kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                )
                if merged_book is not None else None
            )
            if matched is not None:
                updated = replace(item, recipe=matched) if matched and matched != item.recipe else item
                bound_ops.append(_strip_unbound_metadata(updated, kernel_name=kernel_name))
            else:
                bound_ops.append(item)
        else:
            bound_ops.append(_strip_unbound_metadata(item, kernel_name=kernel_name))
    return tuple(bound_ops)


def _attach_entry_deps(replacement: tuple[Any, ...], inherited_deps: tuple[str, ...]) -> tuple[Any, ...]:
    if not inherited_deps:
        return replacement
    internal_ids = {op.op_id for op in replacement if isinstance(op, OpSpec) and op.op_id}
    patched: list[Any] = []
    for op in replacement:
        if isinstance(op, OpSpec) and not (set(op.deps) & internal_ids):
            merged_deps = tuple(dict.fromkeys((*inherited_deps, *op.deps)))
            patched.append(replace(op, deps=merged_deps))
        else:
            patched.append(op)
    return tuple(patched)


def _terminal_op_ids(replacement: tuple[Any, ...]) -> tuple[str, ...]:
    ops = [op for op in replacement if isinstance(op, OpSpec) and op.op_id]
    if not ops:
        return ()
    internal_ids = {op.op_id for op in ops}
    consumed = {dep for op in ops for dep in op.deps if dep in internal_ids}
    terminals = tuple(op.op_id for op in ops if op.op_id not in consumed)
    return terminals or (ops[-1].op_id,)


def _namespace_replacement_ops(
    replacement: tuple[Any, ...],
    prefix: str,
    used_ids: set[str],
) -> tuple[Any, ...]:
    rename: dict[str, str] = {}
    for idx, op in enumerate(replacement):
        if isinstance(op, OpSpec) and op.op_id:
            base = f"{prefix}/{op.op_id}" if prefix else op.op_id
            cand = base
            seq = 1
            while cand in used_ids or cand in rename.values():
                cand = f"{base}#{seq}"
                seq += 1
            rename[op.op_id] = cand
            used_ids.add(cand)
    if not rename:
        return replacement
    patched: list[Any] = []
    for op in replacement:
        if isinstance(op, OpSpec) and op.op_id in rename:
            new_deps = tuple(rename.get(dep, dep) for dep in op.deps)
            patched.append(replace(op, op_id=rename[op.op_id], deps=new_deps))
        else:
            patched.append(op)
    return tuple(patched)


def _replace_sequence_with_impl(
    items: tuple[Any, ...],
    bound_impl_ops: tuple[Any, ...],
    *,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> tuple[Any, ...]:
    impl_by_id = {
        op.op_id: op for op in bound_impl_ops
        if isinstance(op, OpSpec) and op.op_id
    }
    incoming_ids = {
        op.op_id for op in items
        if isinstance(op, OpSpec) and op.op_id
    }
    # Use 1:1 per-op_id mapping only for single-op replacements or when multiple
    # distinct ops in the incoming group already match distinct implementation ops.
    use_per_id = len(bound_impl_ops) == 1 or len(set(impl_by_id) & incoming_ids) > 1
    match_flags: list[tuple[bool, bool]] = []
    for item in items:
        is_per_id = use_per_id and isinstance(item, OpSpec) and item.op_id in impl_by_id
        is_seq_match = not is_per_id and (
            (isinstance(item, OpSpec) and (
                (not use_per_id and item.op_id in impl_by_id)
                or _op_matches_kernel_identity(
                    item, kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                )
            ))
            or (isinstance(item, FormulaKernelSpec) and _formula_matches_kernel(
                item, kernel_name=kernel_name, abi_signature_id=abi_signature_id,
            ))
        )
        match_flags.append((is_per_id, is_seq_match))
    seq_match_count = sum(1 for _, is_seq in match_flags if is_seq)
    used_ids: set[str] = {
        item.op_id for item, (_is_per_id, is_seq) in zip(items, match_flags)
        if isinstance(item, OpSpec) and item.op_id and not is_seq
    }
    entries: list[tuple[Any, str | None]] = []
    id_remap: dict[str, tuple[str, ...]] = {}
    occ_idx = 0
    for item, (is_per_id, is_seq_match) in zip(items, match_flags):
        if is_per_id:
            rep_op = impl_by_id[item.op_id]
            inherited = tuple(dep for dep in item.deps if dep != rep_op.op_id)
            if inherited:
                merged_deps = tuple(dict.fromkeys((*inherited, *rep_op.deps)))
                rep_op = replace(rep_op, deps=merged_deps)
            used_ids.add(rep_op.op_id)
            entries.append((rep_op, item.op_id))
            continue
        if is_seq_match:
            occ_idx += 1
            inherited = item.deps if isinstance(item, OpSpec) else ()
            source_id = item.op_id if isinstance(item, OpSpec) else None
            impl_ids = {op.op_id for op in bound_impl_ops if isinstance(op, OpSpec) and op.op_id}
            needs_namespace = seq_match_count > 1 or bool(impl_ids & used_ids)
            if needs_namespace:
                prefix = source_id or f"call{occ_idx}"
                scoped_ops = _namespace_replacement_ops(bound_impl_ops, prefix, used_ids)
            else:
                used_ids.update(impl_ids)
                scoped_ops = bound_impl_ops
            seq = _attach_entry_deps(scoped_ops, inherited)
            for elem in seq:
                entries.append((elem, source_id))
            if source_id:
                terminals = _terminal_op_ids(seq)
                if terminals and terminals != (source_id,):
                    id_remap[source_id] = terminals
        else:
            entries.append((item, None))
    if not id_remap:
        return tuple(elem for elem, _ in entries)
    rewritten: list[Any] = []
    for elem, source_id in entries:
        if isinstance(elem, OpSpec) and elem.deps:
            new_deps: list[str] = []
            changed = False
            for dep in elem.deps:
                if dep in id_remap and dep != source_id:
                    new_deps.extend(id_remap[dep])
                    changed = True
                else:
                    new_deps.append(dep)
            if changed:
                rewritten.append(replace(elem, deps=tuple(dict.fromkeys(new_deps))))
                continue
        rewritten.append(elem)
    return tuple(rewritten)


def _bind_implementation_mapping(
    ops: Any,
    implementation: Mapping[str, Any],
    *,
    recipe_book: Any,
    caller_recipes: Any = None,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
) -> dict[str, Any]:
    out = dict(implementation)
    caller_book = (
        _normalize_recipe_book(caller_recipes, slug=slug)
        if caller_recipes is not None else None
    )
    impl_book = (
        _normalize_recipe_book(out.get("recipes"), slug=slug)
        if "recipes" in out else None
    )
    merged_book = _merge_recipe_books(caller_book, recipe_book, impl_book)
    if merged_book is not None:
        out["recipes"] = merged_book
    if "ops" not in out and "operators" not in out:
        out["ops"] = _map_ops_structure(
            ops,
            lambda group: _bind_recipe_group(
                group, merged_book,
                kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
            ),
        )
        return out

    raw_impl_ops = out.get("ops", out.get("operators"))
    out.pop("operators", None)
    if _is_rank_mapping(raw_impl_ops):
        impl_by_rank = {
            rank: _prepare_impl_sequence(
                _iter_ops_items(rank_ops), merged_book,
                kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
            )
            for rank, rank_ops in raw_impl_ops.items()
        }
        if not _iter_ops_items(ops):
            out["ops"] = impl_by_rank
            return out
        if _is_rank_mapping(ops):
            out["ops"] = {
                rank: (
                    _replace_sequence_with_impl(
                        _iter_ops_items(group), impl_by_rank[rank],
                        kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                    )
                    if rank in impl_by_rank else _iter_ops_items(group)
                )
                for rank, group in ops.items()
            }
            return out
        flat_impl = impl_by_rank.get(0, next(iter(impl_by_rank.values())))
        out["ops"] = _replace_sequence_with_impl(
            _iter_ops_items(ops), flat_impl,
            kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
        )
        return out

    bound_impl_ops = _prepare_impl_sequence(
        _iter_ops_items(raw_impl_ops), merged_book,
        kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
    )
    if not _iter_ops_items(ops):
        out["ops"] = bound_impl_ops
        return out
    out["ops"] = _map_ops_structure(
        ops,
        lambda group: _replace_sequence_with_impl(
            group, bound_impl_ops,
            kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
        ),
    )
    return out


def _bind_aclnn_ops(
    ops: Any,
    *,
    target_symbol: str,
    recipe_book: Any,
    kernel_name: str,
    slug: str,
    abi_signature_id: str,
    call_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    from mindie_sim.prediction.call_builders import (
        _recipes as _aclnn_default_recipes,
        build_call,
        resolve_aclnn_abi,
    )
    clean_symbol = target_symbol.strip()
    if clean_symbol.startswith("aclnn."):
        clean_symbol = clean_symbol.split(".", 1)[1]
    resolved_entry, (target_kind, target_recipe, _target_family, _mech) = resolve_aclnn_abi(clean_symbol)
    override_keys = tuple(k for k in call_kwargs if k not in _BUILDER_PROTOCOL_KEYS)
    default_books: list[Any] = []
    consumed_overrides = False

    def _lower_group(items: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal consumed_overrides
        lowered: list[Any] = []
        for item in items:
            if not _item_matches_aclnn(
                item,
                target_symbol=clean_symbol,
                resolved_entry=resolved_entry,
                target_recipe=target_recipe,
                kernel_name=kernel_name,
                slug=slug,
                abi_signature_id=abi_signature_id,
            ):
                lowered.append(item)
                continue
            if (
                not override_keys
                and isinstance(item, OpSpec)
                and _is_exact_aclnn_op(
                    item, resolved_entry=resolved_entry, target_recipe=target_recipe,
                )
            ):
                default_books.append(_aclnn_default_recipes(target_kind, target_recipe)[0])
                lowered.append(_strip_unbound_metadata(item, kernel_name=kernel_name))
                continue
            args = _aclnn_args_from_item(
                item, symbol=clean_symbol, kind=target_kind, call_kwargs=call_kwargs,
            )
            if override_keys:
                consumed_overrides = True
            built = build_call({"builder": "aclnn", "arguments": args})
            default_books.append(built.recipes)
            if isinstance(item, OpSpec) and len(built.ops) == 1:
                rebuilt = replace(
                    built.ops[0],
                    op_id=item.op_id, deps=item.deps, stage=item.stage,
                    layer=item.layer, requests=item.requests, stream=item.stream,
                )
                lowered.append(_strip_unbound_metadata(rebuilt, kernel_name=kernel_name))
            else:
                lowered.extend(
                    _strip_unbound_metadata(op, kernel_name=kernel_name)
                    for op in built.ops
                )
        return tuple(lowered)

    if not _iter_ops_items(ops) and override_keys:
        args = _aclnn_args_from_item(
            {}, symbol=clean_symbol, kind=target_kind, call_kwargs=call_kwargs,
        )
        consumed_overrides = True
        built = build_call({"builder": "aclnn", "arguments": args})
        default_books.append(built.recipes)
        result_ops: Any = tuple(
            _strip_unbound_metadata(op, kernel_name=kernel_name) for op in built.ops
        )
    else:
        result_ops = _map_ops_structure(ops, _lower_group)
    caller_book = (
        _normalize_recipe_book(call_kwargs["recipes"], slug=slug)
        if call_kwargs.get("recipes") is not None else None
    )
    merged = _merge_recipe_books(*default_books, caller_book, recipe_book)
    payload: dict[str, Any] = {"ops": result_ops}
    if merged is not None:
        payload["recipes"] = merged
    if consumed_overrides:
        payload["consumed_kwargs"] = override_keys
    return payload


def bind_kernel_implementation(
    spec_or_name: FormulaKernelSpec | str,
    *,
    implementation: Callable[..., Any] | Mapping[str, Any] | None = None,
    extension_id: str | None = None,
    version: str = "1.0.0",
    provider_name: str | None = None,
    capability: str = "operator",
    recipe: Any = None,
    aclnn_api: str | None = None,
    aclnn_symbol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    default: bool = True,
) -> Extension:
    """Layer-2 SDK: bind a formula kernel or custom op to a real ``Extension`` / ``Provider``.

    The returned immutable ``Extension`` can be loaded via ``load_extensions(extensions=[ext])``
    and passed to ``simulate_operator`` / ``simulate_native_call`` / ``simulate_graph`` in
    strict Simulation Mode (``allow_unbound_formula=False``).
    """
    kernel_name = spec_or_name.name if isinstance(spec_or_name, FormulaKernelSpec) else str(spec_or_name or "").strip()
    if not kernel_name:
        raise ExtensionError("bind_kernel_implementation_requires_kernel_name")
    if implementation is None and recipe is None and not (aclnn_api or aclnn_symbol):
        raise ExtensionError(f"bind_kernel_implementation_requires_real_binding:{kernel_name}")

    slug = _SLUG_RE.sub("_", kernel_name.lower()).strip("_") or "custom_kernel"
    ext_id = extension_id or f"bound_{slug}"
    prov_name = provider_name or f"bound_{slug}"
    abi_signature_id = (
        spec_or_name.resolved_abi_signature()
        if isinstance(spec_or_name, FormulaKernelSpec) else ""
    )
    binding_meta: dict[str, Any] = {
        "binding_state": BINDING_STATE_BOUND,
        "bound_kernel_name": kernel_name,
        "layer": "layer2_bound_implementation",
    }
    if isinstance(spec_or_name, FormulaKernelSpec):
        binding_meta["equations"] = list(spec_or_name.equations)
        binding_meta["abi_signature_id"] = abi_signature_id
    if aclnn_api or aclnn_symbol:
        binding_meta["aclnn_api"] = str(aclnn_api or aclnn_symbol)
    if metadata:
        binding_meta.update(dict(metadata))

    if callable(implementation):
        purpose = "report" if capability in {"operator", "native", "graph"} and recipe is None and not (aclnn_api or aclnn_symbol) else "builder"

        def _callable_factory(*, case: Any, ops: Any, **kwargs: Any) -> Any:
            return implementation(case=case, ops=ops, **kwargs)

        provider = Provider(
            capability=capability,
            name=prov_name,
            factory=_callable_factory,
            purpose=purpose,
            version=version,
            default=default,
            description=f"Layer-2 bound implementation for {kernel_name}",
            metadata=binding_meta,
        )
        return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)

    normalized_recipe = _normalize_recipe_book(recipe, slug=slug)
    target_aclnn = str(aclnn_api or aclnn_symbol or "").strip()

    def _builder_factory(*, case: Any, ops: Any, **kwargs: Any) -> Mapping[str, Any]:
        del case
        if isinstance(implementation, Mapping):
            return _bind_implementation_mapping(
                ops, implementation,
                recipe_book=normalized_recipe,
                caller_recipes=kwargs.get("recipes"),
                kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
            )
        if target_aclnn:
            return _bind_aclnn_ops(
                ops,
                target_symbol=target_aclnn,
                recipe_book=normalized_recipe,
                kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                call_kwargs=kwargs,
            )
        caller_book = (
            _normalize_recipe_book(kwargs["recipes"], slug=slug)
            if kwargs.get("recipes") is not None else None
        )
        merged_book = _merge_recipe_books(caller_book, normalized_recipe)
        return {
            "ops": _map_ops_structure(
                ops,
                lambda group: _bind_recipe_group(
                    group, normalized_recipe,
                    kernel_name=kernel_name, slug=slug, abi_signature_id=abi_signature_id,
                ),
            ),
            "recipes": merged_book,
        }

    provider = Provider(
        capability=capability,
        name=prov_name,
        factory=_builder_factory,
        purpose="builder",
        version=version,
        default=default,
        description=f"Layer-2 recipe/ABI binding for {kernel_name}",
        metadata=binding_meta,
    )
    return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)


def model_target_ceiling(
    model: Any,
    hardware: Any = "a5",
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """First-class Modeling Mode entrypoint (``allow_unbound_formula=True``).

    Independently solves ``Before (Parallel0, B0)`` (current-stack simulation)
    vs ``After (Parallel1*, B1*)`` (target ceiling modeling with unimplemented
    algebraic kernel synthesis) under the declared SLA and HBM budget.
    """
    from mindie_sim.solve import solve_deployment

    call_kwargs = dict(kwargs)
    call_kwargs["mode"] = MODE_MODELING
    result = solve_deployment(model, hardware, **call_kwargs)
    result["analysis_mode"] = MODE_MODELING
    result["allow_unbound_formula"] = True
    return result


__all__ = [
    "FormulaKernelEvaluation",
    "FormulaKernelSpec",
    "FormulaTensorSpec",
    "bind_kernel_implementation",
    "enforce_simulation_binding_guard",
    "evaluate_formula_kernel",
    "model_target_ceiling",
    "simulate_formula_kernel",
]
