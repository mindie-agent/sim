"""One normalized actual-invocation identity for declared native call boundaries.

[Role] Join an existing ``OpSpec``/``OperatorPlan`` boundary to the real ACLNN,
  Triton, CANN-kernel or HCCL entry that performs it, and carry the typed call
  arguments and physical work into the structure/profiler consumer.
[Invariants] The record is metadata only: it never prices, selects a core,
  admits/rejects an operator or fabricates a symbol.  A source call may expand
  into several declared kernels; those children keep one ``call_id`` and are
  charged once at their timing owner.  Unknown evidence stays in ``unresolved``.
[Timeouts & Locks] Pure finite metadata arithmetic; no I/O, device or locks.
[Side-Effect Policy] Returns immutable bindings; never imports a hardware
  runtime, weights, the profiler or a model catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math

from mindie_sim.schema.common import ModelingError

SCHEMA = "mindie-sim/native-call-binding/1"
_MAX_NESTED = 64

#: Execution classes a producer may prove.  These are declarations, not a
#: name->core rule; anything else remains unresolved.
PROVED_CORES = frozenset(("aic", "aiv", "mix", "aicpu", "hccl", "host"))
_NAMESPACES = frozenset(("aclnn", "torch", "hccl", "cann_kernel", "triton"))


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _digest(value):
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _text(value):
    return value if isinstance(value, str) and value else None


def _source_record(raw):
    if not isinstance(raw, dict) or not raw:
        return None
    record = {
        "repository": _text(raw.get("repository")),
        "path": _text(raw.get("path")),
        "sha256": _text(raw.get("sha256")),
        "symbol": _text(raw.get("symbol")),
        "revision": _text(raw.get("revision") or raw.get("base_revision")),
        "state": _text(raw.get("source_state") or raw.get("state")) or "unknown",
    }
    return record if record["path"] or record["sha256"] else None


def _first_source(meta):
    for key in ("source", "source_reference", "native_source"):
        record = _source_record(meta.get(key))
        if record is not None:
            return record
    binding = meta.get("implementation_binding")
    if isinstance(binding, dict):
        record = _source_record(binding.get("source"))
        if record is not None:
            return record
    for declaration in meta.get("structure_subgraphs", ()):
        if isinstance(declaration, dict):
            record = _source_record(declaration.get("source"))
            if record is not None:
                return record
    return None


def _typed_payload(op, meta, *, include_launch=True):
    """Only declared call fields; no shape/attr is inferred from a kernel name.

    A fused call's grid belongs to each selected kernel, not to the shared call
    identity, so ``include_launch`` is false when declared children exist.
    """
    shapes = {}
    keys = ("local_shape", "padded_shape", "launch_shape") if include_launch else (
        "local_shape", "padded_shape")
    for key in keys:
        value = getattr(op, key, ())
        if value:
            shapes[key.replace("_shape", "")] = list(value)
    attrs = {}
    view = meta.get("native_tensor_view") or meta.get("attention_tensor_view")
    if isinstance(view, dict):
        for key, value in view.items():
            if key.endswith("_shape"):
                shapes[key[:-6]] = list(value) if isinstance(value, (list, tuple)) else value
            elif key not in {"kind"}:
                attrs[key] = _jsonable(value)
    for key in ("sharedkv_attention_geometry", "indexed_attention_geometry",
                "paged_mla_geometry"):
        geometry = meta.get(key)
        if isinstance(geometry, dict):
            attrs[key] = _jsonable(geometry)
    wrapper = meta.get("matrix_wrapper")
    if isinstance(wrapper, dict):
        for key in ("input_rank", "group_list_shape", "group_list_dtype",
                    "group_type", "split_item", "graph_mode"):
            if key in wrapper:
                attrs[key] = _jsonable(wrapper[key])
        shapes.setdefault("group_list", list(wrapper.get("group_list_shape", ())))
    selection = meta.get("hccl_selection")
    if isinstance(selection, dict):
        for key in ("expansion", "collective", "template", "ranks",
                    "payload_bytes", "dtype_bytes", "topology"):
            if key in selection:
                attrs["hccl_" + key] = _jsonable(selection[key])
        shapes.setdefault("payload", [selection.get("payload_bytes", 0)])
    typed = {
        "dtype": _text(getattr(op, "dtype", "")),
        "layout": _text(getattr(op, "layout", "")) or "contiguous",
        "shapes": {key: shapes[key] for key in sorted(shapes) if shapes[key]},
        "attrs": {key: attrs[key] for key in sorted(attrs)},
        "rank_width": max((len(value) for value in shapes.values()
                           if isinstance(value, (list, tuple))), default=0),
    }
    payload = {**typed, "matrix_flops": _jsonable(getattr(op, "matrix_flops", ()))}
    typed["signature_hash"] = _digest(payload)
    return typed


def _api_payload(op, meta, explicit):
    api = dict(explicit.get("api") or {}) if isinstance(explicit, dict) else {}
    namespace = _text(api.get("namespace"))
    entry = _text(api.get("entry"))
    symbol = _text(api.get("aclnn_symbol"))
    version = _text(api.get("version"))
    candidates = []
    if entry is None and symbol is not None:
        entry = symbol
    if namespace is None and symbol is not None:
        namespace = "aclnn"
    if entry is None and meta.get("aclnn_symbol"):
        entry = symbol = _text(meta["aclnn_symbol"])
        namespace = namespace or "aclnn"
    implementation = meta.get("native_implementation") or {}
    if not isinstance(implementation, dict):
        implementation = {}
    kinds = implementation.get("native_kinds") or ()
    native_kind = meta.get("native_kind")
    if native_kind:
        kinds = (*kinds, native_kind)
    if entry is None:
        declared = meta.get("native_api")
        if isinstance(declared, dict):
            namespace = namespace or _text(declared.get("namespace"))
            entry = _text(declared.get("entry"))
            symbol = symbol or _text(declared.get("aclnn_symbol"))
    if family_is_hccl(op, meta):
        namespace = namespace or "hccl"
        entry = entry or _text(meta.get("hccl_entry"))
        if symbol is not None:
            # An ACLNN symbol cannot name a genuine collective.
            raise ModelingError("hccl_call_cannot_carry_aclnn_symbol")
    if entry is None:
        candidates = sorted({str(kind) for kind in kinds if isinstance(kind, str)})
    if namespace is not None and namespace not in _NAMESPACES:
        raise ModelingError(f"native_call_namespace_unknown:{namespace}")
    return {
        "namespace": namespace,
        "entry": entry,
        "aclnn_symbol": symbol,
        "version": version,
        "candidates": candidates,
    }


def family_is_hccl(op, meta=None):
    """Genuine collective boundary by declared family/selection, never by name text."""
    meta = meta or {}
    return (op.family in {"collective", "hccl_boundary", "hccl_local", "allreduce",
                          "allgather", "reduce_scatter", "alltoall", "alltoallv"}
            or "hccl_selection" in meta or "hccl_phase" in meta)


def _kernel_records(op, meta, children):
    records = []
    explicit = meta.get("native_call")
    if isinstance(explicit, dict):
        for raw in explicit.get("kernels", ()):
            if isinstance(raw, dict) and raw.get("name"):
                record = {
                    "name": str(raw["name"]),
                    "role": _text(raw.get("role")) or "unknown",
                    "source_path": _text(raw.get("source_path")),
                    "source_sha256": _text(raw.get("source_sha256")),
                    "evidence": _text(raw.get("evidence")) or "source_declared",
                }
                for key in ("launch_shape", "source_stages", "variant", "count"):
                    if raw.get(key) is not None:
                        record[key] = _jsonable(raw[key])
                records.append(record)
    for child in children:
        if child is op or (child.op_id == op.op_id and child.family == "barrier"):
            # The expansion join is a logical barrier with the parent op_id,
            # never a native child kernel of the call.
            continue
        child_meta = dict(child.metadata)
        implementation = child_meta.get("native_implementation") or {}
        name = (child_meta.get("native_kind")
                or (implementation.get("native_kinds") or [None])[0]
                or child.recipe or child.family)
        source = _first_source(child_meta) or {}
        records.append({
            "name": str(name),
            "role": _text(child_meta.get("kernel_role")) or "unknown",
            "source_path": source.get("path"),
            "source_sha256": source.get("sha256"),
            "launch_shape": list(child.launch_shape) if child.launch_shape else None,
            "evidence": ("source_declared"
                         if child_meta.get("native_kind") or implementation else "unresolved"),
        })
    seen, unique = set(), []
    for record in records:
        if record["name"] in seen:
            continue
        seen.add(record["name"])
        unique.append(record)
    return tuple(unique)


def _explicit_core(meta):
    candidates = [meta.get("execution_core"), meta.get("invocation_class")]
    execution = meta.get("program_execution")
    if isinstance(execution, dict):
        candidates.append(execution.get("execution_class"))
    for value in candidates:
        if isinstance(value, str) and value in PROVED_CORES:
            return value
    return None


def _merges(explicit, derived):
    """Explicit producer fields win; missing keys still derive from the op."""
    if not isinstance(explicit, dict):
        return derived
    result = dict(derived)
    api = dict(derived.get("api") or {})
    api.update({key: value for key, value in (explicit.get("api") or {}).items()
                if value is not None})
    result["api"] = api
    typed = dict(derived.get("typed") or {})
    declared_typed = explicit.get("typed")
    if isinstance(declared_typed, dict):
        typed.update({key: value for key, value in declared_typed.items()
                      if key not in {"shapes", "attrs"}})
        for key in ("shapes", "attrs"):
            if isinstance(declared_typed.get(key), dict):
                typed[key] = {**(typed.get(key) or {}), **declared_typed[key]}
        typed["signature_hash"] = _digest({
            key: typed[key] for key in
            ("dtype", "layout", "shapes", "attrs", "occurrence_attrs", "rank_width")
            if key in typed})
    result["typed"] = typed
    for key in ("source", "execution_core", "unresolved", "non_claims"):
        if explicit.get(key) not in (None, (), [], {}):
            result[key] = explicit[key]
    if explicit.get("kernels"):
        result["kernels"] = explicit["kernels"]
    if explicit.get("members"):
        result["members"] = explicit["members"]
    if explicit.get("occurrence"):
        result["occurrence"] = _jsonable(explicit["occurrence"])
    if explicit.get("observed_aliases"):
        result["observed_aliases"] = _jsonable(explicit["observed_aliases"])
    return result


@dataclass(frozen=True)
class NativeCallBinding:
    """Immutable typed description of one actual native invocation."""

    call_id: str
    op_id: str
    family: str
    recipe: str
    source_boundary: str
    api: dict
    source: dict | None
    typed: dict
    occurrence: dict = field(default_factory=dict)
    observed_aliases: tuple[dict, ...] = ()
    kernels: tuple[dict, ...] = ()
    member_op_ids: tuple[str, ...] = ()
    execution_core: str | None = None
    timing_owner: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    physical: dict | None = None
    non_claims: tuple[str, ...] = (
        "static", "no_hardware_dispatch_claim", "no_measured_duration_claim")

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "call_id": self.call_id,
            "op_id": self.op_id,
            "family": self.family,
            "recipe": self.recipe,
            "source_boundary": self.source_boundary,
            "api": self.api,
            "source": self.source,
            "typed": self.typed,
            "occurrence": dict(self.occurrence),
            "observed_aliases": [dict(row) for row in self.observed_aliases],
            "kernels": [dict(row) for row in self.kernels],
            "members": {"op_ids": list(self.member_op_ids), "charge": "call",
                        "charge_note": "priced once at timing_owner.op_id"},
            "execution_core": self.execution_core,
            "timing_owner": dict(self.timing_owner),
            "evidence": dict(self.evidence),
            "unresolved": list(self.unresolved),
            "physical": self.physical,
            "non_claims": list(self.non_claims),
        }

    def prepare_args(self):
        """Complete typed invocation arguments for a prepare-only caller."""
        return {
            "dtype": self.typed.get("dtype"),
            "layout": self.typed.get("layout"),
            "shapes": _jsonable(self.typed.get("shapes") or {}),
            "attrs": _jsonable({
                **(self.typed.get("attrs") or {}),
                **(self.occurrence or {}),
            }),
        }

    def forward(self):
        """Consumer view: typed args + physical work, not a serialized ledger."""
        return {
            "call_id": self.call_id,
            "typed": self.typed,
            "api": self.api,
            "source": self.source,
            "physical": self.physical,
            "timing_owner": self.timing_owner,
        }


def _evidence(api, source, kernels, core):
    # "established" stays reserved for a verified symbol/entry; a declared
    # source identity is source_declared even when the entry string is exact.
    verified = api.get("verified") is True
    entry = ("established" if api.get("entry") and verified else
             "source_declared" if api.get("entry") else "unresolved")
    symbol = ("not_applicable" if api.get("namespace") == "hccl"
              else "established" if api.get("aclnn_symbol") and verified
              else "source_declared" if api.get("aclnn_symbol")
              else "unresolved")
    kernel_evidence = ("source_declared" if kernels and all(
        row["evidence"] == "source_declared" for row in kernels) else
        "unresolved" if not kernels else "partial")
    return {
        "entry": entry,
        "aclnn_symbol": symbol,
        "source": "source_declared" if source else "unresolved",
        "typed_args": "source_declared",
        "kernels": kernel_evidence,
        "core": "proved" if core else "not_declared",
    }


def declare_native_call(op, *, api=None, source=None, typed=None, kernels=None,
                        execution_core=None, unresolved=(), members=None,
                        occurrence=None, observed_aliases=None):
    """Attach a producer's provable fields to an existing operator.

    Producers supply only what their source establishes; BIND1 normalizes the
    identity.  An empty declaration is rejected rather than silently ignored.
    """
    payload = {}
    for key, value in (("api", api), ("source", source), ("typed", typed),
                       ("kernels", kernels), ("execution_core", execution_core),
                       ("unresolved", unresolved), ("members", members),
                       ("occurrence", occurrence),
                       ("observed_aliases", observed_aliases)):
        if value not in (None, (), [], {}):
            payload[key] = _jsonable(value)
    if not payload:
        raise ModelingError("native_call_declaration_empty")
    meta = dict(op.metadata)
    meta["native_call"] = payload
    return replace(op, metadata=tuple(meta.items()))


def bind_native_call(op, *, contract=None, implementation=None, children=(),
                     plan=None, prepared=None):
    """Bind one existing operator/plan boundary to its normalized native call.

    ``children`` are the declared kernels of an already-expanded source call
    (for example ``RecipeBook.expand((op,))``).  ``plan`` is the existing
    ``OperatorPlan`` when the boundary has been prepared; its work is forwarded
    unchanged as a physical view, never re-charged.
    """
    meta = dict(op.metadata)
    explicit = meta.get("native_call") or meta.get("native_call_binding")
    if not isinstance(explicit, dict):
        explicit = {}
    if not explicit:
        # HCCL phase/exit atoms are members of the enclosing collective call,
        # not independent invocations.  The entry (or an explicit declaration)
        # owns the call identity; phases must not mint a second one.
        if op.family == "hccl_local":
            return None
        if op.family == "hccl_boundary":
            group = meta.get("device_kernel_group")
            if not (isinstance(group, dict) and group.get("entry")):
                return None
        if op.family == "barrier":
            return None
    declared = tuple(
        child for child in children
        if child is not op
        and not (child.op_id == op.op_id and child.family == "barrier"))
    declared_fused = (len(declared) > 1
                      or len(explicit.get("kernels") or ()) > 1)
    derived = {
        "api": _api_payload(op, meta, explicit),
        "source": _first_source(meta) or (
            _source_record(explicit.get("source"))
            if isinstance(explicit, dict) else None),
        # A fused call identity excludes any single member's launch grid.
        "typed": _typed_payload(op, meta, include_launch=not declared_fused),
        "kernels": _kernel_records(op, meta, tuple(children)),
        "execution_core": _explicit_core(meta),
    }
    merged = _merges(explicit, derived)
    api = merged.get("api") or {}
    source = merged.get("source")
    typed = merged.get("typed") or {}
    occurrence = merged.get("occurrence") or {}
    observed_aliases = tuple(merged.get("observed_aliases") or ())
    kernels = tuple(merged.get("kernels") or ())
    core = merged.get("execution_core")
    if core is not None and core not in PROVED_CORES:
        raise ModelingError(f"native_call_core_not_proved:{core}")
    if api.get("namespace") == "hccl" and api.get("aclnn_symbol"):
        raise ModelingError("hccl_call_cannot_carry_aclnn_symbol")
    raw_members = merged.get("members")
    if isinstance(raw_members, dict):
        member_ids = tuple(str(value) for value in raw_members.get("op_ids", ()))
    elif isinstance(raw_members, (list, tuple)):
        member_ids = tuple(str(value) for value in raw_members)
    else:
        member_ids = ()
    if not member_ids:
        member_ids = tuple(child.op_id for child in declared) or (op.op_id,)
    identity = {
        "namespace": api.get("namespace"), "entry": api.get("entry"),
        "aclnn_symbol": api.get("aclnn_symbol"),
        "source_path": (source or {}).get("path"), "source_sha256": (source or {}).get("sha256"),
        "source_symbol": (source or {}).get("symbol"),
        "typed": typed, "kernels": [row["name"] for row in kernels],
    }
    call_id = _digest(identity)
    physical = None
    if plan is not None:
        work = getattr(plan, "work", None)
        if work is not None:
            physical = {
                "basis": "existing_operator_plan_work",
                "matrix": _jsonable(getattr(work, "matrix", {})),
                "resources": _jsonable(getattr(work, "resources", {})),
                "charged": "once_at_timing_owner",
            }
            prepared_physical = getattr(prepared, "flops", None)
            if prepared_physical is not None:
                physical["prepared_flops"] = prepared_physical
    unresolved = [str(value) for value in (merged.get("unresolved") or ())]
    if not api.get("entry"):
        unresolved.append("exact_native_entry_not_selected"
                          if api.get("candidates") else "exact_native_entry_unresolved")
    if api.get("namespace") != "hccl" and not api.get("aclnn_symbol"):
        unresolved.append("aclnn_symbol_unresolved")
    if api.get("namespace") == "hccl":
        unresolved.append("hccl_native_c_symbol_unresolved" if not api.get("entry")
                          else "hccl_native_c_symbol_not_claimed")
    if source is None:
        unresolved.append("attributable_source_reference_unresolved")
    elif not source.get("sha256"):
        unresolved.append("source_sha256_unresolved")
    if not kernels:
        unresolved.append("declared_child_kernels_unresolved")
    if core is None:
        unresolved.append("execution_core_not_declared")
    unresolved = tuple(dict.fromkeys(unresolved))
    owner = explicit.get("timing_owner")
    if not (isinstance(owner, dict) and owner.get("op_id")):
        owner = {"op_id": op.op_id, "scope": "call"}
    return NativeCallBinding(
        call_id=call_id, op_id=op.op_id, family=op.family, recipe=op.recipe,
        source_boundary=op.source_boundary, api=api, source=source, typed=typed,
        occurrence=occurrence, observed_aliases=observed_aliases,
        kernels=kernels, member_op_ids=member_ids, execution_core=core,
        timing_owner=owner,
        evidence=_evidence(api, source, kernels, core), unresolved=unresolved,
        physical=physical,
        non_claims=tuple(_jsonable(merged.get("non_claims") or self_non_claims())),
    )


def self_non_claims():
    return ("static", "no_hardware_dispatch_claim", "no_measured_duration_claim")


def validate_invocation(binding, args):
    """Prepare-only typed invocation check against the declared signature.

    A real invocation must carry every declared tensor role and attribute.
    Missing fields are rejected before they can silently validate as the same
    call; dynamic occurrence facts are declared separately on
    ``typed['occurrence_attrs']`` and are required too.  No device runtime is
    imported and no kernel is replayed.
    """
    if not isinstance(args, dict) or not args:
        raise ModelingError("native_call_invocation_requires_complete_typed_arguments")
    typed = binding.typed
    shapes = args.get("shapes") or {}
    attrs = args.get("attrs") or {}
    # Preserve mismatch-first diagnostics for an explicitly wrong value.
    for key in ("dtype", "layout"):
        declared, actual = typed.get(key), args.get(key)
        if declared is not None and actual is not None and declared != actual:
            raise ModelingError(f"native_call_signature_{key}_mismatch")
    for name, shape in (typed.get("shapes") or {}).items():
        actual = shapes.get(name)
        if actual is None:
            continue
        if tuple(actual) != tuple(shape):
            raise ModelingError(f"native_call_signature_shape_mismatch:{name}")
    for name, value in (typed.get("attrs") or {}).items():
        actual = attrs.get(name)
        if actual is not None and _jsonable(actual) != _jsonable(value):
            raise ModelingError(f"native_call_signature_attr_mismatch:{name}")
    missing = []
    for key in ("dtype", "layout"):
        if typed.get(key) is not None and args.get(key) is None:
            missing.append(key)
    missing.extend(name for name in (typed.get("shapes") or {})
                   if name not in shapes)
    missing.extend(name for name in (typed.get("attrs") or {})
                   if name not in attrs)
    for name, kind in (typed.get("occurrence_attrs") or {}).items():
        if name not in attrs:
            missing.append(name)
            continue
        value = attrs[name]
        if isinstance(kind, dict) and "const" in kind:
            if _jsonable(value) != _jsonable(kind["const"]):
                raise ModelingError(f"native_call_occurrence_attr_const_mismatch:{name}")
        elif kind in {"int", "positive_int"}:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ModelingError(f"native_call_occurrence_attr_not_int:{name}")
            if kind == "positive_int" and value < 1:
                raise ModelingError(f"native_call_occurrence_attr_not_positive:{name}")
        elif kind in {"int_list", "positive_int_list"}:
            if (not isinstance(value, (list, tuple)) or not value
                    or any(not isinstance(item, int) or isinstance(item, bool)
                           for item in value)):
                raise ModelingError(f"native_call_occurrence_attr_not_int_list:{name}")
            if kind == "positive_int_list" and any(item < 1 for item in value):
                raise ModelingError(f"native_call_occurrence_attr_not_positive:{name}")
        elif kind == "int_matrix":
            if (not isinstance(value, (list, tuple)) or not value
                    or any(not isinstance(row, (list, tuple)) or not row
                           or any(not isinstance(item, int) or isinstance(item, bool)
                                  for item in row) for row in value)):
                raise ModelingError(f"native_call_occurrence_attr_not_int_matrix:{name}")
            if any(item < 0 for row in value for item in row):
                raise ModelingError(f"native_call_occurrence_attr_not_nonnegative:{name}")
        elif kind == "string":
            if not isinstance(value, str) or not value:
                raise ModelingError(f"native_call_occurrence_attr_not_string:{name}")
        elif kind == "typed_ref":
            if (not isinstance(value, dict) or not value.get("ref")
                    or not isinstance(value.get("dtype"), str)
                    or not isinstance(value.get("rank"), int)
                    or isinstance(value.get("rank"), bool) or value["rank"] < 1):
                raise ModelingError(f"native_call_occurrence_attr_not_typed_ref:{name}")
        elif kind == "declared":
            if value is None:
                raise ModelingError(f"native_call_occurrence_attr_missing:{name}")
    if missing:
        raise ModelingError(
            "native_call_invocation_missing_fields:" + ",".join(sorted(set(missing))))
    return {
        "call_id": binding.call_id,
        "dtype": args.get("dtype"),
        "layout": args.get("layout"),
        "shapes": _jsonable(shapes),
        "attrs": _jsonable(attrs),
    }


def _bound_state(evidence, source_sha256, typed):
    """True bound = callable/entry OR declared kernel, plus source and typed ABI."""
    has_call = (evidence.get("entry") in {"established", "source_declared"}
                or evidence.get("aclnn_symbol") in {"established", "source_declared"}
                or evidence.get("kernels") in {"source_declared", "partial"})
    reasons = []
    if not has_call:
        reasons.append("no_bound_entry_or_kernel")
    if not source_sha256:
        reasons.append("source_sha256_unresolved")
    if not isinstance(typed, dict) or not typed.get("signature_hash"):
        reasons.append("typed_abi_unresolved")
    return has_call and not reasons, reasons


def bind_expanded_call(op, book):
    """Normal-path binding for a source call: children come from the real book."""
    if op.source_boundary == "call":
        children = tuple(book.expand((op,)))
    elif op.source_boundary == "kernel":
        recipe = book.select(op)
        children = (op,) if getattr(recipe, "kernels", None) is None else tuple(
            book.expand((op,)))
    else:
        children = (op,)
    return bind_native_call(op, children=children)


def native_call_coverage(ops, *, bindings=None):
    """Exact per-source-family denominator for the current normal emission.

    Never treats a self-written catalogue as coverage: only the supplied actual
    emitted operators are counted.  ``bound`` requires an established or
    source-declared entry/symbol/kernel, not merely a returned record.
    """
    ops = tuple(ops)
    grouped = {}
    for index, op in enumerate(ops):
        binding = (bindings or {}).get(index) if bindings else None
        if bindings is None:
            binding = bind_native_call(op)
        key = (op.family, op.source_boundary or "analytical", op.recipe or "-")
        row = grouped.setdefault(key, {
            "family": key[0], "source_boundary": key[1], "recipe": key[2],
            "emitted": 0, "bound": 0, "unresolved": 0,
            "with_entry": 0, "with_source_sha256": 0, "with_kernels": 0,
            "with_core": 0, "fully_paired": 0,
            "evidence": {}, "unbound_reasons": {},
        })
        row["emitted"] += 1
        if binding is None:
            row["unresolved"] += 1
            row["unbound_reasons"]["no_binding_record"] = (
                row["unbound_reasons"].get("no_binding_record", 0) + 1)
            continue
        evidence = binding.evidence
        is_bound, reasons = _bound_state(
            evidence, (binding.source or {}).get("sha256"), binding.typed)
        row["bound" if is_bound else "unresolved"] += 1
        row["fully_paired"] += int(
            is_bound and "installed_selected_binary_not_established"
            not in binding.unresolved)
        for reason in reasons:
            row["unbound_reasons"][reason] = row["unbound_reasons"].get(reason, 0) + 1
        row["with_entry"] += int(bool(binding.api.get("entry")))
        row["with_source_sha256"] += int(bool((binding.source or {}).get("sha256")))
        row["with_kernels"] += int(bool(binding.kernels))
        row["with_core"] += int(binding.execution_core is not None)
        for key_name, value in evidence.items():
            counter = row["evidence"].setdefault(key_name, {})
            counter[value] = counter.get(value, 0) + 1
    families = {}
    for row in grouped.values():
        family = families.setdefault(row["family"], {
            "family": row["family"], "emitted": 0, "bound": 0, "unresolved": 0})
        for key in ("emitted", "bound", "unresolved"):
            family[key] += row[key]
    return {
        "schema": "mindie-sim/native-call-coverage/1",
        "denominator": {"emitted_operators": len(ops),
                        "distinct_source_families": len(grouped),
                        "distinct_families": len(families)},
        "totals": {
            "bound": sum(row["bound"] for row in grouped.values()),
            "unresolved": sum(row["unresolved"] for row in grouped.values()),
            "with_entry": sum(row["with_entry"] for row in grouped.values()),
            "with_source_sha256": sum(row["with_source_sha256"] for row in grouped.values()),
            "with_kernels": sum(row["with_kernels"] for row in grouped.values()),
            "with_core": sum(row["with_core"] for row in grouped.values()),
            "fully_paired": sum(row["fully_paired"] for row in grouped.values()),
        },
        "by_family": [families[key] for key in sorted(families)],
        "by_source_family": [grouped[key] for key in sorted(grouped)],
        "unresolved_source_families": sorted(
            key for key, row in grouped.items() if row["unresolved"]),
        "non_claims": ["actual_normal_emission_only", "no_catalogue_counted_as_coverage"],
    }


def join_profiler_rows(rows, bindings):
    """Join captured kernel rows to bindings by declared entry/kernel name.

    ``rows`` are mapping rows (at least ``name``/``op_type``).  The result is a
    pure descriptive join: a name match is correspondence evidence, not proof
    that the measured occurrence is the declared node.
    """
    kernels, entries = {}, {}
    for binding in bindings:
        if binding is None:
            continue
        if binding.api.get("entry"):
            entries.setdefault(binding.api["entry"], []).append(binding)
        for kernel in binding.kernels:
            kernels.setdefault(kernel["name"], []).append(binding)
    joined, unmatched = [], []
    for row in rows:
        name = str(row.get("name") or row.get("op_type") or "")
        candidates = kernels.get(name) or entries.get(name) or []
        record = {
            "row_name": name, "call_ids": sorted({c.call_id for c in candidates}),
            "basis": ("declared_child_kernel" if kernels.get(name)
                      else "declared_entry" if entries.get(name) else "unmapped"),
            "tensor_match_implied": False,
        }
        (joined if candidates else unmatched).append(record)
    return {
        "schema": "mindie-sim/native-call-profiler-join/1",
        "rows": len(rows), "joined": len(joined), "unmatched": len(unmatched),
        "records": joined, "unmatched_rows": unmatched,
        "non_claims": ["correspondence_not_occurrence_proof"],
    }


def member_physical_view(binding, plan, prepared=None):
    """Same call identity, carrying one member's existing plan work as a view.

    The view is explicitly not a second charge: fused members share the parent
    call and only ``timing_owner`` owns its duration.
    """
    work = getattr(plan, "work", None)
    if work is None:
        return binding
    physical = {
        "basis": "member_view_of_shared_call_work",
        "matrix": _jsonable(getattr(work, "matrix", {})),
        "resources": _jsonable(getattr(work, "resources", {})),
        "charged": "never_separately; shared call owner",
    }
    if prepared is not None and getattr(prepared, "flops", None) is not None:
        physical["prepared_flops"] = prepared.flops
    return replace(binding, physical=physical)


def coverage_from_records(records, *, emitted=None):
    """Coverage of actual exported nodes; unrecorded emissions stay in the denominator.

    ``records`` are the BIND1 records present on the emitted structure.
    ``emitted`` is the actual emitted node/rank/operator count; when it exceeds
    ``len(records)`` the difference is explicitly unresolved rather than
    silently shrinking the denominator.
    """
    records = [record for record in records if isinstance(record, dict)]
    grouped = {}
    for record in records:
        key = (record.get("family"), record.get("source_boundary") or "analytical",
               record.get("recipe") or "-")
        row = grouped.setdefault(key, {
            "family": key[0], "source_boundary": key[1], "recipe": key[2],
            "emitted": 0, "bound": 0, "unresolved": 0,
            "with_entry": 0, "with_source_sha256": 0, "with_kernels": 0,
            "with_core": 0, "fully_paired": 0, "unbound_reasons": {}})
        row["emitted"] += 1
        evidence = record.get("evidence") or {}
        is_bound, reasons = _bound_state(
            evidence, (record.get("source") or {}).get("sha256"), record.get("typed"))
        row["bound" if is_bound else "unresolved"] += 1
        row["fully_paired"] += int(
            is_bound and "installed_selected_binary_not_established"
            not in (record.get("unresolved") or ()))
        for reason in reasons:
            row["unbound_reasons"][reason] = row["unbound_reasons"].get(reason, 0) + 1
        api = record.get("api") or {}
        row["with_entry"] += int(bool(api.get("entry")))
        row["with_source_sha256"] += int(bool((record.get("source") or {}).get("sha256")))
        row["with_kernels"] += int(bool(record.get("kernels")))
        row["with_core"] += int(record.get("execution_core") is not None)
    denominator = len(records) if emitted is None else int(emitted)
    if denominator < len(records):
        raise ModelingError("coverage_emitted_below_recorded")
    unrecorded = denominator - len(records)
    bound = sum(row["bound"] for row in grouped.values())
    unresolved_families = [key for key, row in grouped.items() if row["unresolved"]]
    if unrecorded:
        unresolved_families.append(("__unrecorded__", "none", "no_binding_record"))
    return {
        "schema": "mindie-sim/native-call-coverage/1",
        "denominator": {"emitted_operators": denominator,
                        "recorded_bindings": len(records),
                        "unrecorded_emitted": unrecorded,
                        "distinct_source_families": len(grouped)},
        "totals": {
            "bound": bound,
            "unresolved": denominator - bound,
            "with_entry": sum(row["with_entry"] for row in grouped.values()),
            "with_source_sha256": sum(row["with_source_sha256"] for row in grouped.values()),
            "with_kernels": sum(row["with_kernels"] for row in grouped.values()),
            "with_core": sum(row["with_core"] for row in grouped.values()),
            "fully_paired": sum(row["fully_paired"] for row in grouped.values()),
        },
        "by_source_family": [grouped[key] for key in sorted(grouped)],
        "unresolved_source_families": sorted(
            unresolved_families, key=lambda key: tuple(str(value) for value in key)),
        "non_claims": ["actual_exported_nodes_only", "no_catalogue_counted_as_coverage"],
    }
