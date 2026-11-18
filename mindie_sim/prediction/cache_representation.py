"""Analytic cache conversion producers with explicit packed output formats.

[Role] Make cache allocation, updates and attention consume the same row ABI,
  and bind declared store/load calls to one resolved logical/GM/on-chip/HBM
  access. Undeclared legacy calls retain their previous byte semantics.
[Invariants] BF16 input bytes are not relabelled as packed output bytes. A
  declared on-chip source read is not HBM traffic; an unresolved source is.
  Retained write lines defer exactly their dirty prefix. Written lines beyond
  the retained prefix materialize now. Dirty eviction writeback is included in
  both executable projections (program and OpSpec), never dropped as metadata.
[Timeouts & Locks] Constant/row-order arithmetic, no I/O or locks.
[Side-Effect Policy] Adds immutable graph operators to a local builder and
  commits the caller-supplied CacheState in graph order.
"""
from dataclasses import replace
from math import ceil
from collections import deque
import json

from mindie_sim.catalog.cache import cache_row_split, packed_region_row_bytes
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import cache_store_op
from mindie_sim.prediction.cache_execution import (
    CacheAccessOperation, CacheAccessRequest, CacheExecution, CacheTypedView,
)
from mindie_sim.prediction.cache_path import (
    CacheAccessGeometry, CacheGeometry, CachePolicy, CacheState,
)
from mindie_sim.prediction.engine import Event
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

SOURCE_RESIDENCY = frozenset({"unresolved", "on_chip"})


def cache_identity(region, *, layer=None, request=None, block=None, rank=None,
                   storage=None):
    """Build an unambiguous data identity for a reusable cache object.

    Every supplied reuse axis is encoded with JSON, so requests ``a`` and ``b``
    never collide with one request named ``a,b``. Callers that cannot supply a
    storage/block/rank identity should omit ``identity`` and take the cold path.
    """
    if not isinstance(region, str) or not region:
        raise ModelingError("cache_identity_region_must_be_nonempty_string")
    fields = {"region": region}
    for name, value in (("layer", layer), ("request", request), ("block", block),
                        ("rank", rank), ("storage", storage)):
        if value is not None:
            fields[name] = value
    if len(fields) == 1:
        raise ModelingError("cache_identity_requires_a_reuse_axis")
    try:
        return json.dumps(fields, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"cache_identity_is_not_json_serializable:{exc}") from exc


def _effective_geometry(geometry, state):
    if state is not None and not isinstance(state, CacheState):
        raise ModelingError("cache_state_must_be_cache_state")
    if geometry is not None:
        if not isinstance(geometry, CacheGeometry):
            raise ModelingError("cache_geometry_must_be_cache_geometry")
        if state is not None and state.geometry != geometry:
            raise ModelingError("cache_state_geometry_conflicts_with_declared_geometry")
        return geometry
    return None if state is None else state.geometry


def _resolve_state(builder, state, geometry):
    if state is not None:
        return state, "explicit_state"
    attached = getattr(builder, "cache_state", None)
    if attached is not None:
        if not isinstance(attached, CacheState):
            raise ModelingError("builder_cache_state_must_be_cache_state")
        return attached, "builder_cache_state"
    return CacheState(geometry), "ephemeral_cold"


def _identity(identity, region):
    if identity is None:
        # No reusable storage/block identity was supplied. A unique cold key
        # would still fill shared state, so this path must never look up, fill
        # or evict. Existing callers keep their previous cold byte semantics.
        return f"<cold:{region}>", True
    if not isinstance(identity, str) or not identity:
        raise ModelingError("cache_access_identity_must_be_nonempty_string")
    return identity, False


def _access_metadata(result, resolution, fmt, row_bytes, geometry, region, *,
                     source_bytes=None, source_residency=None, write_policy=None,
                     execution_gap=False):
    metadata = [
        ("cache_access", result.to_dict()),
        ("cache_resolution", resolution),
        ("cache_geometry", {
            "capacity_bytes": geometry.capacity_bytes,
            "line_bytes": geometry.line_bytes,
            "transfer_bytes": geometry.transfer_bytes,
            "staging_lines": geometry.staging_lines,
        }),
        ("cache_row_bytes", row_bytes),
        ("cache_format", {
            "region": region, "dtype": fmt.dtype,
            "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
            "suffix_elements": fmt.suffix_elements,
            "suffix_dtype": fmt.suffix_dtype,
            "packed_elements": fmt.packed_elements,
        }),
        ("cache_traffic_basis",
         "logical_row_bytes_then_GM_granule_then_next_level_lines"),
        # Both executable projections carry the same ledger. The write
        # program/OpSpec consumes hbm_write_bytes; state flush consumes the
        # dirty prefix. direct + writeback equals hbm_write by construction.
        ("cache_hbm_write_ledger", {
            "direct_write_bytes": result.direct_write_bytes,
            "writeback_bytes": result.writeback_bytes,
            "hbm_write_bytes": result.hbm_write_bytes,
        }),
    ]
    if source_bytes is not None:
        metadata.append(("cache_source_read_bytes", source_bytes))
    if source_residency is not None:
        metadata.append(("cache_source_residency", source_residency))
    if write_policy is not None:
        metadata.append(("cache_write_policy", write_policy))
    if result.cold_path:
        metadata.append(("cache_cold_path", "unidentified_storage"))
    if execution_gap:
        metadata.append((
            "cache_execution_gap",
            "static_graph_records_work_but_requires_CacheLowering.execute_"
            "or_CacheAccessProgram_for_ready_staging_allocation"))
    return tuple(metadata)


def _legacy_store(builder, name, deps, rows, width, region, fmt):
    """Preserve the pre-cache-level byte semantics of an undeclared call."""
    common = builder.common
    row_bytes = packed_region_row_bytes(builder.model, region, width)
    stored = rows * row_bytes
    legacy = (
        ("cache_resolution", "legacy_no_cache_declaration"),
        ("cache_legacy_semantics", True),
        ("cache_traffic_basis", "legacy_logical_source_plus_stored_bytes"),
        ("cache_format", {
            "region": region, "dtype": fmt.dtype,
            "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
            "suffix_elements": fmt.suffix_elements,
            "suffix_dtype": fmt.suffix_dtype,
            "packed_elements": fmt.packed_elements,
        }),
    )
    if not fmt.group_size:
        # Exactly the former builder.store()/cache_store_op ABI.
        op = cache_store_op(
            op_id=builder.prefix + "." + name, deps=tuple(deps),
            stage=common["stage"], layer=common["layer"], requests=common["requests"],
            phase=common["phase"], recipe="kv_store" if name == "cache" else name, rows=rows, row_bytes=row_bytes,
            dtype=fmt.dtype, kind=name, metadata=legacy,
        )
        builder.add(op)
        return op.op_id
    source = rows * width * dtype_bytes(builder.model.activation_dtype)
    groups = ceil(width / fmt.group_size)
    op = OpSpec(
        op_id=builder.prefix + "." + name, family="cache_store",
        phase=common["phase"], stage=common["stage"], layer=common["layer"],
        requests=common["requests"], semantic_shape=(rows, width),
        local_shape=(rows, width), padded_shape=(rows, width),
        launch_shape=(rows, width), vector_ops=float(rows * (4 * width + groups)),
        hbm_bytes=float(source + stored), mte2_bytes=float(source),
        mte3_bytes=float(stored), deps=tuple(deps), recipe="kv_store" if name == "cache" else name,
        dtype=builder.model.activation_dtype,
        metadata=legacy + (
            ("cache_kind", name), ("cache_visible", True), ("shared_owner", False),
            ("cache_representation", {
                "region": region, "dtype": fmt.dtype,
                "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
                "row_bytes": row_bytes,
                "producer": "analytic_group_quantization_and_cache_write",
                "source_kernel_identity": "not_established",
            }),
        ),
    )
    builder.add(op)
    return op.op_id


def _store_cache_aware(builder, name, deps, rows, width, region, fmt, *,
                       state, geometry, identity, policy, source_residency):
    common = builder.common
    row_bytes = packed_region_row_bytes(builder.model, region, width)
    stored = rows * row_bytes
    source = rows * width * dtype_bytes(builder.model.activation_dtype)
    groups = ceil(width / fmt.group_size) if fmt.group_size else 0
    resolved_state, resolution = _resolve_state(builder, state, geometry)
    access_identity, cold = _identity(identity, region)
    access_policy = replace(policy, cold=cold)
    result = resolved_state.access(access_identity, stored, write=True,
                                   policy=access_policy)
    if result.cold_path and policy.write_policy == "write_back":
        effective_write_policy = "cold_materialize"
    else:
        effective_write_policy = policy.write_policy
    if fmt.group_size:
        shape = (rows, width)
        padded_shape = shape
        operations = rows * (4 * width + groups)
        dtype = builder.model.activation_dtype
        mte2_bytes = source
    else:
        # Preserve the plain cache_store row ABI.
        shape = (max(1, rows), max(1, row_bytes))
        padded_rows = (max(1, rows) + 7) // 8 * 8
        padded_shape = (padded_rows, shape[1])
        operations = rows * max(1, row_bytes)
        dtype = fmt.dtype
        mte2_bytes = rows * row_bytes
    # An unresolved source remains external traffic. A declared on-chip source
    # removes only that read from HBM, never the cache output write.
    hbm_bytes = result.hbm_write_bytes
    source_scope = "on_chip_not_HBM"
    if source_residency != "on_chip":
        hbm_bytes += source
        source_scope = "unresolved_source_read_counted_in_HBM"
    metadata = _access_metadata(
        result, resolution, fmt, row_bytes, resolved_state.geometry, region,
        source_bytes=source, source_residency=source_residency,
        write_policy=effective_write_policy,
        execution_gap=bool(result.staging_bytes or policy.ready_ms),
    )
    op = OpSpec(
        op_id=builder.prefix + "." + name, family="cache_store",
        phase=common["phase"], stage=common["stage"], layer=common["layer"],
        requests=common["requests"], semantic_shape=shape, local_shape=shape,
        padded_shape=padded_shape, launch_shape=shape,
        vector_ops=float(operations), hbm_bytes=float(hbm_bytes),
        mte2_bytes=float(mte2_bytes), mte3_bytes=float(result.gm_bytes),
        deps=tuple(deps), recipe="kv_store" if name == "cache" else name, dtype=dtype,
        metadata=metadata + (
            ("cache_kind", name), ("cache_visible", True), ("shared_owner", False),
            ("cache_representation", {
                "region": region, "dtype": fmt.dtype,
                "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
                "row_bytes": row_bytes,
                "producer": "analytic_group_quantization_and_cache_write",
                "source_read_scope": source_scope,
                "source_kernel_identity": "not_established",
            }),
        ),
    )
    builder.add(op)
    return op.op_id


def _load_cache_aware(builder, name, deps, rows, width, region, fmt, *,
                      state, geometry, identity, policy):
    common = builder.common
    row_bytes = packed_region_row_bytes(builder.model, region, width)
    stored = rows * row_bytes
    resolved_state, resolution = _resolve_state(builder, state, geometry)
    access_identity, cold = _identity(identity, region)
    result = resolved_state.access(access_identity, stored, write=False,
                                   policy=replace(policy, cold=cold))
    metadata = _access_metadata(
        result, resolution, fmt, row_bytes, resolved_state.geometry, region,
        execution_gap=bool(result.staging_bytes or policy.ready_ms),
    )
    op = OpSpec(
        op_id=builder.prefix + "." + name, family="cache_load",
        phase=common["phase"], stage=common["stage"], layer=common["layer"],
        requests=common["requests"], semantic_shape=(rows, width),
        local_shape=(rows, width), padded_shape=(rows, width),
        launch_shape=(rows, width), flops=0.0, vector_ops=0.0,
        # A read that evicts a dirty line performs a real HBM write. Both
        # directions are part of this operator's traffic.
        hbm_bytes=float(result.hbm_read_bytes + result.hbm_write_bytes),
        mte2_bytes=float(result.gm_bytes), mte3_bytes=0.0,
        deps=tuple(deps), recipe=name, dtype=fmt.dtype,
        metadata=metadata + (
            ("cache_kind", name), ("cache_visible", True),
            ("shared_owner", False),
            ("cache_read", "declared_representation_row"),
        ),
    )
    builder.add(op)
    return op.op_id


def store(b, name, deps, rows, width, region, *, state=None, geometry=None,
          identity=None, write_policy="write_through", ready_ms=0.0, bypass=False,
          source_residency="unresolved"):
    """Write ``rows`` packed cache rows.

    Without a declared geometry/state this retains the former byte semantics.
    A declared path resolves cache lines and write policy. ``source_residency``
    must be ``on_chip`` to exclude the producing dependency's read from HBM;
    otherwise that read remains external. ``write_back`` requires a persistent
    state and retains only the dirty prefix; non-retained lines materialize now.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ModelingError("cache_rows_must_be_nonnegative_integer")
    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ModelingError("cache_row_width_must_be_nonnegative_integer")
    if source_residency not in SOURCE_RESIDENCY:
        raise ModelingError("cache_source_residency_must_be_unresolved_or_on_chip")
    if write_policy not in {"write_through", "write_back"}:
        raise ModelingError("cache_write_policy_must_be_write_through_or_write_back")
    if type(bypass) is not bool:
        raise ModelingError("cache_bypass_must_be_boolean")
    if (not isinstance(ready_ms, (int, float)) or isinstance(ready_ms, bool)
            or not ready_ms >= 0):
        raise ModelingError("cache_ready_ms_must_be_nonnegative")
    fmt = cache_format(b.model, region)
    attached = getattr(b, "cache_state", None)
    if attached is not None and not isinstance(attached, CacheState):
        raise ModelingError("builder_cache_state_must_be_cache_state")
    resolved_state = state if state is not None else attached
    effective_geometry = _effective_geometry(geometry, resolved_state)
    if effective_geometry is None:
        if bypass or ready_ms:
            raise ModelingError("cache_geometry_required_for_direct_or_ready_path")
        if identity is not None:
            raise ModelingError("cache_identity_requires_declared_geometry")
        if write_policy == "write_back":
            raise ModelingError("cache_write_back_requires_persistent_state")
        if source_residency == "on_chip":
            raise ModelingError("cache_source_residency_requires_declared_geometry")
        return _legacy_store(b, name, deps, rows, width, region, fmt)
    if write_policy == "write_back" and resolved_state is None:
        raise ModelingError("cache_write_back_requires_persistent_state")
    policy = CachePolicy(bypass=bypass, write_policy=write_policy, ready_ms=ready_ms)
    return _store_cache_aware(
        b, name, deps, rows, width, region, fmt, state=state,
        geometry=effective_geometry, identity=identity, policy=policy,
        source_residency=source_residency,
    )


def load(b, name, deps, rows, width, region, *, state=None, geometry=None,
         identity=None, bypass=False, ready_ms=0.0):
    """Read ``rows`` packed cache rows against a declared cache level.

    A hit removes only the next-level line fill. GM copy service, staging
    occupancy and the declared ready delay remain. A miss that evicts a dirty
    line carries the writeback in ``hbm_bytes`` as well as in ``cache_access``.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ModelingError("cache_rows_must_be_nonnegative_integer")
    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ModelingError("cache_row_width_must_be_nonnegative_integer")
    if type(bypass) is not bool:
        raise ModelingError("cache_bypass_must_be_boolean")
    if (not isinstance(ready_ms, (int, float)) or isinstance(ready_ms, bool)
            or not ready_ms >= 0):
        raise ModelingError("cache_ready_ms_must_be_nonnegative")
    fmt = cache_format(b.model, region)
    attached = getattr(b, "cache_state", None)
    if attached is not None and not isinstance(attached, CacheState):
        raise ModelingError("builder_cache_state_must_be_cache_state")
    resolved_state = state if state is not None else attached
    effective_geometry = _effective_geometry(geometry, resolved_state)
    if effective_geometry is None:
        raise ModelingError("cache_load_requires_declared_geometry")
    policy = CachePolicy(bypass=bypass, write_policy="write_through",
                         ready_ms=ready_ms)
    return _load_cache_aware(
        b, name, deps, rows, width, region, fmt, state=state,
        geometry=effective_geometry, identity=identity, policy=policy,
    )


def _op_output(op):
    outputs = getattr(op, "outputs", None) or ()
    if not outputs:
        return ""
    first = outputs[0]
    return first if isinstance(first, str) else str(first)


def _typed_view(model, op, fmt, rows, width, row_bytes, region, identity, deps=()):
    """Actual producer/output/view of one declared cache row operation.

    OpSpec currently has no output namespace. The producer output name is kept
    as a declaration so declaration projections stay meaningful, but it is not
    registered as an actual dependency port unless the producer really exposes
    that output.
    """
    stored = int(rows * row_bytes)
    _payload_one, scale_one = cache_row_split(model, region, width)
    scale_bytes = int(round(rows * scale_one))
    payload_bytes = stored - scale_bytes
    packed = fmt.packed_elements != 1 or bool(fmt.group_size)
    outputs = tuple(getattr(op, "outputs", None) or ())
    registered = bool(outputs)
    output = str(outputs[0]) if registered else op.op_id + ".output"
    return CacheTypedView(
        producer=op.op_id, output=output, dtype=fmt.dtype,
        layout="packed_row" if packed else "contiguous",
        shape=(rows, width), storage=region, owner=identity,
        inputs=tuple(deps), payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        packed_elements=fmt.packed_elements, group_size=fmt.group_size,
        output_registered=registered)


class CacheLowering:
    """Builder-facing cache program with event-time execution.

    ``store``/``load`` emit static cold-declared graph nodes and record
    immutable ``CacheAccessRequest`` objects. The graph nodes never cache the
    previous iteration's hit/miss result. ``execute`` runs the recorded
    requests on the shared Engine against the live ``CacheState``; the same
    Lowering can run across iterations because requests are immutable.
    """

    def __init__(self, builder, state, *, hbm_resource="hbm",
                 onchip_resource="mte2", pool="ub", vector_resource="vector",
                 write_resource="mte3"):
        if not isinstance(state, CacheState):
            raise ModelingError("cache_lowering_requires_cache_state")
        self.builder = builder
        self.state = state
        self.hbm_resource = hbm_resource
        self.onchip_resource = onchip_resource
        self.pool = pool
        self.vector_resource = vector_resource
        self.write_resource = write_resource
        self.requests = []
        # One explicit alias table: source request name, producer op id, and a
        # real producer output (when registered). A merely declared output is
        # tracked separately and never becomes a dependency port.
        self._bindings = {}
        self._declared_outputs = {}
        self._executions = deque(maxlen=64)
        self._name_by_request = {}

    def _register(self, name, op, request):
        aliases = [name, op.op_id]
        view = request.view
        if view is not None and view.output_registered:
            aliases.append(view.output)
        aliases = list(dict.fromkeys(aliases))
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                raise ModelingError("cache_lowering_port_name_invalid")
            if alias in self._bindings or alias in self._declared_outputs:
                raise ModelingError(f"cache_lowering_port_duplicate:{alias}")
        declared = None
        if view is not None and not view.output_registered:
            declared = view.output
            if declared in self._bindings or declared in self._declared_outputs:
                raise ModelingError(
                    f"cache_lowering_port_duplicate:{declared}")
        for alias in aliases:
            self._bindings[alias] = request
        if declared is not None:
            self._declared_outputs[declared] = request

    def _mark_static(self, write_policy=None, outstanding=0, response_ms=0.0,
                     geometry=None, write=False, view=None, request=None):
        op = self.builder.ops[-1]
        metadata = dict(op.metadata)
        if geometry is not None:
            metadata["cache_transaction_summary"] = geometry.summary(
                self.state.geometry.line_bytes, write=write,
                transfer_bytes=self.state.geometry.transfer_bytes).to_dict()
        metadata["cache_static_basis"] = "cold_declared_request"
        metadata["cache_dynamic_state_resolved_at_execution"] = True
        metadata["cache_program_entry"] = "CacheLowering.execute"
        if write_policy is not None:
            metadata["cache_dynamic_write_policy"] = write_policy
        if outstanding:
            metadata["cache_request_outstanding"] = outstanding
        if response_ms:
            metadata["cache_request_response_ms"] = response_ms
        if view is not None:
            metadata["cache_typed_view"] = view.to_dict()
            metadata["cache_typed_output"] = view.output
        if request is not None:
            metadata["cache_access_program"] = {
                "identity": request.identity,
                "write": bool(request.write),
                "producer": request.producer,
                "output": request.output,
                "deps": list(request.deps),
                "source_bytes": request.source_bytes,
                "convert_ops": request.convert_ops,
                "source_residency": request.source_residency,
                "source_port": request.source_port,
                "geometry": {
                    "offset_bytes": request.geometry.offset_bytes,
                    "rows": request.geometry.rows,
                    "row_bytes": request.geometry.row_bytes,
                    "stride_bytes": request.geometry.stride_bytes,
                    "page_bytes": request.geometry.page_bytes,
                    "packing": request.geometry.packing,
                    "partial_line": request.geometry.partial_line,
                },
            }
        self.builder.ops[-1] = replace(op, metadata=tuple(sorted(metadata.items())))

    def store(self, name, deps, rows, width, region, *, identity,
              write_policy="write_through", ready_ms=0.0, bypass=False,
              source_residency="unresolved", outstanding=0, response_ms=0.0,
              offset_bytes=0, stride_bytes=None, page_bytes=0, packing=1,
              partial_line="masked", source_port=""):
        fmt = cache_format(self.builder.model, region)
        row_bytes = packed_region_row_bytes(self.builder.model, region, width)
        static = CacheState(self.state.geometry)
        store(self.builder, name, deps, rows, width, region, state=static,
              identity=identity, write_policy="write_through",
              ready_ms=ready_ms, bypass=bypass,
              source_residency=source_residency)
        geometry = CacheAccessGeometry(
            offset_bytes=offset_bytes, rows=rows, row_bytes=row_bytes,
            stride_bytes=row_bytes if stride_bytes is None else stride_bytes,
            page_bytes=page_bytes, packing=packing, partial_line=partial_line)
        op = self.builder.ops[-1]
        view = _typed_view(self.builder.model, op, fmt, rows, width, row_bytes,
                           region, identity, deps=deps)
        # The declared source view is activation-width bytes recorded by the
        # cache-aware store. It is not the packed output row width, so a plain
        # cache_store op's mte2 byte count cannot stand in for it.
        source_bytes = dict(op.metadata).get("cache_source_read_bytes")
        if source_bytes is None:
            source_bytes = int(op.mte2_bytes)
        request = CacheAccessRequest(
            identity, geometry, write=True,
            policy=CachePolicy(bypass=bypass, write_policy=write_policy,
                               ready_ms=ready_ms),
            outstanding=outstanding, response_ms=response_ms,
            source_bytes=int(source_bytes), convert_ops=float(op.vector_ops),
            source_residency=source_residency, deps=tuple(deps),
            producer=op.op_id, output=view.output,
            output_dtype=fmt.dtype, output_layout=view.layout, view=view,
            source_port=source_port)
        self._register(name, op, request)
        self.requests.append((name, request))
        self._mark_static(write_policy, outstanding, response_ms,
                          geometry=geometry, write=True, view=view,
                          request=request)
        return op.op_id

    def load(self, name, deps, rows, width, region, *, identity,
             ready_ms=0.0, bypass=False, outstanding=0, response_ms=0.0,
             offset_bytes=0, stride_bytes=None, page_bytes=0, packing=1,
             partial_line="masked", source_port=""):
        fmt = cache_format(self.builder.model, region)
        row_bytes = packed_region_row_bytes(self.builder.model, region, width)
        static = CacheState(self.state.geometry)
        load(self.builder, name, deps, rows, width, region, state=static,
             identity=identity, bypass=bypass, ready_ms=ready_ms)
        geometry = CacheAccessGeometry(
            offset_bytes=offset_bytes, rows=rows, row_bytes=row_bytes,
            stride_bytes=row_bytes if stride_bytes is None else stride_bytes,
            page_bytes=page_bytes, packing=packing, partial_line=partial_line)
        op = self.builder.ops[-1]
        view = _typed_view(self.builder.model, op, fmt, rows, width, row_bytes,
                           region, identity, deps=deps)
        request = CacheAccessRequest(
            identity, geometry, write=False,
            policy=CachePolicy(bypass=bypass, ready_ms=ready_ms),
            outstanding=outstanding, response_ms=response_ms,
            deps=tuple(deps), producer=op.op_id,
            output=view.output, output_dtype=fmt.dtype,
            output_layout=view.layout, view=view,
            source_port=source_port)
        self._register(name, op, request)
        self.requests.append((name, request))
        self._mark_static(None, outstanding, response_ms,
                          geometry=geometry, write=False, view=view,
                          request=request)
        return op.op_id

    def execute(self, engine, *, ready=None, ready_by_name=None, owner=""):
        """Preflight the complete declared port graph before any submission.

        No Engine task, promise or state mutation exists when a later node has
        an unbound/ambiguous port, an unregistered output alias, a missing
        source or a dependency-order cycle.
        """
        named = {} if ready_by_name is None else dict(ready_by_name)
        if ready is not None and not isinstance(ready, Event):
            raise ModelingError("cache_lowering_ready_event_invalid:generic")
        for key, value in named.items():
            if not isinstance(key, str) or not key:
                raise ModelingError("cache_lowering_ready_name_invalid")
            if not isinstance(value, Event):
                raise ModelingError(f"cache_lowering_ready_event_invalid:{key}")
        self._name_by_request = {id(request): name
                                 for name, request in self.requests}
        order = {id(request): index
                 for index, (_name, request) in enumerate(self.requests)}
        plan = []
        for index, (name, request) in enumerate(self.requests):
            internal = []
            external = []
            dep_bindings = []
            for dep in request.deps:
                bound = self._bindings.get(dep)
                if bound is not None:
                    bound_name = self._name_by_request.get(id(bound))
                    if bound_name is None:
                        raise ModelingError(
                            f"cache_lowering_port_unknown:{dep}->{name}")
                    if order[id(bound)] >= index:
                        raise ModelingError(
                            f"cache_lowering_dependency_order:{dep}->{name}")
                    internal.append(bound_name)
                    dep_bindings.append(("internal", dep, bound_name))
                elif dep in self._declared_outputs:
                    if dep in named:
                        external.append(named[dep])
                        dep_bindings.append(("external", dep, named[dep]))
                    else:
                        raise ModelingError(
                            f"cache_lowering_port_unregistered:{dep}->{name}")
                elif dep in named:
                    external.append(named[dep])
                    dep_bindings.append(("external", dep, named[dep]))
                else:
                    # A generic ready is the call boundary, never a substitute
                    # for a named producer/output port.
                    raise ModelingError(
                        f"cache_lowering_port_unbound:{dep}->{name}")
            source_event = None
            source_internal = None
            source_role = None
            if request.source_port:
                port = request.source_port
                if self._bindings.get(port) is not None:
                    raise ModelingError(
                        f"cache_lowering_port_ambiguous:{port}->{name}")
                if port in self._declared_outputs:
                    raise ModelingError(
                        f"cache_lowering_port_unregistered:{port}->{name}")
                if port not in named:
                    raise ModelingError(
                        f"cache_lowering_port_unbound:{port}->{name}")
                source_event = named[port]
                source_role = port
                external.append(source_event)
                dep_bindings.append(("external", port, source_event))
            elif len(request.deps) == 1:
                # A single declared dependency is the actual data producer.
                # Multiple deps remain generic gates; no source is guessed.
                dep = request.deps[0]
                bound = self._bindings.get(dep)
                if bound is not None:
                    source_internal = self._name_by_request[id(bound)]
                    source_role = dep
                elif dep in named:
                    source_event = named[dep]
                    source_role = dep
            ready_events = []
            for key in dict.fromkeys((name, request.producer)):
                if key and key in named:
                    ready_events.append(named[key])
            if ready is not None:
                ready_events.append(ready)
            plan.append({
                "name": name, "request": request, "internal": tuple(internal),
                "external": tuple(external), "ready": tuple(ready_events),
                "source": source_event, "source_internal": source_internal,
                "source_role": source_role,
                "dep_bindings": tuple(dep_bindings),
            })

        # Complete read-only resource/unit/pool preflight for the whole launch
        # graph. A later node's missing required resource must leave the Engine,
        # cache state and promises untouched.
        execution = CacheExecution(
            self.state, self.hbm_resource, self.onchip_resource, self.pool,
            self.vector_resource, self.write_resource)
        for item in plan:
            preflight = CacheAccessOperation(
                self.state, item["request"], self.hbm_resource,
                self.onchip_resource, self.pool, self.write_resource,
                self.vector_resource)
            preflight.validate(engine)
        completed = {}
        ordered = []
        for item in plan:
            name = item["name"]
            request = item["request"]
            predecessors = [completed[dep] for dep in item["internal"]]
            predecessors.extend(item["external"])
            predecessors.extend(item["ready"])
            predecessors = list({event.sequence: event
                                 for event in predecessors}.values())
            if predecessors:
                stage_ready = (predecessors[0] if len(predecessors) == 1 else
                               engine.submit(name + ":deps",
                                             after=tuple(predecessors),
                                             owner=owner))
            else:
                stage_ready = engine.submit(name + ":start", release=0,
                                            owner=owner)
            dep_events = {}
            for kind, dep, value in item["dep_bindings"]:
                dep_events[dep] = (completed[value] if kind == "internal"
                                   else value)
            source_event = item["source"]
            if item["source_internal"] is not None:
                source_event = completed[item["source_internal"]]
            event = execution.access(
                request, engine, name, stage_ready, owner=owner,
                source=source_event, deps=dep_events,
                ports_verified=True)
            completed[name] = event
            completed[request.producer] = event
            ordered.append(event)
        self.execution = execution
        self._executions.append(execution)
        return tuple(ordered)

    def state_access(self, key, *, storage=None, occurrence=None, **kwargs):
        """Resolve one runtime occurrence receipt, including earlier launches."""
        request = self._bindings.get(key)
        if request is None:
            if key in self._declared_outputs:
                raise ModelingError(
                    f"cache_lowering_declared_output_not_actual:{key}")
            raise ModelingError(f"cache_lowering_unknown_request:{key}")
        if not self._executions:
            raise ModelingError("cache_lowering_not_executed")
        name = self._name_by_request.get(id(request))
        if occurrence is None:
            occurrence = name
        last_error = None
        for execution in reversed(self._executions):
            try:
                return execution.state_access(
                    request, occurrence=occurrence, storage=storage, **kwargs)
            except ModelingError as exc:
                if str(exc).startswith("cache_access_occurrence_unknown"):
                    last_error = exc
                    continue
                raise
        raise last_error or ModelingError(
            f"cache_lowering_occurrence_unknown:{occurrence}")

    def flush(self, engine, name, ready, *, owner=""):
        execution = CacheExecution(
            self.state, self.hbm_resource, self.onchip_resource, self.pool,
            self.vector_resource, self.write_resource)
        return execution.flush(engine, name, ready, owner=owner)


__all__ = ["CacheLowering", "cache_identity", "load", "store"]
