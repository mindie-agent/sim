"""Assemble the family compiler registry and hardware-declared service rules.

[Role] Combine the matrix/memory, attention, recurrent and collective family
  compilers and convert declared hardware facts into the only service
  adjustments the piecewise engine applies: an executed-precision cube peak, a
  declared kernel-invocation completion floor and declared link-latency rounds.
[Invariants] Every registered family maps to one declared compiler; a missing
  family remains an explicit ``operator_family_unsupported`` error.  Work and
  dependencies come from the family compiler's structural program.  This module
  does not scale work by token, batch, cohort, route, graph-bucket, model,
  suite or measurement lookups, and it does not move work between clocks.
  Precision promotion uses only the declared ``compute_peaks`` entry for the
  executed matrix dtype when both weight and activation are quantized.
  Invocation floors use only ``Device.kernel_invocations``; a floor is inclusive
  (``max(minimum, startup + body)``), never an additive per-family multiple.
  Collective latency uses the declared link latency for the physical ring
  rounds of the named primitive; the transfer volumes still come from the
  compiler's structural bytes.  Synthetic ``PipeSupply`` instances without
  hardware declarations remain untouched.
[Timeouts & Locks] Pure bounded arithmetic only; no I/O or locks.
[Side-Effect Policy] No I/O or global model mutation.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Mapping

from .expressions import as_fraction, evaluate
from .physical import US_TO_MS
from mindie_sim.schema.common import ModelingError
from .families_attention import (
    _compile_attention, _compile_draft, _compile_mla, _compile_vision,
)
from .families_matrix import (
    _compile_collective, _compile_copy, _compile_elementwise, _compile_embedding,
    _compile_gemm, _compile_grouped_gemm, _compile_kv_access, _compile_router,
    _compile_transport,
)
from .families_recurrent_indexer import (
    compile_compression, compile_indexer, compile_recurrent,
    compile_state_gather, compile_state_scatter,
)
from .families_moe_mc2 import compile_mc2_transport

FAMILY_COMPILERS = {"gemm": _compile_gemm, "router": _compile_router, "attention": _compile_attention, "mla": _compile_mla, "indexer": compile_indexer, "compression": compile_compression, "kv_access": _compile_kv_access, "elementwise": _compile_elementwise, "quantize": _compile_elementwise, "hc_post": _compile_elementwise, "copy": _compile_copy, "embedding": _compile_embedding, "collective": _compile_collective, "transport": _compile_transport, "mc2_transport": compile_mc2_transport, "grouped_gemm": _compile_grouped_gemm, "recurrent": compile_recurrent, "state_gather": compile_state_gather, "state_scatter": compile_state_scatter, "vision": _compile_vision, "draft": _compile_draft, }

_CUBE_FAMILIES = frozenset({
    "gemm", "grouped_gemm", "attention", "mla", "indexer",
})


def extract_hardware_meta(hardware: Any, time_scale: float) -> tuple[
    tuple[tuple[str, float], ...], tuple[tuple[str, int], ...],
    tuple[tuple[str, float], ...],
]:
    """Extract declared kernel invocations, unit counts and dtype peaks from Device."""
    die = hardware.dies[0]
    raw_inv = getattr(die, "kernel_invocations", ()) or ()
    raw_units = getattr(die, "resource_units", ()) or ()
    if not raw_inv and not raw_units:
        return (), (), ()
    inv: dict[str, float] = {}
    for k, v in raw_inv:
        ms = float(getattr(v, "minimum_ms", 0.0) or 0.0)
        start_ms = float(getattr(v, "startup_ms", 0.0) or 0.0)
        if ms >= 0.0:
            inv[str(k)] = ms
        if start_ms > 0.0:
            inv[f"{k}_startup_ms"] = start_ms
    intra_lat = inter_lat = 0.0
    intra_rate = inter_rate = 0.0
    package_lat, package_width, node_lat = 0.0, 0, 0.0
    for link in (getattr(hardware, "links", ()) or ()):
        lat = float(getattr(link, "latency_us", 0.0) or 0.0)
        rate = float(getattr(link, "capacity_bytes_per_ms", 0.0) or 0.0)
        channel = str(getattr(link, "channel", getattr(link, "name", "")) or "")
        text = f"{channel} {getattr(link, 'name', '')}".lower()
        if "inter" in channel:
            if lat > 0:
                inter_lat = lat if inter_lat <= 0 else min(inter_lat, lat)
            if rate > 0:
                inter_rate = rate if inter_rate <= 0 else min(inter_rate, rate)
        else:
            if lat > 0:
                intra_lat = lat if intra_lat <= 0 else min(intra_lat, lat)
            if rate > 0:
                intra_rate = rate if intra_rate <= 0 else min(intra_rate, rate)
            # A3 declares a two-die package SIO route and a wider on-node
            # switch route.  A serial peer schedule must not charge the
            # package latency to cross-package peers.
            if "package" in text or "sio" in text:
                if lat > 0:
                    package_lat = lat if package_lat <= 0 else min(package_lat, lat)
                package_width = max(package_width, len(getattr(link, "members", ()) or ()))
            elif lat > 0:
                node_lat = max(node_lat, lat)
    if intra_lat <= 0 and inter_lat <= 0:
        ch_lats = [float(dict(ch_kv).get("latency_us", dict(ch_kv).get("alpha_us", 0.0)) or 0.0) for _, ch_kv in (getattr(hardware, "transfer_channels", ()) or ())]
        intra_lat = min((x for x in ch_lats if x > 0), default=0.0)
    lat = min((x for x in (intra_lat, inter_lat) if x > 0), default=0.0)
    if lat > 0:
        inv["link_latency_ms"] = lat * US_TO_MS
        inv["link_latency_intra_ms"] = (intra_lat or lat) * US_TO_MS
        inv["link_latency_inter_ms"] = (inter_lat or lat) * US_TO_MS
    if package_lat > 0:
        inv["link_latency_package_ms"] = package_lat * US_TO_MS
        inv["ranks_per_package"] = package_width
    if node_lat > 0:
        inv["link_latency_node_ms"] = node_lat * US_TO_MS
    if intra_rate > 0:
        inv["link_rate_intra_bytes_per_ms"] = intra_rate
    if int(getattr(hardware, "ranks_per_node", 0) or 0) > 0:
        inv["ranks_per_node"] = int(getattr(hardware, "ranks_per_node"))
    if inter_rate > 0:
        inv["link_rate_inter_bytes_per_ms"] = inter_rate
    units: dict[str, int] = {str(k): int(v) for k, v in raw_units if int(v) > 0}
    units.setdefault("compute_units", max(1, int(getattr(die, "compute_units", 1) or 1)))
    # Per-core unified buffer.  DynamicQuant's host compares this with the
    # row.  It is not an invocation floor; only named timing keys are.
    ub = int(dict(getattr(die, "buffers", ()) or ()).get("ub", 0) or 0)
    if ub > 0:
        inv["ub_bytes"] = float(ub)
    # Dtype peaks and propagation latency are independent profile quantities;
    # they must survive a device profile that declares no invocation minimum.
    peaks: dict[str, float] = { str(k): float(v) * float(time_scale) for k, v in (getattr(die, "compute_peaks", ()) or ()) if float(v) > 0 }
    return (tuple(sorted(inv.items())), tuple(sorted(units.items())), tuple(sorted(peaks.items())))


def _canonical_format(value: Any) -> str:
    name = str(value or "").strip().lower()
    return {
        "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
        "mx_fp4": "mxfp4", "mxfp4_e2m1": "mxfp4", "fp4": "fp4_e2m1",
        "fp8": "fp8_e4m3",
    }.get(name, name)


def _declared_cube_rate(op: Any, all_work: dict[str, Fraction]) -> Fraction:
    """Declared MMAD execution rate over the supplied cube rate.

    The native ``2*M*N*K`` work is never rewritten.  A quantized matrix
    instruction executes at the declared activation/MMAD dtype peak, while
    ``PipeSupply`` may already have been built from a BF16 (or any other) base.
    The returned dimensionless value multiplies only the elapsed denominator.
    An undeclared execution format keeps the supplied base rate.
    """
    if "cube" not in all_work or op.family not in _CUBE_FAMILIES:
        return Fraction(1)
    peaks = {str(k): float(v) for k, v in (getattr(op.supply, "flops_by_dtype", ()) or ())}
    theta = op.operator.theta
    # The MMAD execution dtype is the activation/group dtype, not the fastest
    # of the two operands and not necessarily the supplied base.  An op may
    # legally execute below the base (for example BF16 activation under an
    # already-FP8 supply); a missing declaration keeps the supplied rate.
    activation = _canonical_format(theta.find("act_format", theta.find("activation_format", "bf16")))
    peak = peaks.get(activation, 0.0)
    if peak <= 0.0:
        return Fraction(1)
    base = as_fraction(op.supply.effective("cube"))
    if base <= 0:
        return Fraction(1)
    # ``peak/base`` is meaningful only when the supplied cube rate is one of
    # the declared execution peaks.  An arbitrary calibrated/synthetic base is
    # not silently rescaled by an unrelated clock ratio.
    if not any(abs(base - as_fraction(value)) <= max(Fraction(1), as_fraction(value)) / 10**9
               for value in peaks.values() if value > 0):
        return Fraction(1)
    return as_fraction(peak) / base


def _invocation_classes(op: Any, all_work: dict[str, Fraction]) -> tuple[str, ...]:
    """Class names whose declared minimum constrains this kernel once."""
    explicit = str(op.operator.theta.find("invocation_class", "") or "").strip()
    if explicit:
        return (explicit,)
    if "notify" in op.name.lower():
        return ("empty_aic",)
    classes: list[str] = []
    if "cube" in all_work:
        classes.append("cube")
    if "vector" in all_work:
        classes.append("vector")
    # Pure DMA, transport and join kernels receive no inferred vector charge.
    return tuple(classes)


def _body_ms(op: Any, env: dict[str, Any], rate_scale: dict[str, Fraction],
             all_work: dict[str, Fraction]) -> Fraction:
    """Longest modeled per-stage time after declared rate promotion."""
    supply = op.supply
    raw_st = getattr(op, "_raw_stages", None)

    def one_stage(work: dict[str, Fraction]) -> Fraction:
        times = [
            value / (as_fraction(supply.effective(pipe)) * rate_scale.get(pipe, Fraction(1)))
            for pipe, value in work.items() if supply.effective(pipe) > 0
        ]
        return max(times, default=Fraction(0))

    if raw_st:
        return sum((one_stage({d.pipe: evaluate(d.work, env) for d in st}) for st in raw_st), Fraction(0))
    return one_stage(all_work)


def _group_ranks(name: str, theta: Any) -> int:
    """Declared participating ranks for this collective/transport group.

    ``members`` is the bound group size.  Only when it is absent do we use the
    primitive axis named by the operator (TP allreduce vs EP dispatch), never
    the maximum of unrelated TP/DP/EP axes.
    """
    members = int(theta.find("members", 0) or 0)
    if members > 0:
        return members
    lower = name.lower()
    if "allreduce" in lower or "all_reduce" in lower or "tp_" in lower:
        axis = "tp"
    elif "dispatch" in lower or "combine" in lower or "alltoall" in lower or "all_to_all" in lower or "ep_" in lower:
        axis = "ep"
    elif "dp_" in lower:
        axis = "dp"
    else:
        axis = "tp"
    value = int(theta.find(axis, 0) or 0)
    if value > 0:
        return value
    # A generic collective may declare only its total rank count.
    return max(1, int(theta.find("ranks", 0) or 0))


def _group_shape(name: str, theta: Any, inv: Mapping[str, Any] | None = None) -> tuple[int, int, int]:
    """Declared participating ranks, ranks per node and node count.

    The physical hardware node size is authoritative when the device profile
    declares it; a stale operator-default ranks_per_node must not turn one
    node into two.
    """
    ranks = _group_ranks(name, theta)
    topology = str(theta.find("topology", "") or "").lower()
    if topology == "hccs":
        return ranks, ranks, 1
    declared_rpn = int((inv or {}).get("ranks_per_node", 0) or 0)
    op_rpn = int(theta.find("ranks_per_node", theta.find("node_ranks", 0)) or 0)
    node_ranks = max(1, min(ranks, declared_rpn or op_rpn or 8))
    return ranks, node_ranks, max(1, math.ceil(ranks / node_ranks))


def _collective_rounds(name: str, theta: Any, inv: Mapping[str, Any] | None = None) -> tuple[int, int]:
    """Ring dependency rounds on declared intra-node and inter-node links."""
    ranks, node_ranks, nodes = _group_shape(name, theta, inv)
    if ranks <= 1:
        return 0, 0
    if "allreduce" in name or "all_reduce" in name:
        # Hierarchical ring: intra reduce-scatter, inter allreduce, intra allgather.
        return 2 * (node_ranks - 1), 2 * (nodes - 1)
    if "reduce_scatter" in name or "all_gather" in name:
        return node_ranks - 1, nodes - 1
    # Dispatch/combine and all-to-all exchange every peer concurrently once per
    # topology level.
    return 1, 1 if nodes > 1 else 0


def compute_hardware_service_factors(
    op: Any, geometry: Any, env: dict[str, Any], all_work: dict[str, Fraction],
) -> tuple[dict[str, Fraction], Fraction, str]:
    """Declare execution rates and nonshared latency; native work is immutable.

    The returned ``scale_pipes`` is a dimensionless elapsed-denominator
    multiplier: a declared FP8 activation/MMAD peak divides cube time, it does
    not double FLOPs.  The scalar return is the inclusive completion floor;
    startup and propagation are separate amounts in ``HW_STARTUP_MS`` /
    ``HW_DEPENDENCY_MS`` so a concurrent branch can anchor the floor to
    readiness: ``duration = max(floor, startup + body + dependency)``.  None of
    them creates resource work.  ``target_pipe`` is retained for the public
    helper ABI.
    """
    supply = op.supply
    pipes = set(all_work)
    family = op.family
    rates = dict(supply.rates)
    undeclared = set(getattr(supply, "undeclared_pipes", ()) or ())
    tokens = max(1, int(geometry.tokens))
    env["CM_RIDGE_SHIFT"] = Fraction(1)
    env["RIDGE_EFFECTIVE_TOKENS"] = Fraction(tokens)
    env["VEC_EFFECTIVE_TOKENS"] = Fraction(tokens)
    env["HW_COMPLETION_FLOOR_MS"] = Fraction(0)
    env["HW_STARTUP_MS"] = Fraction(0)
    # Binders may publish a time-only tail. Collective rounds replace it
    # when they are positive; a zero incoming stays zero.
    incoming_dependency = as_fraction(env.get("HW_DEPENDENCY_MS", 0))
    env["HW_DEPENDENCY_MS"] = Fraction(0)

    # The compiler fixes the latency target structurally; keep the same pipe.
    if family in {"collective", "transport"} and "vector" in pipes:
        target_pipe = "vector"
    elif family in {"collective", "transport"} and "link" in pipes:
        target_pipe = "link"
    elif "cube" in pipes:
        target_pipe = "cube"
    elif "vector" not in undeclared:
        target_pipe = "vector"
    elif pipes:
        target_pipe = sorted(pipes)[0]
    else:
        target_pipe = "hbm"

    inv = dict(supply.invocations)
    rate_scale: dict[str, Fraction] = {}
    cube_rate = _declared_cube_rate(op, all_work) if "cube" in pipes else Fraction(1)
    if cube_rate != 1:
        rate_scale["cube"] = cube_rate

    dep_span = Fraction(0)
    if family in {"collective", "transport"}:
        _ranks, node_ranks, nodes = _group_shape(op.name.lower(), op.operator.theta, inv)
        intra_rounds, inter_rounds = _collective_rounds(op.name.lower(), op.operator.theta, inv)
        global_lat = as_fraction(inv.get("link_latency_ms", 0.0))
        intra_lat = as_fraction(inv.get("link_latency_intra_ms", global_lat))
        inter_lat = as_fraction(inv.get("link_latency_inter_ms", global_lat))
        # An explicit global declaration overrides derived per-route values
        # when it differs from both (public InvocationTiming-style rebind).
        if global_lat > 0 and global_lat != intra_lat and global_lat != inter_lat:
            intra_lat = inter_lat = global_lat
        # A ring reduce/scatter step performs a vector reduction before the
        # step completes.  Its declared vector floor and link propagation are
        # concurrent; a serial ring pays the larger per step, not their sum.
        vector_min = as_fraction(inv.get("vector", 0.0))
        name = op.name.lower()
        reduces = "allreduce" in name or "all_reduce" in name or "reduce_scatter" in name
        step_intra = max(intra_lat, vector_min) if reduces else intra_lat
        step_inter = max(inter_lat, vector_min) if reduces else inter_lat
        dep_span = Fraction(intra_rounds) * step_intra + Fraction(inter_rounds) * step_inter
        peers = max(0, _group_ranks(name, op.operator.theta) - 1)
        if family == "transport" and peers > 0 and any(
                token in name for token in ("dispatch", "combine", "alltoall", "all_to_all")):
            # Serial per-peer transport: charge only the route each peer
            # actually traverses.  Intra-node peers never touch the inter-node
            # latency, and a transfer stage has no unconditional vector floor.
            package_width = max(0, int(inv.get("ranks_per_package", 0) or 0))
            package_lat = as_fraction(inv.get("link_latency_package_ms", intra_lat))
            node_lat = as_fraction(inv.get("link_latency_node_ms", intra_lat))
            mode = str(op.operator.theta.find(
                "transport_algorithm", "") or "per_source_serial").lower()

            def route_lat(source: int, destination: int) -> Fraction:
                if (package_width > 1
                        and source // package_width == destination // package_width):
                    return package_lat
                if source // node_ranks == destination // node_ranks:
                    return node_lat
                return inter_lat

            active = env.get("transport_active_cells")
            if active is None:
                # Uniform probability: every off-diagonal peer is enabled.
                # Inclusive per-source route sum, including an unpaired last
                # die that has no same-package partner.
                per_source, max_route = [], Fraction(0)
                for source in range(_ranks):
                    if package_width > 1:
                        start = (source // package_width) * package_width
                        package_peers = max(
                            0, min(package_width, _ranks - start) - 1)
                    else:
                        package_peers = 0
                    node_start = (source // node_ranks) * node_ranks
                    in_node = max(0, min(node_ranks, _ranks - node_start) - 1)
                    node_peers = max(0, in_node - package_peers)
                    inter_peers = max(0, peers - package_peers - node_peers)
                    per_source.append(
                        Fraction(package_peers) * package_lat
                        + Fraction(node_peers) * node_lat
                        + Fraction(inter_peers) * inter_lat)
                    if package_peers:
                        max_route = max(max_route, package_lat)
                    if node_peers:
                        max_route = max(max_route, node_lat)
                    if inter_peers:
                        max_route = max(max_route, inter_lat)
                rounds = peers
            else:
                # Explicit directed matrix: only the actually enabled cells
                # participate, even when the reverse direction is empty.
                per_source_by_source, count_by_source, max_route = {}, {}, Fraction(0)
                for source, destination in active:
                    source, destination = int(source), int(destination)
                    latency = route_lat(source, destination)
                    per_source_by_source[source] = (
                        per_source_by_source.get(source, Fraction(0)) + latency)
                    count_by_source[source] = count_by_source.get(source, 0) + 1
                    max_route = max(max_route, latency)
                per_source = list(per_source_by_source.values())
                rounds = max(count_by_source.values(), default=0)

            if mode in {"max_path_rounds", "native_upper"}:
                # Declared native upper assumption: the global declared peer
                # count times the slowest enabled route.
                dep_span = Fraction(peers) * max_route
            elif mode in {"pairwise_rounds", "synchronized_rounds"}:
                # Globally synchronized rounds: max active peers per source,
                # each round paying the slowest active route.
                dep_span = Fraction(rounds) * max_route
            else:
                # Default ordinary model: per-source serialized route sum,
                # hotspot over sources.
                dep_span = max(per_source, default=Fraction(0))
        if not any(as_fraction(all_work.get(pipe, 0))
                   for pipe in pipes if pipe.startswith("link")):
            # Empty traffic consumes no wire or propagation (Topology.messages
            # returns no flows; the envelope must match).
            dep_span = Fraction(0)
        env["HW_DEPENDENCY_MS"] = dep_span
        env["HW_STAGE_ROUNDS"] = Fraction(intra_rounds + inter_rounds)
        if family == "transport" and peers > 0:
            env["HW_PEER_STEPS"] = Fraction(peers)
        if reduces and ("allreduce" in name or "all_reduce" in name):
            # Ring reduce-scatter and all-gather are distinct dependency
            # stages; the exported counts let the pipeline scheduler place
            # them without guessing from the combined elapsed time.
            env["HW_REDUCE_ROUNDS"] = Fraction(node_ranks - 1)
            env["HW_GATHER_ROUNDS"] = Fraction(node_ranks - 1)
        elif "reduce_scatter" in name:
            env["HW_REDUCE_ROUNDS"] = Fraction(node_ranks - 1)
            env["HW_GATHER_ROUNDS"] = Fraction(0)
        if nodes == 1:
            # A one-node group traverses only the declared UB mesh.  If the
            # supply has a single global link rate (a minimum over unrelated
            # routes), promote it to the selected intra-node route.  When the
            # supply already carries link_hccs/link_roce, the compiler selects
            # the route itself and no correction is needed.
            selected = as_fraction(inv.get("link_rate_intra_bytes_per_ms", 0.0))
            link_pipes = [pipe for pipe in pipes if pipe.startswith("link") and pipe in rates]
            for pipe in link_pipes:
                base = as_fraction(supply.effective(pipe))
                # A selected physical route may be slower than the supply
                # route, but it never overrides a lower explicit structure cap
                # or a slower source-declared link.  No generic promotion.
                if 0 < selected < base:
                    rate_scale[pipe] = selected / base

    if not inv or not any(v > 0 for v in inv.values()):
        return rate_scale, Fraction(0), target_pipe

    body = _body_ms(op, env, rate_scale, all_work)
    classes = [c for c in _invocation_classes(op, all_work) if c in inv]
    floor = max((as_fraction(inv[c]) for c in classes), default=Fraction(0))
    startup = max((as_fraction(inv.get(f"{c}_startup_ms", 0.0)) for c in classes), default=Fraction(0))
    env["HW_COMPLETION_FLOOR_MS"] = floor
    env["HW_STARTUP_MS"] = startup
    if dep_span <= 0 < incoming_dependency:
        dep_span = incoming_dependency
    env["HW_DEPENDENCY_MS"] = dep_span
    # Standalone TP2 AllGather/ReduceScatter on the source-declared AICPU,
    # BF16 R35 AG/RS branch uses the independent rack08/CANN9.2.0 group2
    # native curve.  The axis is the exact logical per-rank input:
    # shard for AllGather, padded full tensor for ReduceScatter.  Link/ring
    # counters and packed MXFP4/8 MC2 work are never substituted here.
    if (family == "collective"
            and str(getattr(supply, "arch_family", "")) == "A5"
            and str(op.operator.theta.find("a5_native_group2", ""))
            == "native-20261107"):
        ranks, _node_ranks, nodes = _group_shape(
            op.name.lower(), op.operator.theta, inv)
        name = op.name.lower()
        curve_name = ("all_gather" if "all_gather" in name or "allgather" in name
                      else "reduce_scatter" if "reduce_scatter" in name
                      or "reducescatter" in name else "")
        sp_rows = str(op.operator.theta.find("sp_rows", "") or "")
        wire = as_fraction(op.operator.theta.find(
            "wire_bytes", op.operator.theta.find("bytes_per_token", 0)))
        rows = (max(1, int(geometry.tokens)) + 1) // 2
        input_bytes = (Fraction(rows) * wire if sp_rows == "shard"
                       else Fraction(2 * rows) * wire if sp_rows == "padded"
                       else Fraction(0))
        if ranks == 2 and nodes == 1 and curve_name and input_bytes > 0:
            from mindie_sim.observations.a5_comm_calibration import (
                collect_native_service,
            )
            source_path = str(op.operator.theta.find(
                "a5_native_source_path", "") or "")
            stack = str(op.operator.theta.find("a5_native_stack", ""))
            cann = str(op.operator.theta.find("a5_native_cann", ""))
            core = str(op.operator.theta.find("a5_native_execution_core", ""))
            placement = str(op.operator.theta.find("a5_native_placement", ""))
            deployment = str(op.operator.theta.find(
                "a5_native_deployment_image_id", ""))
            try:
                bundle = collect_native_service(
                    source_path, stack=stack, cann=cann,
                    execution_core=core, placement=placement,
                    deployment_image_id=deployment, groups=(ranks,))
                curve = bundle.lookup_curve(
                    curve_name, ranks, placement_scope=placement,
                    dtype="bf16", expansion_mode=core,
                    stack=stack, cann=cann)
            except (OSError, ValueError, ModelingError):
                curve = None
            estimate = curve.estimate(float(input_bytes)) if curve else None
            if estimate is not None:
                floor = max(floor, as_fraction(estimate.duration_ms))
    # The tuple carries the inclusive completion floor; startup/dependency are
    # separate causal quantities in env: duration = max(floor,
    # startup + body + dependency).
    return rate_scale, floor, target_pipe


def apply_hardware_service(
    op: Any, geometry: Any, env: dict[str, Any],
    stage_work: list[dict[str, Fraction]], stage_times: list[Fraction],
    all_work: dict[str, Fraction], all_times: dict[str, Fraction],
    peak_times: dict[str, Fraction],
) -> None:
    """Apply declared execution rates and nonshared completion latency.

    Native ``stage_work``/``all_work`` are never changed.  Elapsed time uses
    the returned rate multiplier; ``add_span`` is a time-only tail implementing
    ``max(floor, startup + body + dependency)``, so no pipe operations are
    invented.  The ordinary compiler path applies the same tuple directly.
    """
    rate_scale, floor, _target_pipe = compute_hardware_service_factors(op, geometry, env, all_work)
    startup = as_fraction(env.get("HW_STARTUP_MS", 0))
    dependency = as_fraction(env.get("HW_DEPENDENCY_MS", 0))
    supply = op.supply
    if rate_scale:
        all_times.clear()
        peak_times.clear()
        for idx, work in enumerate(stage_work):
            times = {
                pipe: value / (as_fraction(supply.effective(pipe)) * rate_scale.get(pipe, Fraction(1)))
                for pipe, value in work.items()
            }
            stage_times[idx] = max(times.values()) if times else Fraction(0)
            for pipe, val in times.items():
                all_times[pipe] = all_times.get(pipe, Fraction(0)) + val
                peak_times[pipe] = max(peak_times.get(pipe, Fraction(0)), val)
    body = sum(stage_times, Fraction(0))
    completion = max(floor, startup + body + dependency)
    tail = completion - body
    if tail > 0 and stage_times:
        stage_times[-1] = stage_times[-1] + tail
