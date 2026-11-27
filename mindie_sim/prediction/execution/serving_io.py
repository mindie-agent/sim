"""Explicit host preparation, device transfers and output delivery.

[Role] Add serving I/O to the same resource engine as model execution.
[Invariants] Transfers conserve declared bytes and contend with device memory.
  Host results follow device commits; client delivery exists only when declared.
  Payload coefficients describe one representative shard, not all TP ranks.
[Timeouts & Locks] Active batches and constant-sized per-request counters only.
[Side-Effect Policy] Mutates only local Engine events and request aggregates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
from mindie_sim.prediction.engine import Resource


@dataclass(frozen=True)
class TransferPolicy:
    bytes_per_s: float
    fixed_bytes: float = 0
    bytes_per_token: float = 0
    bytes_per_request: float = 0
    latency_us: float = 0
    token_basis: str = "valid"

    def __post_init__(self):
        if (not math.isfinite(self.bytes_per_s) or self.bytes_per_s <= 0
                or any(not math.isfinite(value) or value < 0 for value in (
                    self.fixed_bytes, self.bytes_per_token,
                    self.bytes_per_request, self.latency_us))):
            raise ModelingError("serving_transfer_requires_positive_rate_and_nonnegative_work")
        if self.token_basis not in {"valid", "padded"}:
            raise ModelingError("serving_transfer_token_basis_requires_valid_or_padded")

    def volume(self, tokens, requests):
        return self.fixed_bytes + self.bytes_per_token * tokens + self.bytes_per_request * requests


@dataclass(frozen=True)
class WorkBasis:
    """One resource demand whose actual amount is fixed per batch/request/token.

    ``basis`` names the scope whose declared coefficients are multiplied; it is
    not an observed latency. Variable transfer demand is evaluated on the
    actual batch event in ``ServingIO.batch_events``.
    """

    resource: str
    unit: str
    fixed: float = 0.0
    per_token: float = 0.0
    per_request: float = 0.0
    basis: str = "actual_batch"

    def volume(self, tokens, requests):
        return self.fixed + self.per_token * tokens + self.per_request * requests

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FrameworkFragment:
    """A no-layer framework task and its typed ports/event boundary.

    The declaration names the actual trigger and event boundary. ``work`` is a
    formula evaluated from the same batch that executes; it is not a second
    timing path. A delivery barrier carries no resource work because the
    declared latency is an event boundary, not device service.
    """

    name: str
    trigger: str
    ports: PortSet
    work: tuple[WorkBasis, ...] = ()
    layer: int | None = None
    origin: str = "framework"
    event_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.layer is not None:
            raise ModelingError("framework_fragment_must_not_own_a_layer")
        if not self.name or not self.trigger:
            raise ModelingError("framework_fragment_requires_name_and_trigger")
        for item in self.work:
            if (not math.isfinite(item.fixed) or not math.isfinite(item.per_token)
                    or not math.isfinite(item.per_request)
                    or min(item.fixed, item.per_token, item.per_request) < 0):
                raise ModelingError(
                    f"framework_fragment_work_invalid:{self.name}:{item.resource}")

    def to_dict(self):
        return {
            "name": self.name, "origin": self.origin, "layer": self.layer,
            "trigger": self.trigger, "ports": self.ports.to_dict(),
            "work": [item.to_dict() for item in self.work],
            "events": list(self.event_names), "notes": list(self.notes),
        }


class ServingIO:
    def __init__(self, raw, namespace, replicas, client_tails=None):
        raw = dict(raw)
        unknown = raw.keys() - {"host_prepare_ms", "host_result_ms", "h2d", "d2h", "client_delivery_ms"}
        if unknown:
            raise ModelingError("unknown_serving_io_fields:" + ",".join(sorted(unknown)))
        self.enabled = bool(raw)
        self.namespace, self.replicas = namespace, replicas
        self.client_tails = client_tails if client_tails is not None else {}
        self.host_prepare_ms = float(raw.get("host_prepare_ms", 0))
        self.host_result_ms = float(raw.get("host_result_ms", 0))
        self.client_delivery_ms = raw.get("client_delivery_ms")
        if any(not math.isfinite(value) or value < 0 for value in (
                self.host_prepare_ms, self.host_result_ms,
                *(() if self.client_delivery_ms is None else (self.client_delivery_ms,)))):
            raise ModelingError("serving_host_times_require_finite_nonnegative_ms")
        self.transfers = {direction: TransferPolicy(**raw[direction])
                          for direction in ("h2d", "d2h") if direction in raw}
        if "d2h" in self.transfers and self.transfers["d2h"].token_basis != "valid":
            raise ModelingError("d2h_payload_uses_committed_tokens_not_graph_padding")
        # Actual event handles/work volumes for the active batch. They are
        # retained only for an assembler that pops each batch; ordinary serving
        # keeps this disabled so request history stays bounded.
        self.record_events = False
        self.batch_events = {}

    def resources(self):
        return [Resource(self.name(direction, replica), policy.bytes_per_s / 1000, "bytes")
                for direction, policy in self.transfers.items() for replica in range(self.replicas)]

    def name(self, direction, replica):
        return f"{self.namespace}{direction}@r{replica}"

    def transfer_plan(self, direction, replica, *, stage_hbm=None,
                      tokens=0, requests=0):
        """Instantiate the same WorkBasis used by execution and fragments."""
        policy = self.transfers.get(direction)
        if policy is None:
            return ()
        link = WorkBasis(
            self.name(direction, replica), "bytes", fixed=policy.fixed_bytes,
            per_token=policy.bytes_per_token,
            per_request=policy.bytes_per_request, basis=policy.token_basis)
        if stage_hbm is None:
            return (link,)
        return (link, WorkBasis(
            stage_hbm, "bytes", fixed=policy.fixed_bytes,
            per_token=policy.bytes_per_token,
            per_request=policy.bytes_per_request, basis=policy.token_basis))

    def transfer(self, execution, direction, prefix, replica, ready, tokens, requests):
        policy = self.transfers.get(direction)
        if policy is None:
            return ready
        stage = 0 if direction == "h2d" else execution.case.runtime.parallel.pp - 1
        stage_hbm = execution.stage_names[replica, stage]["hbm"]
        plan = self.transfer_plan(direction, replica, stage_hbm=stage_hbm,
                                  tokens=tokens, requests=requests)
        volume = policy.volume(tokens, requests)
        demand = {basis.resource: basis.volume(tokens, requests) for basis in plan}
        coupled = False
        dies = execution.stage_dies.get((replica, stage), ())
        machine = getattr(execution.case, "machine", None)
        if volume > 0.0 and dies and machine is not None:
            from mindie_sim.prediction.collectives.ports import (
                host_dma_resources_for_dies,
                host_transfer_demand,
            )
            missing = [
                res for res in host_dma_resources_for_dies(
                    machine, dies, namespace=execution.namespace,
                )
                if res.name not in execution.engine.resources
            ]
            if missing:
                execution.engine.extend(missing)
            host_demand = host_transfer_demand(
                machine, dies[0], direction, float(volume),
                namespace=execution.namespace, dies=dies,
            )
            if host_demand:
                demand.update(host_demand)
                coupled = True
        copied = execution.engine.submit(
            prefix + ":" + direction + ":payload",
            demand,
            after=(ready,), owner=f"replica:{replica}", coupled=coupled)
        if not policy.latency_us:
            return copied
        return execution.engine.submit(prefix + ":" + direction + ":complete", after=(copied,),
                                       latency=policy.latency_us / 1000, owner=f"replica:{replica}")

    def input(self, execution, prefix, replica, schedule, work):
        if not self.enabled or not work.plan.requests:
            return schedule, schedule
        prepared = execution.engine.submit(
            prefix + ":host_prepare", {f"{self.namespace}scheduler@r{replica}": self.host_prepare_ms},
            after=(schedule,), owner=f"replica:{replica}")
        policy = self.transfers.get("h2d")
        tokens = work.padded_tokens if policy and policy.token_basis == "padded" else work.valid_tokens
        client_requests = sum(
            1 for view in work.plan.requests if not getattr(view, "padding", False)
        )
        copied = self.transfer(execution, "h2d", prefix, replica, prepared,
                               tokens, client_requests)
        if not self.record_events:
            return prepared, copied
        self.batch_events[prefix] = {
            "replica": replica,
            "requests": tuple(view.request_id for view in work.plan.requests),
            "tokens": tokens,
            "schedule": schedule,
            "host_prepare": prepared,
            "input_ready": copied,
            "input_bytes": (policy.volume(tokens, client_requests)
                            if policy is not None else 0.0),
        }
        return prepared, copied

    def output(self, execution, prefix, replica, work, commits, body_complete, emitted):
        if not self.enabled:
            return {request: commits[request] for request in emitted}, {}, body_complete
        if not work.plan.requests:
            return {}, {}, body_complete
        ready = execution.engine.submit(
            prefix + ":output_ready",
            after=tuple(commits[request] for request in emitted) or (body_complete,),
            owner=f"replica:{replica}")
        transfer_ready, transfer_bytes = ready, 0.0
        policy = self.transfers.get("d2h")
        if emitted:
            transfer_ready = self.transfer(execution, "d2h", prefix, replica, ready,
                                           sum(emitted.values()), len(emitted))
            if policy is not None:
                transfer_bytes = policy.volume(sum(emitted.values()), len(emitted))
        host = execution.engine.submit(
            prefix + ":host_result", {f"{self.namespace}scheduler@r{replica}": self.host_result_ms},
            after=(transfer_ready,), owner=f"replica:{replica}")
        host_results = {request: host for request in emitted}
        client = {}
        propagated = None
        if emitted and self.client_delivery_ms is not None:
            propagated = execution.engine.submit(
                prefix + ":client_delivery", after=(host,), latency=self.client_delivery_ms,
                owner=f"replica:{replica}")
            for request in emitted:
                previous = self.client_tails.get(request)
                delivered = execution.engine.submit(
                    prefix + ":client_order:" + request,
                    after=(propagated, previous) if previous is not None else (propagated,),
                    owner=f"replica:{replica}")
                client[request] = self.client_tails[request] = delivered
        if self.record_events:
            event = self.batch_events.setdefault(prefix, {"replica": replica})
            event.update({
                "output_ready": ready, "d2h_ready": transfer_ready,
                "d2h_bytes": transfer_bytes, "host_result": host,
                "client_propagation": propagated, "deliveries": dict(client),
            })
        # Even a chunk with no sampled token has CPU completion work.
        return host_results, client, host

    def pop_batch_events(self, prefix):
        """Take the active batch's actual event handles and release the entry."""
        return self.batch_events.pop(prefix, None)

    def fragments(self, replica=0, *, h2d_hbm=None, d2h_hbm=None):
        """Declared no-layer framework tasks, triggers and typed ports.

        These declarations describe the same submissions performed by
        ``input``/``output``; ``batch_events`` carries their actual occurrence.
        Work formulas are evaluated on the batch that executes.
        """
        if not self.enabled:
            return ()
        scheduler = f"{self.namespace}scheduler@r{replica}"

        def ready(name):
            return EventPortSpec(name="ready." + name, direction="in", event="ready",
                                 scope="cohort", readiness="same_invocation")

        def complete(name, scope="cohort"):
            return EventPortSpec(name="complete." + name, direction="out",
                                 event="complete", scope=scope,
                                 readiness="producer_complete")

        def done(name, scope="cohort"):
            return EventPortSpec(name="done." + name, direction="in",
                                 event="complete", scope=scope,
                                 readiness="producer_complete")

        def recorded(name):
            return EventPortSpec(name="record." + name, direction="out",
                                 event="record", scope="cohort",
                                 readiness="producer_complete")

        fragments = [FrameworkFragment(
            name="host_prepare", trigger="per_batch",
            ports=PortSet(event_in=(ready("schedule"),),
                          event_out=(recorded("host_prepare"),
                                     complete("host_prepare"))),
            work=(WorkBasis(scheduler, "ms", fixed=self.host_prepare_ms,
                            basis="per_batch"),),
            event_names=("host_prepare",),
            notes=("reads static request/shape fields; may overlap prior device work",),
        )]
        h2d = self.transfers.get("h2d")
        if h2d is not None:
            fragments.append(FrameworkFragment(
                name="input_transfer", trigger="per_batch",
                ports=PortSet(event_in=(ready("host_prepare"),),
                              event_out=(recorded("h2d"),
                                         complete("h2d"))),
                work=self.transfer_plan("h2d", replica, stage_hbm=h2d_hbm),
                event_names=("h2d", "h2d_complete") if h2d.latency_us else ("h2d",),
                notes=("payload coefficients describe one representative shard",),
            ))
        fragments.append(FrameworkFragment(
            name="output_ready", trigger="per_batch_with_commits",
            ports=PortSet(
                event_in=(EventPortSpec(name="commit.token", direction="in",
                                        event="commit", scope="request",
                                        optional=True),
                          ready("body_complete")),
                event_out=(EventPortSpec(name="ready.output", direction="out",
                                         event="ready", scope="cohort"),)),
            event_names=("output_ready",),
        ))
        d2h = self.transfers.get("d2h")
        if d2h is not None:
            fragments.append(FrameworkFragment(
                name="output_transfer", trigger="per_emitted_batch",
                ports=PortSet(event_in=(EventPortSpec(
                    name="ready.output", direction="in", event="ready",
                    scope="cohort"),),
                    event_out=(recorded("d2h"), complete("d2h"))),
                work=self.transfer_plan("d2h", replica, stage_hbm=d2h_hbm),
                event_names=("d2h", "d2h_complete") if d2h.latency_us else ("d2h",),
                notes=("d2h payload uses committed tokens, never graph padding",),
            ))
        fragments.append(FrameworkFragment(
            name="host_result", trigger="per_batch",
            ports=PortSet(event_in=(
                EventPortSpec(name="done.d2h", direction="in", event="complete",
                              scope="cohort", optional=True),
                EventPortSpec(name="ready.output", direction="in", event="ready",
                              scope="cohort", optional=True),
            ), event_out=(complete("host_result"),)),
            work=(WorkBasis(scheduler, "ms", fixed=self.host_result_ms,
                            basis="per_batch"),),
            event_names=("host_result",),
        ))
        if self.client_delivery_ms is not None:
            fragments.extend((
                FrameworkFragment(
                    name="client_delivery", trigger="per_batch_with_emissions",
                    ports=PortSet(event_in=(done("host_result"),),
                                  event_out=(complete("propagation"),)),
                    event_names=("client_propagation",),
                    notes=("one declared propagation per batch; declared latency "
                           "is an event boundary, not device service",),
                ),
                FrameworkFragment(
                    name="client_order", trigger="per_request_per_batch",
                    ports=PortSet(event_in=(done("propagation"),),
                                  event_out=(complete("delivered", "request"),)),
                    event_names=("client_delivery",),
                    notes=("one per-request delivery event after the batch propagation",),
                ),
            ))
        return tuple(fragments)

    def summary(self):
        return {"enabled": self.enabled, "host_prepare_ms": self.host_prepare_ms,
                "host_result_ms": self.host_result_ms,
                "client_delivery_ms": self.client_delivery_ms,
                "transfers": {name: asdict(policy) for name, policy in self.transfers.items()},
                "payload_scope": "representative_shard",
                "host_resource": "replica_scheduler_fifo",
                "copy_resources": "separate_direction_links_shared_with_endpoint_hbm"}


@dataclass
class TokenTimeline:
    tokens: int = 0
    first_ms: float | None = None
    last_ms: float | None = None
    itl_sum: float = 0
    itl_max: float = 0

    def record(self, count, at_ms):
        if self.first_ms is None:
            self.first_ms = at_ms
        else:
            gap = at_ms - self.last_ms
            self.itl_sum += gap
            self.itl_max = max(self.itl_max, gap)
        self.last_ms = at_ms
        self.tokens += count

    def row(self, state):
        row = state.to_dict()
        complete = self.tokens >= state.request.output_tokens
        intervals = max(0, self.tokens - 1)
        return {**row, "generated_tokens": self.tokens,
                "first_token_abs_ms": self.first_ms, "last_token_abs_ms": self.last_ms,
                "finish_ms": self.last_ms if complete else None,
                "ttft_ms": None if self.first_ms is None else self.first_ms - row["arrival_ms"],
                "tpot_ms": ((self.last_ms - self.first_ms) / intervals
                            if complete and intervals else None),
                "e2e_ms": self.last_ms - row["arrival_ms"] if complete else None,
                "status": "served" if complete else "unserved",
                "itl": {"count": intervals, "mean_ms": self.itl_sum / intervals if intervals else None,
                        "max_ms": self.itl_max if intervals else None}}
