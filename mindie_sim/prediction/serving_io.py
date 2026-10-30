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
from .engine import Resource


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

    def resources(self):
        return [Resource(self.name(direction, replica), policy.bytes_per_s / 1000, "bytes")
                for direction, policy in self.transfers.items() for replica in range(self.replicas)]

    def name(self, direction, replica):
        return f"{self.namespace}{direction}@r{replica}"

    def transfer(self, execution, direction, prefix, replica, ready, tokens, requests):
        policy = self.transfers.get(direction)
        if policy is None:
            return ready
        stage = 0 if direction == "h2d" else execution.case.runtime.parallel.pp - 1
        volume = policy.volume(tokens, requests)
        return execution.engine.submit(
            prefix + ":" + direction,
            {self.name(direction, replica): volume,
             execution.stage_names[replica, stage]["hbm"]: volume},
            after=(ready,), latency=policy.latency_us / 1000, owner=f"replica:{replica}")

    def input(self, execution, prefix, replica, schedule, work):
        if not self.enabled or not work.plan.requests:
            return schedule, schedule
        prepared = execution.engine.submit(
            prefix + ":host_prepare", {f"{self.namespace}scheduler@r{replica}": self.host_prepare_ms},
            after=(schedule,), owner=f"replica:{replica}")
        policy = self.transfers.get("h2d")
        tokens = work.padded_tokens if policy and policy.token_basis == "padded" else work.valid_tokens
        return prepared, self.transfer(execution, "h2d", prefix, replica, prepared,
                                       tokens, len(work.plan.requests))

    def output(self, execution, prefix, replica, work, commits, body_complete, emitted):
        if not self.enabled:
            return {request: commits[request] for request in emitted}, {}, body_complete
        if not work.plan.requests:
            return {}, {}, body_complete
        ready = execution.engine.submit(
            prefix + ":output_ready",
            after=tuple(commits[request] for request in emitted) or (body_complete,),
            owner=f"replica:{replica}")
        if emitted:
            ready = self.transfer(execution, "d2h", prefix, replica, ready,
                                  sum(emitted.values()), len(emitted))
        host = execution.engine.submit(
            prefix + ":host_result", {f"{self.namespace}scheduler@r{replica}": self.host_result_ms},
            after=(ready,), owner=f"replica:{replica}")
        host_results = {request: host for request in emitted}
        client = {}
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
        # Even a chunk with no sampled token has CPU completion work.
        return host_results, client, host

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
