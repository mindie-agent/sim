"""Independent comparison with measured service and kernel observations.

[Role] Quantify prediction error without feeding observations into execution.
[Invariants] Comparison preserves metric boundary and statistic. Missing pairs
  remain absent; an unmatched row is never assigned zero error.
[Timeouts & Locks] Streaming extraction; bounded metric summaries.
[Side-Effect Policy] Reads explicit inputs only.
"""
import json
import math
from pathlib import Path

import ijson


class _Moments:
    """Constant-memory exact moments; no invented quantiles from chunk stats."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self):
        return {} if not self.count else {
            "count": self.count, "mean": self.total / self.count,
            "min": self.minimum, "max": self.maximum,
        }


def service_observation(path):
    wanted = {"label", "model", "input_len", "output_len", "concurrency", "num_prompts",
              "n_ok", "n_failed", "wall_s", "ttft_ms", "tpot_ms", "e2e_ms",
              "output_tokens_per_s", "total_output_tokens", "output_throughput",
              "output_throughput_tok_s", "token_exact"}
    request_fields = {"ok", "completion_tokens", "decode_ms", "ttft_ms", "n_text_chunks"}
    record, request = {}, {}
    tpot, last_text_e2e = _Moments(), _Moments()
    observed_requests = successful = exact_usage = output_tokens = 0
    # Parse scalar leaves so token arrays and full response text are never
    # accumulated, even in captures hundreds of megabytes in size.
    with Path(path).open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == "per_request.item" and event == "start_map":
                request = {}
            elif prefix == "per_request.item" and event == "end_map":
                observed_requests += 1
                if request.get("ok"):
                    successful += 1
                    tokens = request.get("completion_tokens")
                    decode = request.get("decode_ms")
                    first = request.get("ttft_ms")
                    if tokens is not None:
                        exact_usage += 1
                        output_tokens += tokens
                        if tokens > 1 and decode is not None:
                            tpot.add(decode / (tokens - 1))
                    if first is not None and decode is not None:
                        last_text_e2e.add(first + decode)
            elif prefix.startswith("per_request.item.") and event in {"number", "boolean", "null"}:
                key = prefix.removeprefix("per_request.item.")
                if key in request_fields:
                    request[key] = value
            keys = prefix.split(".")
            if keys[0] not in wanted or len(keys) > 2 or event not in {"string", "number", "boolean", "null"}:
                continue
            if len(keys) == 1:
                record[keys[0]] = value
            else:
                record.setdefault(keys[0], {})[keys[1]] = value
    if "n_ok" not in record or "ttft_ms" not in record:
        raise ValueError("not_a_raw_service_measurement")
    # The historical bench_stream producer divides by nonempty SSE events,
    # which can carry multiple speculative tokens. Preserve that original
    # statistic, but normalize TPOT only from actual usage and elapsed decode.
    record["reported_metrics"] = {key: record.pop(key) for key in
                                  ("tpot_ms", "e2e_ms", "output_throughput_tok_s",
                                   "output_tokens_per_s", "output_throughput", "total_output_tokens")
                                  if key in record}
    record["tpot_ms"] = tpot.summary()
    record["e2e_ms"] = last_text_e2e.summary()
    complete_usage = successful == exact_usage == record["n_ok"] and successful > 0
    record["normalization"] = {
        "per_request_records": observed_requests,
        "successful_requests": successful, "requests_with_usage": exact_usage,
        "tpot": "per-request decode_ms / (usage.completion_tokens - 1)",
        "e2e": "first text latency + first-to-last text elapsed time",
        "quantiles": "not reconstructed from source chunk-interval quantiles",
    }
    wall = record.get("wall_s")
    record["output_tps"] = output_tokens / wall if complete_usage and wall and wall > 0 else None
    record["source"] = str(Path(path).resolve())
    record["boundary"] = "client_stream"
    return record


def compare(prediction, observed, *, boundary, statistics=("p50", "mean")):
    if observed.get("boundary") != boundary:
        raise ValueError("observation_boundary_conflicts_with_declared_source_boundary")
    rows = []
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        actual = observed.get(name, {})
        for statistic in statistics:
            measured = actual.get(statistic, actual.get("median") if statistic == "p50" else None)
            estimated = prediction["metrics"].get(name, {}).get(statistic)
            if measured is None or estimated is None:
                continue
            rows.append({"metric": name, "statistic": statistic, "predicted": estimated,
                         "observed": measured, "error": estimated - measured,
                         "relative_error": (estimated / measured - 1) if measured else None})
    measured, estimated = observed.get("output_tps"), prediction["metrics"].get("output_tps")
    if measured is not None and estimated is not None:
        rows.append({"metric": "output_tps", "statistic": "aggregate", "predicted": estimated,
                     "observed": measured, "error": estimated - measured,
                     "relative_error": estimated / measured - 1 if measured else None})
    matching = prediction.get("metric_boundary") == boundary
    return {"schema": "mindie-sim.validation/1", "source": observed.get("source"),
            "prediction_boundary": prediction.get("metric_boundary"), "observation_boundary": boundary,
            "comparable": matching, "rows": rows, "normalization": observed.get("normalization"),
            "boundary_difference": None if matching else
            "Client observations include transport and frontend work outside device token commitment."}


def fit_resource_rate(samples):
    """Fit a primitive's nonnegative setup + work/rate from training trials."""
    from .service_fitting import nonnegative_affine
    train = [sample for sample in samples if sample["split"] == "train"]
    heldout = [sample for sample in samples if sample["split"] == "heldout"]
    n = len(train)
    if any(not math.isfinite(s[key]) or s[key] <= 0 for s in samples
           for key in ("work_bytes", "duration_ms")):
        raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
    intercept, rate = nonnegative_affine([(s["work_bytes"], s["duration_ms"]) for s in train])
    return {"bytes_per_ms": rate, "setup_ms": intercept, "train_trials": n,
            "heldout": [{"work_bytes": s["work_bytes"], "observed_ms": s["duration_ms"],
                         "predicted_ms": intercept + s["work_bytes"] / rate,
                         "relative_error": (intercept + s["work_bytes"] / rate) / s["duration_ms"] - 1}
                        for s in heldout]}
