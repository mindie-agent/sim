"""Reproduce the scalar-store queue identification from primitive training rows.

[Role] Fit response and issue/control service against independent address probes.
[Invariants] Only the declared 24 training geometries enter parameter selection.
  No model/capture timings or heldout rows are read. This identifies a candidate
  physical queue; it does not turn that candidate into a hardware specification.
[Timeouts & Locks] Compiler <=30s; finite capacity/coordinate search, one CPU.
[Side-Effect Policy] Temporary CPU library and a caller-selected JSON report.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np


def fit(source, output):
    with source.open() as stream:
        rows = [json.loads(line) for index, line in enumerate(stream) if index < 24]
    observed = np.array([statistics.median(row["durations_us"]) for row in rows])
    with tempfile.TemporaryDirectory(prefix="sim-store-fit-") as temporary:
        library = Path(temporary) / "oracle.so"
        subprocess.run(["c++", "-O2", "-std=c++17", "-shared", "-fPIC",
                        str(Path(__file__).with_name("store_queue_oracle.cpp")), "-o", str(library)],
                       check=True, timeout=30)
        oracle = ctypes.CDLL(str(library))
        oracle.tile.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double] * 3
        oracle.tile.restype = ctypes.c_double

        def predict(capacity, parameters):
            return np.array([
                4 * oracle.tile(64, row["heads"], row["leading_dim"], 0, 128,
                                capacity, *parameters) / 1000 for row in rows])

        def loss(capacity, parameters):
            predicted = predict(capacity, parameters)
            setup = max(0., min(4., sum((observed - predicted) / observed**2) /
                               sum(1 / observed**2)))
            return float(np.mean(((predicted + setup - observed) / observed)**2)), setup

        candidates = []
        for capacity in range(4, 17):
            parameters = [90., 4., 4.]
            for steps in ((20., 4., 4.), (5., 1., 1.), (1., .25, .25), (.25, .0625, .0625)):
                for _ in range(20):
                    before = tuple(parameters)
                    for index, step in enumerate(steps):
                        choices = []
                        for delta in (-step, 0, step):
                            changed = parameters.copy()
                            changed[index] = max(.001, changed[index] + delta)
                            choices.append((loss(capacity, changed)[0], changed))
                        parameters = min(choices, key=lambda item: item[0])[1]
                    if tuple(parameters) == before:
                        break
            score, setup = loss(capacity, parameters)
            candidates.append({
                "outstanding": capacity, "granule_bytes": 128,
                "response_ns": parameters[0], "issue_ns": parameters[1],
                "row_control_ns": parameters[2], "setup_us": setup,
                "mean_squared_relative_error": score,
            })
        selected = min(candidates, key=lambda item: item["mean_squared_relative_error"])
        prediction = predict(selected["outstanding"], [
            selected["response_ns"], selected["issue_ns"], selected["row_control_ns"],
        ]) + selected["setup_us"]
    report = {
        "source": str(source), "source_rows": [1, 24], "split": "train",
        "scope": "isolated BF16 scalar-store primitive; no model timing inputs",
        "identification": "128-byte granule inferred from separate stride experiment",
        "candidates": candidates, "selected": selected,
        "training_comparison": [
            {"heads": row["heads"], "leading_dim": row["leading_dim"],
             "predicted_us": float(value), "observed_us": float(raw),
             "error_percent": float(100 * (value / raw - 1))}
            for row, value, raw in zip(rows, prediction, observed)
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(selected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    fit(args.source, args.output)
