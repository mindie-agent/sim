"""Derive conditional copy pipe-busy supply from retained PMU observations.

[Role] Freeze B16/32/64 affine responses and retain untouched B1/4/8 checks.
[Invariants] Fit PMU busy only, never native duration or Body residual. Program,
  metadata and byte features have rank one; no isolated primitive is identified.
[Timeouts & Locks] One bounded summary JSON; no raw capture scan or device work.
[Side-Effect Policy] Writes the explicitly requested parameter packet.
"""
import argparse
import hashlib
import json
from pathlib import Path


FIELDS = {"scalar": "aiv_scalar_time(us)", "mte2": "aiv_mte2_time(us)",
          "mte3": "aiv_mte3_time(us)"}


def fit(points):
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum(
        (x - mean_x) ** 2 for x, _ in points)
    return mean_y - slope * mean_x, slope


def packet(source):
    content = source.read_bytes()
    data = json.loads(content)
    directions = {}
    training = data["directional_copy_points"]
    for direction in ("gather", "scatter"):
        selected = [p for p in training if p["direction"] == direction]
        assert {p["batch"] for p in selected} == {16, 32, 64}
        pipes = {}
        for pipe, field in FIELDS.items():
            intercept, slope = fit([(p["source_programs_per_declared_owner"],
                                     p["pmu"][field]["mean"]) for p in selected])
            pipes[pipe] = {"resource": "copy_joint_" + pipe + "_busy",
                           "owner_busy_ns": intercept * 1000,
                           "program_busy_ns": slope * 1000}
        directions[direction] = pipes
    observations = []
    for row in training:
        observations.append({"split": "conditional_train", "batch": row["batch"],
            "direction": row["direction"], "calls": row["calls"],
            "source_identity": row["source_identity"],
            "busy_us": {pipe: row["pmu"][field]["mean"] for pipe, field in FIELDS.items()},
            "duration_us": row["pmu"]["Duration(us)"]["mean"]})
    for row in data["records"]:
        if row["native_operator_type"] != "_copy_conv_state_kernel":
            continue
        shapes = row["conditional_state_shapes"]
        batch = shapes[0][0] if isinstance(shapes[0], list) else shapes[0]
        observations.append({"split": "independent_check", "batch": batch,
            "direction": row["copy_direction"], "calls": row["calls"],
            "source_identity": row["source_identity"],
            "busy_us": {pipe: row["pmu"][field]["statistics"]["mean"]
                        for pipe, field in FIELDS.items()},
            "duration_us": row["pmu"]["Duration(us)"]["statistics"]["mean"]})
    provenance = {"kind": "conditional_joint_source_program_busy",
        "source": str(source), "sha256": hashlib.sha256(content).hexdigest(),
        "train_batches": [16, 32, 64], "independent_check_batches": [1, 4, 8],
        "feature_rank": 1, "features": {"programs": 1, "metadata_loads": 3,
                                            "payload_bytes": 512},
        "native_duration_used_for_fit": False, "body_residual_used": False,
        "capture_binary_and_grid_assignment_confirmed": False,
        "pmu_scope": data["pmu_scope"],
        "interpretation": "overlapping busy lower bounds; not independent DMA or scalar prices"}
    return {"schema": "mindie-sim.copy-joint-service/1", "provenance": provenance,
        "joint_program_service": {
            "contract": {"payload_bytes_per_program": 512, "metadata_bytes_per_program": 12,
                         "state_tail_shape": [6, 3072], "dtype": "bf16", "owners": 64},
            "directions": directions, "provenance": provenance},
        "compute_arch": {"instruction_services": {
            "copy_joint_" + pipe + "_busy": {"unit": "busy_ns", "owner_axis": "vector",
                "per_owner_rate": 1e9, "provenance": provenance} for pipe in FIELDS}},
        "observations": observations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet(args.source), indent=2) + "\n")
