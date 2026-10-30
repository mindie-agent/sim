"""Report independent timing accuracy and declared scaling/configuration checks.

[Role] Run the generic report or adapt a retained, frozen GPU copy calibration.
[Invariants] Recompute comparisons from timings, never input pass flags. The copy
  adapter models the actual 24-launch batch; training shapes are not validation.
[Timeouts & Locks] Bounded metadata and CPU-only finite programs; no device work.
[Side-Effect Policy] Reads sources and writes only the requested report.
"""
import argparse
from dataclasses import replace
from pathlib import Path

from mindie_sim.__main__ import emit, read_json
from mindie_sim.observations.accuracy import accuracy_report
from mindie_sim.observations.store import digest_file
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.services import ServiceCurve, service_identity
from tools.measure.replay_memory_service import copy_at, probe_inputs


def retained_copy_campaign(report_path, *, root=".", tolerances=None):
    """Reuse frozen service parameters while evaluating the real event boundary."""
    root, report_path = Path(root), Path(report_path)
    report = read_json(report_path)
    if report["schema"] != "mindie-sim.memory-service-replay/1":
        raise ValueError("accuracy_copy_adapter_requires_retained_service_replay")
    source = root / report["source"]
    source_hash = digest_file(source)
    if source_hash != report["source_sha256"]:
        raise ValueError("accuracy_copy_capture_hash_mismatch")
    raw = read_json(source)
    case, template, packet = probe_inputs(
        raw, report["source"],
        memory_bytes_per_second=report["hardware_projection"]["memory_bytes_per_second"])
    frozen = ServiceCurve.from_mapping(report["calibration"]["model"])
    if (frozen.model_id != report["calibration"]["model_id"]
            or dict(frozen.identity) != service_identity(case, template)):
        raise ValueError("accuracy_copy_frozen_model_identity_mismatch")
    configured = replace(case, runtime=replace(case.runtime, kernels={"recipes": {
        template.recipe: {"kind": "resource_service", "model": frozen.to_dict(),
                          "outside_domain": "analytic"},
    }}))
    hardware = {"uuid": raw["uuid"], "gpu": raw["gpu"], "driver": raw["driver"]}
    implementation = {"kernel": raw["kernel"], "sources": dict(frozen.identity["implementation"])}
    repetitions = 24
    predictions, identities = [], {}
    for work in sorted({sample["work"] for sample in packet["samples"]}):
        identity = {
            "level": "subgraph", "boundary": "cuda_event_device_batch",
            "hardware": hardware, "implementation": implementation, "phase": "primitive",
            "shape": {"copy_bytes": work // 2}, "dtype": "uint32", "layout": "contiguous",
            "execution": {
                "invocations": repetitions, "dependency": "serial",
                "axes": dict(frozen.identity["axes"]),
            },
        }
        identities[work] = identity
        ops = tuple(replace(copy_at(template, work), op_id=f"copy{index}", stream="copy",
                            deps=(f"copy{index - 1}",) if index else ())
                    for index in range(repetitions))
        duration = graph_costs(configured, ops, isolated=False)["duration_ms"]
        predictions.append({
            "point_id": f"copy_{work // 2}x{repetitions}", "identity": identity,
            "duration_ms": duration, "occurrences": 1,
        })
    samples = []
    for index, sample in enumerate(raw["samples"]):
        samples.append({
            "sample_id": f"{source_hash}:{index}",
            "identity": identities[sample["work_bytes"]],
            # The original field is elapsed event time divided by 24. Restore
            # the event-batch duration, matching the 24 serial modeled kernels.
            "duration_ms": sample["duration_ms"] * sample["repetitions"],
            "run_id": source_hash,
            "split": sample["split"], "source": f"{source}#/samples/{index}",
            "metadata": {"trial": sample["trial"], "invocations": sample["repetitions"]},
        })
    training = []
    prefix = report["source"] + "#/samples/"
    for reference in frozen.provenance["training_sources"]:
        if not reference.startswith(prefix):
            raise ValueError("accuracy_copy_training_source_not_in_retained_capture")
        index = int(reference.removeprefix(prefix))
        training.append(samples[index]["sample_id"])
    heldout = {sample["work_bytes"] for sample in raw["samples"] if sample["split"] == "heldout"}
    return {
        "schema": "mindie-sim.accuracy-input/1",
        "tolerances": tolerances or {}, "training_sample_ids": training,
        "predictions": predictions, "observations": samples,
        "relations": [
            {"id": "copy_size_scaling", "kind": "trend", "axis": "shape.copy_bytes",
             "points": [point["point_id"] for point in predictions]},
            {"id": "copy_implementation_order", "kind": "ranking", "vary": ["implementation"],
             "points": [point["point_id"] for point in predictions
                        if point["identity"]["shape"]["copy_bytes"] * 2 in heldout]},
        ],
        "provenance": {
            "capture": str(source), "capture_sha256": source_hash,
            "frozen_service_report": str(report_path), "report_sha256": digest_file(report_path),
            "frozen_service_model_id": frozen.model_id,
            "prediction": "public graph_costs; 24 serial copy programs; frozen service model",
            "observation": "original CUDA event batch reconstructed from recorded per-launch average",
            "training_split": "original split, plus exclusion of frozen-model training source IDs",
            "new_hardware_execution": False,
        },
        "limits": [
            "one heldout copy size in the original capture; not an independent-run validation",
            "five trial batch averages in one capture run; launches are not independent observations",
            "training sizes cannot establish heldout scaling trends",
            "one measured implementation; configuration ordering is unassessed",
            "retained producer/PTX hashes do not identify the loaded binary",
            "caller-declared memory ceiling is not a measured general GPU bandwidth",
            "host launch gaps and cache-state changes are not independently modeled",
            "no other operator family, full-model, serving or concurrent-device accuracy is established",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", nargs="?")
    parser.add_argument("--copy-replay", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--absolute-tolerance-ms", type=float)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    if bool(args.campaign) == bool(args.copy_replay):
        parser.error("supply one campaign or --copy-replay")
    override = {name: value for name, value in (
        ("absolute_ms", args.absolute_tolerance_ms), ("relative", args.relative_tolerance)
    ) if value is not None}
    if args.copy_replay:
        campaign = retained_copy_campaign(args.copy_replay, root=args.root, tolerances=override)
    else:
        campaign = read_json(args.campaign)
        campaign["tolerances"] = {**campaign.get("tolerances", {}), **override}
    result = accuracy_report(campaign)
    emit(result, args.output)
    print(result["coverage"])


if __name__ == "__main__":
    main()
