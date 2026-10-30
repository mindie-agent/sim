"""Replay retained copy observations through primitive and graph modeling layers.

[Role] Adapt the CUDA memory probe, train a service curve and score heldout work.
[Invariants] Original splits and trial identities survive. Timing is per launch
  averaged over the recorded batch. A copy response is not a universal DRAM rate.
[Timeouts & Locks] Bounded metadata and finite CPU-only programs; no locks.
[Side-Effect Policy] Reads retained sources; writes only requested reports.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mindie_sim.__main__ import emit, read_json
from mindie_sim.observations.service_fitting import fit_service_curve
from mindie_sim.observations.validation import fit_resource_rate
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.services import service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView


def probe_inputs(raw, source, *, memory_bytes_per_second):
    if raw["schema"] != "mindie-sim.cuda-memory-observations/1":
        raise ValueError("expected_cuda_memory_probe_v1")
    if raw["boundary"] != "cuda_event_device" or raw["kernel"] != "stream_copy":
        raise ValueError("copy_probe_implementation_or_boundary_mismatch")
    # This is a memory-only hardware projection. Required compute fields are
    # unused placeholders, not recovered GPU capabilities.
    case = parse_case({
        "machine": {
            "name": raw["gpu"], "hardware_key": raw["uuid"], "architecture": "cuda",
            "device": {
                "memory_bytes": 1024**3, "compute_units": 1, "compute_peaks": {"bf16": 1},
                "rates": {"memory": memory_bytes_per_second, "vector": 1},
                "resource_units": {"hbm": 1},
            },
        },
        "model": {"name": "memory-probe", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16},
        "runtime": {"vllm_impl": "cuda_driver_api", "vllm_version": raw["driver"]},
        "workload": {"requests": [{"id": "probe", "prompt_tokens": 1, "output_tokens": 1}]},
    })
    implementation = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("cuda_probe.py", "copy.ptx")
    }
    axes = {"block_threads": 256, "word_bytes": 4, "alignment_bytes": 4,
            "load_cache": "cg", "store_cache": "wb"}
    op = OpSpec(
        "copy", "copy", "primitive", 0, 0, (), (1,), (1,), (1,), (1,),
        dtype="uint32", layout="contiguous", recipe=raw["kernel"],
        metadata=(("kernel", raw["kernel"]), ("implementation", implementation),
                  ("service_axes", axes)),
    )
    samples = []
    repetitions = {sample["repetitions"] for sample in raw["samples"]}
    if repetitions != {24}:
        raise ValueError("copy_probe_requires_recorded_24_launch_protocol")
    for index, sample in enumerate(raw["samples"]):
        if not sample["correct"] or sample["work_bytes"] != 2 * sample["size_bytes"]:
            raise ValueError("copy_observation_has_invalid_result_or_traffic")
        samples.append({
            "work": sample["work_bytes"], "duration_ms": sample["duration_ms"],
            "run_id": f"{source}:trial={sample['trial']}", "repeat_index": 0,
            "split": sample["split"], "source": f"{source}#/samples/{index}",
        })
    packet = {
        "resource": "hbm", "work_unit": "bytes", "boundary": "device_kernel",
        "identity": service_identity(case, op), "active_units": 1,
        "provenance": {
            "observation_boundary": raw["boundary"],
            "duration_normalization": "recorded_device_event_batch_ms_divided_by_24_launches",
            "implementation_binding": "retained_probe_source; loaded_binary_hash_not_captured",
            "abstraction": "one_memory_service_plus_startup_for_this_copy_protocol",
        },
        "samples": samples,
    }
    return case, op, packet


def copy_at(template, work):
    shape = (work // 8,)  # uint32 read and write: eight physical bytes per element.
    return replace(template, semantic_shape=shape, local_shape=shape,
                   padded_shape=shape, launch_shape=shape, hbm_bytes=work)


def graph_run(case, ops):
    execution = Execution(case, CostModel(case))
    plan = IterationPlan((RequestView("probe", 1, 0, True, "primitive"),))
    work = IterationWork(tuple(ops), plan, 1, 1, 1, None, "primitive", "copy_control")
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,))
    return {"duration_ms": step.completion.time, "execution": execution.summary(),
            "basis": execution.cost.summary()}


def replay(raw, source, *, memory_bytes_per_second):
    case, template, packet = probe_inputs(
        raw, source, memory_bytes_per_second=memory_bytes_per_second)
    fitted = fit_service_curve(packet)
    analytic = operator_costs(case, tuple(
        copy_at(template, work) for work in sorted({sample["work"] for sample in packet["samples"]})
    ))
    configured = replace(case, runtime=replace(case.runtime, kernels={"recipes": {
        template.recipe: {"kind": "resource_service", "model": fitted["model"],
                          "outside_domain": "analytic"},
    }}))
    estimates = operator_costs(configured, tuple(
        copy_at(template, row["operator"]["hbm_bytes"]) for row in analytic["operators"]
    ), input_ready_ms=7)
    by_work = {row["operator"]["hbm_bytes"]: row for row in estimates["operators"]}
    heldout = []
    for sample in fitted["heldout"]:
        row = by_work[sample["work"]]
        heldout.append({
            **sample, "engine_predicted_ms": row["duration_ms"],
            "engine_relative_error": row["duration_ms"] / sample["observed_ms"] - 1,
            "declared_bytes": row["declared_service"]["memory@r0:s0"],
            "serviced_bytes": row["serviced"]["memory@r0:s0"],
        })
    heldout_work = sorted({sample["work"] for sample in heldout})
    scores, concurrency = [], []
    for amount in heldout_work:
        rows = [row for row in heldout if row["work"] == amount]
        observed = median(row["observed_ms"] for row in rows)
        prediction = by_work[amount]["duration_ms"]
        scores.append({
            "work_bytes": amount, "trials": len(rows), "observed_median_ms": observed,
            "engine_predicted_ms": prediction, "median_relative_error": prediction / observed - 1,
            "trial_error_min": min(row["engine_relative_error"] for row in rows),
            "trial_error_max": max(row["engine_relative_error"] for row in rows),
        })
        first = replace(copy_at(template, amount), op_id="first", stream="first")
        second = replace(first, op_id="second", stream="second")
        concurrency.append({
            "work_per_invocation_bytes": amount, "observed_concurrency": False,
            "physical_shared_rate_bytes_per_ms": memory_bytes_per_second / 1000,
            "parallel": graph_run(configured, (first, second)),
            "serial": graph_run(configured, (first, replace(second, deps=("first",)))),
        })
    return packet, {
        "schema": "mindie-sim.memory-service-replay/1", "source": source,
        "new_device_execution": False,
        "scope": "one_retained_copy_protocol; heldout_work_on_the_same_device_and_capture",
        "hardware_projection": {
            "memory_bytes_per_second": memory_bytes_per_second,
            "rate_basis": "caller_declared_physical_ceiling; not_fit_from_copy_duration",
            "compute_fields": "unused_schema_placeholders; not_GPU_compute_capabilities",
        },
        "limits": [
            "only_one_heldout_work_size; not_independent_run_or_cross_device_validation",
            "batched_event_average; isolated_launch_and_concurrent_device_timings_not_measured",
            "retained_source_hashes_do_not_prove_loaded_binary_identity",
            "copy_service_does_not_calibrate_other_memory_kernels",
            "concurrency_is_a_resource_conservation_control; not_a_GPU_accuracy_result",
        ],
        "calibration": fitted, "heldout_engine": heldout, "scores_by_work": scores,
        "prior_affine_method": fit_resource_rate(raw["samples"]),
        "analytic_operator_costs": analytic, "calibrated_operator_costs": estimates,
        "composition_controls": concurrency,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--memory-bytes-per-second", type=float, required=True)
    parser.add_argument("--packet-output")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    packet, report = replay(read_json(args.source), args.source.as_posix(),
                            memory_bytes_per_second=args.memory_bytes_per_second)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if args.packet_output:
        emit(packet, args.packet_output)
    emit(report, args.output)
    print(report["scores_by_work"])


if __name__ == "__main__":
    main()
