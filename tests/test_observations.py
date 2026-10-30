"""Actual capture parsing, exact primitive reuse and independent validation."""
import json
from pathlib import Path
import statistics

import pytest

from mindie_sim import Case, predict
from mindie_sim.observations.calibration import import_profiler
from mindie_sim.observations.store import ObservationIndex, ProfilerCatalogue, profiler_rows
from mindie_sim.observations.validation import compare, fit_resource_rate, service_observation
from mindie_sim.prediction.cost import primitive_key
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case, linear_work


FIXTURES = Path(__file__).with_name("fixtures") / "observations"


def test_real_profiler_csv_remains_readable_with_companion_api(profiler_api):
    rows = list(profiler_rows(FIXTURES / "profiler.csv"))
    assert rows and all(row["duration_ms"] > 0 and row["kernel"] for row in rows)


def test_existing_profiler_sqlite_remains_readable_without_old_python():
    with ProfilerCatalogue(FIXTURES / "catalogue.sqlite") as catalogue:
        variants = list(catalogue.variants(limit=5))
        samples = list(catalogue.samples(variants[0]["id"]))
    assert samples and all(row["duration_ms"] > 0 and row["source"] for row in samples)
    assert variants[0]["min_ms"] <= min(row["duration_ms"] for row in samples)
    assert variants[0]["max_ms"] >= max(row["duration_ms"] for row in samples)


def test_shared_profiler_row_preserves_existing_consumer_projection(profiler_api):
    from dataclasses import replace
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.observations.kernels import kernel_rows

    capture = FIXTURES / "operator-kernels.csv"
    row = next(kernel_rows(capture))
    projected = next(profiler_rows(capture))
    assert row.start_ns == 1792684744877935379
    assert row.duration_ns == 108051
    assert row.input_shapes == ((16384, 5120), (5120,))
    assert row.output_shapes == ((16384, 5120), (16384, 1))
    assert projected == {
        "line": 2, "duration_ms": .108051, "kernel": "RmsNorm", "operator": "RmsNorm",
        "input_shapes": "16384,5120;5120", "input_dtypes": "DT_BF16;DT_BF16",
        "input_formats": "ND;ND", "output_shapes": "16384,5120;16384,1",
        "output_dtypes": "DT_BF16;FLOAT", "device": "0", "stream": "59",
        "core": "AI_VECTOR_CORE", "block_dim": "64", "boundary": "profiled_device_kernel",
    }
    assert observed_contract(row) == ("rmsnorm", (16384, 5120), "bf16", True)
    assert observed_contract(replace(row, output_shapes=None)) is None
    assert observed_contract(replace(row, output_shapes=((16384, 2560), (16384, 1)))) is None


def test_missing_or_unrecognized_matrix_dtype_does_not_authorize_a_contract(profiler_api):
    from dataclasses import replace
    from mindie_profiler.normalized import row_from_event
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.observations.kernels import kernel_rows

    row = next(row for row in kernel_rows(FIXTURES / "operator-kernels.csv")
               if row.kind == "MatMulV3")
    assert observed_contract(row) == (
        "matrix", (16384, 16384, 5120), "bf16", "bf16", "bf16", (), "")
    for descriptors in ("", "N/A;DT_BF16", "DT_UNRECOGNIZED;DT_BF16"):
        incomplete = row_from_event(replace(row.event, input_dtypes=descriptors))
        assert incomplete.input_shapes == row.input_shapes
        assert observed_contract(incomplete) is None


def test_bound_raw_samples_can_be_calibrated_without_reinterpreting_predictions():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    key = primitive_key(value, linear_work(plan, value.model, value.runtime, value.machine).ops[0], "none")
    with ProfilerCatalogue(FIXTURES / "catalogue.sqlite") as catalogue:
        variant = next(catalogue.variants(limit=1))
        samples = list(catalogue.samples(variant["id"]))
    # This explicit test binding exercises identity translation only; it does
    # not claim that this fixture's kernel implements the test's linear op.
    with PrimitiveStore(":memory:") as store:
        result = import_profiler(FIXTURES / "catalogue.sqlite", store, [
            {"variant": variant["id"], "key": key.to_dict(), "run_id": "translation",
             "split": "heldout"}])
        assert result["imported_samples"] == len(samples)
        assert store.lookup(key) is None
        assert predict(value, lower=linear_work, primitives=store)["metrics"] == predict(
            value, lower=linear_work)["metrics"]


def test_train_primitive_changes_duration_without_fabricating_resource_work():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    op = linear_work(plan, value.model, value.runtime, value.machine).ops[0]
    key = primitive_key(value, op, "none")
    before = predict(value, lower=linear_work)
    with PrimitiveStore(":memory:") as store:
        store.record(key, duration_ms=10, run_id="analytic_oracle", repeat_index=0,
                     source="independent timing oracle")
        after = predict(value, lower=linear_work, primitives=store)
    assert after["requests"][0]["ttft_ms"] == 10
    assert after["execution"]["work"] == before["execution"]["work"]
    assert after["execution"]["serviced"]["matrix@r0:s0"] == before["execution"]["serviced"]["matrix@r0:s0"]


def test_service_boundary_mismatch_does_not_become_a_calibration_or_pass():
    observed = service_observation(FIXTURES / "service.json")
    result = compare(predict(case()), observed, boundary="client_stream")
    assert not result["comparable"]
    assert result["rows"] and all(row["observed"] > 0 for row in result["rows"])
    with pytest.raises(ValueError, match="boundary_conflicts"):
        compare(predict(case()), observed, boundary="device_token_commit")


def test_actual_speculative_capture_uses_tokens_instead_of_sse_chunks_for_tpot():
    raw = json.loads((FIXTURES / "service.json").read_text())
    observed = service_observation(FIXTURES / "service.json")
    requests = raw["per_request"]
    expected = statistics.mean(row["decode_ms"] / (row["completion_tokens"] - 1) for row in requests)
    assert observed["tpot_ms"]["mean"] == pytest.approx(expected)
    assert observed["tpot_ms"]["mean"] < raw["tpot_ms"]["mean"] / 3
    assert observed["reported_metrics"]["tpot_ms"] == raw["tpot_ms"]
    assert observed["e2e_ms"]["mean"] == pytest.approx(
        statistics.mean(row["ttft_ms"] + row["decode_ms"] for row in requests))
    assert observed["output_tps"] == pytest.approx(
        sum(row["completion_tokens"] for row in requests) / raw["wall_s"])


def test_capture_without_usage_does_not_substitute_chunks_or_requested_token_counts(tmp_path):
    raw = json.loads((FIXTURES / "service.json").read_text())
    for row in raw["per_request"]:
        row["completion_tokens"] = None
    path = tmp_path / "without-usage.json"
    path.write_text(json.dumps(raw))
    observed = service_observation(path)
    assert observed["tpot_ms"] == {}
    assert observed["output_tps"] is None
    assert observed["reported_metrics"]["tpot_ms"] == raw["tpot_ms"]
    assert observed["ttft_ms"] == raw["ttft_ms"]


def test_historical_source_index_preserves_hashes_and_source_roles(tmp_path):
    with ObservationIndex(tmp_path / "observations.sqlite") as index:
        index.scan(FIXTURES)
        index.attach_profiler_catalogue(FIXTURES / "catalogue.sqlite")
        rows = index.find_sources("profiler.csv")
        assert rows[0]["kind"] == "profiler" and len(rows[0]["sha256"]) == 64
        assert index.inventory()["profiler_catalogue"]["files"] == 1


def test_rate_fit_holds_out_measurements_and_reproduces_the_saved_gpu_experiment():
    root = Path(__file__).parents[1] / "data" / "measurements" / "gpu"
    raw = json.loads(next(root.glob("*.stdout")).read_text())
    # The fixture is the actual 4090 Driver API capture, not generated timings.
    samples = raw["samples"]
    fitted = fit_resource_rate(samples)
    changed = [dict(sample, duration_ms=sample["duration_ms"] * 2)
               if sample["split"] == "heldout" else sample for sample in samples]
    assert fitted["bytes_per_ms"] == fit_resource_rate(changed)["bytes_per_ms"]
    assert len(fitted["heldout"]) == 5
    assert max(abs(row["relative_error"]) for row in fitted["heldout"]) < .08
