"""Apply frozen native FAI parameters through complete ordinary Body prediction.

[Role] Compare unchanged calibration against independently retained Body windows.
[Invariants] No Body measurement is used to generate any response parameter.
  Keep analytic source declarations and out-of-domain behavior unchanged.
[Timeouts & Locks] One process, each prediction 90 seconds, no capture reads.
[Side-Effect Policy] Write conditional case inputs and compact validation JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from mindie_sim import predict

HERE = Path(__file__).resolve().parent
PREFIXES = ("a3-qwen35-tp4-decode-", "a3-qwen36-tp2-decode-",
            "a3-qwen36-tp4-decode-", "a3-qwen38-tp2-decode-", "a3-qwen38-tp4-decode-")


def timed_out(signum, frame):
    raise TimeoutError("native_fai_body_90_seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = args.profiler_root / "examples/structure/results/decode.json"
    report = json.loads(source.read_text())
    parameter_file = HERE / "parameters.json"
    parameters = json.loads(parameter_file.read_text())["service_regimes"]
    output = ROOT / "examples/cases/longctx-decode-20261030/a3-fai-regular"
    output.mkdir(exist_ok=True)
    destination = HERE / "body-validation.json"
    rows, written = [], {}
    if args.resume and destination.exists():
        old = json.loads(destination.read_text())
        assert old["parameter_sha256"] == hashlib.sha256(parameter_file.read_bytes()).hexdigest()
        rows = old["cases"]
    completed = {r["case_id"] for r in rows if r["status"] == "completed"}
    selected = [r for r in report["cases"] if r["case_id"].startswith(PREFIXES)
                and r["geometry"]["tokens_per_sequence"] == 4
                and r["geometry"]["context_tokens"] == 126000
                and r["geometry"]["sequences_per_replica"] in (12, 24, 64)]
    signal.signal(signal.SIGALRM, timed_out)
    for observed in selected[:args.limit]:
        if observed["case_id"] in completed:
            continue
        key = observed["model_case"]["path"]
        if key not in written:
            case_path = args.profiler_root / "examples/structure" / key
            source_data = case_path.read_bytes()
            assert hashlib.sha256(source_data).hexdigest() == observed["model_case"]["sha256"]
            raw = json.loads(source_data)
            raw.pop("name", None)
            device = raw["machine"].setdefault("device", {})
            device.setdefault("rates", {})["native_call_service"] = 1e9
            device.setdefault("resource_units", {})["native_call_service"] = 1
            device.setdefault("work_units", {})["native_call_service"] = "busy_ns"
            kernels = raw["runtime"].setdefault("kernels", {})
            kernels["service_regimes"] = kernels.get("service_regimes", []) + parameters
            target = output / Path(key).name
            target.write_text(json.dumps(raw, indent=2)+"\n")
            written[key] = raw, target
        raw, case_path = written[key]
        geometry = observed["geometry"]
        start = time.monotonic()
        entry = dict(case_id=observed["case_id"], geometry=geometry,
                     case_path=str(case_path.relative_to(ROOT)),
                     case_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
                     canonical_input=observed["model_case"],
                     baseline_predicted_ms=observed["comparison"].get("predicted_ms"))
        signal.alarm(90)
        try:
            result = predict(raw, scope="body", phase="decode", tokens=4, context=126000,
                             sequences=geometry["sequences_per_replica"], event_limit=0)
            measured = observed["comparison"]["observed_ms"]["mean"]
            duration = result["body_forward_ms"]
            entry.update(status="completed", predicted_ms=duration, observed_ms=measured,
                         relative_error=duration/measured-1,
                         passed=abs(duration/measured-1) <= .15,
                         service_regimes=result.get("parameters", {}).get("service_regimes", []),
                         execution=result["execution"])
        except TimeoutError as exc:
            entry.update(status="timed_out", error=str(exc))
        finally:
            signal.alarm(0)
        entry["wall_seconds"] = time.monotonic()-start
        rows.append(entry)
        print(entry["case_id"], entry["status"], entry.get("predicted_ms"),
              entry.get("relative_error"), flush=True)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        summary = dict(runtime_commit=revision,
                       report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       parameter_sha256=hashlib.sha256(parameter_file.read_bytes()).hexdigest(),
                       boundary="target_body_device_graph", cases=rows)
        destination.write_text(json.dumps(summary, indent=2)+"\n")
    return 1 if any(r["status"] != "completed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
