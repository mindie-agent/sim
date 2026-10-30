"""Refresh the Sim body scoreboard from completed profiler comparisons.

[Role] Retain compact, source-identified body evidence without rerunning it.
[Invariants] Whole-body errors keep the strict relative criterion and their own
  runtime. Superseded predictions never count as passes; activity is not span.
[Timeouts & Locks] Stream selected rows; no children, network or locks.
[Side-Effect Policy] Writes the requested compact receipt and Markdown section.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import ijson


BEGIN = "<!-- current-body-evidence:start -->"
END = "<!-- current-body-evidence:end -->"


def compact_row(item):
    row = {key: item.get(key) for key in (
        "model_group", "case_id", "variant_id", "receipt", "source_identity",
        "runtime_identity", "geometry", "declared_runtime", "display_status",
        "body_activity_ms", "body_internal_gap_ms", "error", "prediction_failure",
        "calibration_origin")}
    evidence = item.get("calibration_evidence") or {}
    # Histograms and per-kernel observations stay in the identified source
    # receipt. Keep evaluation role and parameter identity in this compact view.
    row["calibration_evidence"] = {key: evidence[key] for key in (
        "evaluation_role", "evaluation_relation", "source_identity", "body_residual_used",
        "parameter_assignment", "scope", "contract_admission_changed",
        "service_regime_declaration_path") if key in evidence}
    if "declared_service_regimes" in evidence:
        row["calibration_evidence"]["service_regime_ids"] = [
            rule["id"] for rule in evidence["declared_service_regimes"]]
    row["comparison"] = item["comparison"]
    comparison = row["comparison"]
    if item.get("display_status") == comparison["status"] == "compared":
        observed = comparison["observed_ms"]["mean"]
        delta = comparison["predicted_ms"] - observed
        row.update(signed_gap_ms=delta, absolute_gap_ms=abs(delta),
                   relative_error=delta / observed,
                   strict_body_15pct=abs(delta / observed) <= 0.15)
    return row


def retain(source):
    path = Path(source).resolve()
    # The producer also retains large raw/invalidated populations. Only selected
    # rows and small summary fields belong to this consumer. Multiple bounded
    # passes avoid materializing unrelated history in the control process.
    def items(prefix, source_path=path):
        with source_path.open("rb") as stream:
            yield from ijson.items(stream, prefix, use_float=True)

    packet = {key: next(items(key)) for key in ("summary", "limitations")}
    publication = next(items("publication_format"), None)
    policy_path = path
    archived = None
    if publication == "mindie-priority-scoreboard-compact/1":
        archived = next(items("full_evidence"))
        policy_path = Path(archived["path"])
        digest = sha256()
        with policy_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if (digest.hexdigest() != archived["sha256"]
                or policy_path.stat().st_size != archived["size_bytes"]):
            raise ValueError("archived_scoreboard_identity_mismatch")
    for key in ("current_selection_policy", "coverage_status"):
        packet[key] = next(items(key, policy_path))
    rows = [compact_row(item) for item in items("current_body_comparisons.item")]
    latest = [compact_row(item) for item in items("latest_body_attempts.item")]
    compared = [row for row in rows if "strict_body_15pct" in row]
    counts = {"compared": len(compared),
              "passed": sum(row["strict_body_15pct"] for row in compared),
              "other_selected_rows": len(rows) - len(compared)}
    if (counts["compared"] != packet["summary"]["current_body_compared"]
            or counts["passed"] != packet["summary"]["current_body_passed"]):
        raise ValueError("source_current_body_summary_disagrees_with_recomputed_rows")
    latest_counts = None
    if "latest_body_attempt_count" in packet["summary"]:
        latest_counts = {
            "count": len(latest),
            "completed": sum("strict_body_15pct" in row for row in latest),
            "failed": sum(row["comparison"]["status"] == "failed" for row in latest),
            "passed": sum(row.get("strict_body_15pct", False) for row in latest),
        }
        if any(value != packet["summary"]["latest_body_attempt_" + key]
               for key, value in latest_counts.items()):
            raise ValueError("source_latest_attempt_summary_disagrees_with_recomputed_rows")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"schema": "mindie-sim.retained-body-scoreboard/1",
            "source": {"path": str(path), "sha256": digest.hexdigest(),
                       "size_bytes": path.stat().st_size},
            "archived_full_evidence": archived,
            "selection_policy": packet["current_selection_policy"],
            "source_summary": packet["summary"], "current_counts": counts,
            "latest_attempt_counts": latest_counts, "latest_attempts": latest,
            "coverage_status": packet["coverage_status"], "rows": rows,
            "boundary": "target_body_device_graph; conditional representative-rank alignment",
            "limitations": packet["limitations"]}


def section(report, receipt_name):
    counts = report["current_counts"]
    lines = [BEGIN, "## October 30 current Decode-body evidence", "",
             f"**{counts['passed']}/{counts['compared']}** selected completed comparisons satisfy",
             "the strict 15% body criterion. These are the latest completed arms for",
             "each retained source/geometry, **not a replay of every case at current HEAD**.",
             "Runtime hashes below keep that distinction visible. Selection remains",
             "conditional on the declared topology, context and rank coverage.", ""]
    attempts = report.get("latest_attempt_counts")
    if attempts is not None:
        lines.extend([
            f"Latest attempts: **{attempts['completed']}/{attempts['count']} completed**, "
            f"**{attempts['failed']} failed**, **{attempts['passed']} passed**.",
            "An older completed prediction retained below does not turn a newer failed attempt into success.", ""])
        failed = [row for row in report["latest_attempts"]
                  if row["comparison"]["status"] == "failed"]
        if failed:
            lines.extend(["| Latest failed attempt | Reason | Budget s | Runtime package |",
                          "| --- | --- | ---: | --- |"])
            for row in failed:
                failure = row.get("prediction_failure") or {}
                reasons = row["comparison"].get("reasons", [])
                runtime = (row.get("runtime_identity") or {}).get("package_sha256", "unknown")[:12]
                lines.append(f"| {row['case_id']} / {row['variant_id']} | "
                             f"{', '.join(reasons)} | {failure.get('timeout_s', 'n/a')} | `{runtime}` |")
            lines.append("")
    lines.extend([
             "The HF-wrapper precision bug invalidated the earlier Kimi predictions.",
             "Superseded predictions are excluded; their measured populations remain.",
             "The calibration column preserves the source's evaluation role;",
             "`in_sample` is a calibrated comparison, not independent validation.",
             "Micro-kernel tolerance does not relax body acceptance. Old completed",
             "arms below this section remain historical evidence.", "",
             "| Case / variant | Observed mean ms | Predicted ms | Signed gap ms | Absolute gap ms | Relative | Pass | Calibration | Runtime package |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"])
    rows = sorted((r for r in report["rows"] if "strict_body_15pct" in r),
                  key=lambda r: -r["absolute_gap_ms"])
    for row in rows:
        comp = row["comparison"]
        identity = row["case_id"] + (f" / {row['variant_id']}" if row["variant_id"] else "")
        runtime = row["runtime_identity"].get("package_sha256", "unknown")[:12]
        calibration = (row.get("calibration_evidence", {}).get("evaluation_role")
                       or row.get("calibration_evidence", {}).get("evaluation_relation")
                       or row.get("calibration_origin") or "not declared")
        lines.append(f"| {identity} | {comp['observed_ms']['mean']:.3f} | "
                     f"{comp['predicted_ms']:.3f} | {row['signed_gap_ms']:+.3f} | "
                     f"{row['absolute_gap_ms']:.3f} | {row['relative_error']:+.2%} | "
                     f"{'yes' if row['strict_body_15pct'] else 'no'} | {calibration} | `{runtime}` |")
    lines.extend(["", f"[Compact receipt]({receipt_name}) retains source hashes, per-case",
                  "geometry, mean/P50, assumptions and original comparison receipts.",
                  "Measured family activity and isolated operator sums are separate",
                  "diagnostics: neither is added to body span. Communication wrapper",
                  "and HCOM stream activity must not be counted twice. A symmetric-DP",
                  "input declaration does not establish that every captured DP rank",
                  "had a request; inspect each row's retained assumptions.",
                  END])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    report = retain(args.source)
    receipt, markdown = Path(args.receipt), Path(args.markdown)
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = markdown.read_text()
    block = section(report, receipt.name)
    if BEGIN in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        title, rest = text.split("\n", 1)
        text = title + "\n\n" + block + "\n" + rest
    markdown.write_text(text)
    print(json.dumps(report["current_counts"]))


if __name__ == "__main__":
    main()
