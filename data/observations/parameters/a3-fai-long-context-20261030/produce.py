"""Freeze native regular-query FAI service from shorter contexts only.

[Role] Generate an optional joint-kernel response and separate validation ledger.
[Invariants] No Body duration, model name or 126K timing enters coefficients.
  Per-owner waves follow the retained regular FAI host schedule. The response
  does not identify cache hits, physical HBM traffic or individual pipe rates.
  Experiment roles remain unassigned; these are retrospective analysis splits.
[Timeouts & Locks] Bounded local census; no processes, devices or waits.
[Side-Effect Policy] Replace two small parameter/evidence files in this directory.
"""
import hashlib
import json
import math
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
SOURCE = "ops-transformer/a3925fd49cd256ad478d2faf983bf2961b8ae2c8"


def geometry(row):
    shapes = row["native_abi"]["Input Shapes"].split(";")
    heads = int(shapes[0].split(",")[1])
    kv_heads = int(shapes[1].split(",")[2]) // 256
    return heads // kv_heads, kv_heads, math.ceil(row["active_batch"] * kv_heads / 20)


def affine(points):
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    slope = max(0, sum((x-mx)*(y-my) for x, y in points)
                / sum((x-mx)**2 for x, _ in points))
    fixed = max(0, my-slope*mx)
    if fixed == 0:
        slope = sum(x*y for x, y in points) / sum(x*x for x, _ in points)
    return fixed, slope


def produce():
    census_path = DIRECTORY / "native-census.json"
    census = json.loads(census_path.read_text())["records"]
    rules, comparisons, models = [], [], []
    for group, train_folder in ((4, "qwen35_35b_tp4_128k_20261030"),
                                (6, "qwen36_27b_tp2_128k_20261030")):
        rows = [r for r in census if geometry(r)[0] == group
                and r["active_batch"] * geometry(r)[1] > 16]
        training = [r for r in rows if train_folder in Path(r["source"]).parts
                    and r["analysis_role"] == "candidate_operator_calibration"]
        fixed, slope = affine([(r["declared_context"], r["median_us"]/geometry(r)[2])
                               for r in training])
        provenance = dict(
            kind="conditional_native_operator_response", source=SOURCE,
            boundary="inclusive BF16 ND FusedInferAttentionScore device call",
            training=[dict(source=r["source"], sha256=r["sha256"],
                           first_line=r["first_line"], context=r["declared_context"],
                           batch=r["active_batch"], waves=geometry(r)[2],
                           median_us=r["median_us"]) for r in training],
            limitations=["retrospective split; original roles unassigned",
                         "block table and actual seq_lens values unavailable",
                         "prompt length excludes generated suffix",
                         "joint service does not identify bandwidth or L2 hit rate",
                         "129K endpoint is a declared extrapolation, not a measured sample",
                         "unmodified hardware only; not a per-pipe bandwidth sweep model"])
        models.append(dict(query_head_group=group, per_owner_fixed_us=fixed,
                           per_owner_kv_row_us=slope, provenance=provenance))
        for kv_heads in ((1,) if group == 4 else (1, 2)):
            for waves in range(1, 5):
                lower = max((17+kv_heads-1)//kv_heads,
                            ((waves-1)*20)//kv_heads+1)
                upper = min(64, waves*20//kv_heads)
                if lower > upper:
                    continue
                rules.append(dict(
                    id=f"a3-fai-regular-g{group}-kv{kv_heads}-wave{waves}",
                    match={"hardware": "Ascend910_9362", "core_count": 20,
                           "phase": "decode", "recipe": "flash_attention_decode",
                           "metadata.gqa_decode.query_dtype": "bf16",
                           "metadata.gqa_decode.kv_dtype": "bf16",
                           "metadata.gqa_decode.head_dim": 256,
                           "metadata.gqa_decode.query_heads": group*kv_heads,
                           "metadata.gqa_decode.kv_heads": kv_heads,
                           "metadata.gqa_decode.uniform_query_tokens": 4,
                           "metadata.gqa_decode.cp": 1,
                           "metadata.gqa_decode.window": 0},
                    coordinates={"batch": {"path": "metadata.gqa_decode.requests", "reduce": "len"},
                                 "context": {"path": "metadata.gqa_decode.uniform_context_tokens"}},
                    domain={"batch": {"min": lower, "max": upper},
                            "context": {"min": 8512, "max": 129024}},
                    response={"fixed_us": waves*fixed,
                              "terms": [{"coordinate": "context", "us_per_unit": waves*slope}]},
                    execution={"kind": "joint_native", "resource": "native_call_service",
                               "reference_rate": 1e9, "native_kernel": "FusedInferAttentionScore"},
                    provenance=provenance))
        training_keys = {(r["source"], r["first_line"]) for r in training}
        for r in rows:
            pred = geometry(r)[2] * (fixed + slope*r["declared_context"])
            role = ("operator_calibration" if (r["source"], r["first_line"]) in training_keys
                    else "separate_context_and_or_geometry_validation")
            comparisons.append(dict(source=r["source"], sha256=r["sha256"],
                first_line=r["first_line"], original_split=r["original_split"], analysis_role=role,
                query_head_group=group, kv_heads=geometry(r)[1], waves=geometry(r)[2],
                batch=r["active_batch"], declared_context=r["declared_context"],
                predicted_us=pred, observed_median_us=r["median_us"],
                observed_mean_us=r["mean_us"], median_relative_error=pred/r["median_us"]-1,
                mean_relative_error=pred/r["mean_us"]-1,
                within_15pct=abs(pred/r["median_us"]-1) <= .15))
    (DIRECTORY / "parameters.json").write_text(json.dumps({"service_regimes": rules}, indent=2)+"\n")
    report = dict(census_sha256=hashlib.sha256(census_path.read_bytes()).hexdigest(),
                  models=models, comparisons=comparisons)
    (DIRECTORY / "operator-validation.json").write_text(json.dumps(report, indent=2)+"\n")
    for role in sorted({c["analysis_role"] for c in comparisons}):
        selected = [c for c in comparisons if c["analysis_role"] == role]
        print(role, sum(c["within_15pct"] for c in selected), "/", len(selected))


if __name__ == "__main__":
    produce()
