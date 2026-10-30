"""Public structural coverage separate from prediction accuracy.

[Role] Carry configuration omissions to every public prediction boundary.
[Invariants] Partial costs describe only modeled components, never an accuracy
  claim. Missing measurements neither change coverage nor prevent execution.
[Timeouts & Locks] Constant-size metadata reads; no I/O or locks.
[Side-Effect Policy] Returns a new JSON-ready mapping.
"""


def model_coverage(model):
    metadata = dict(model.metadata)
    coverage = metadata.get("model_coverage", "complete")
    return {
        "model_coverage": coverage,
        "unmodeled_features": list(metadata.get("unmodeled_features", ())),
        "degraded_submodules": list(metadata.get("degraded_submodules", ())),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if coverage == "partial" else "declared_model_structure"),
    }


def combine_coverage(models):
    reports = [model_coverage(model) for model in models]
    partial = any(report["model_coverage"] == "partial" for report in reports)
    return {
        "model_coverage": "partial" if partial else "complete",
        "unmodeled_features": sorted({gap for report in reports for gap in report["unmodeled_features"]}),
        "degraded_submodules": sorted({gap for report in reports for gap in report["degraded_submodules"]}),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if partial else "declared_model_structure"),
    }
