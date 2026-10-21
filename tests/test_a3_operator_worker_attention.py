"""CPU-only contract tests for the native FA/RoPE capability worker.

No torch/NPU is imported or mocked as native support.  Tests cover pure CPU
reference math, causal/GQA/RoPE invariants, manifest planning, dry-run isolation
and truthful partial accounting.
"""
import builtins
import copy
import json
import math
from pathlib import Path

import pytest

from mindie_sim.a3_operator_worker_attention import (
    AttentionInputError,
    AttentionRecipeMixin,
    BudgetExceeded,
    Runner,
    attention_ref,
    build_plan,
    causal_mask,
    compare,
    dry_run_payload,
    finite,
    load_manifest,
    new_results,
    refresh,
    rope_ref,
    visible_mask,
)
from mindie_sim.a3_operator_worker_attention import main

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples" / "operator-attention" / "capability.manifest.json"


def _doc():
    return load_manifest(MANIFEST)[0]


def _plan():
    return build_plan(_doc())


def test_bottom_right_causal_and_full_visible_masks():
    assert causal_mask(2, 3) == [[False, False, True], [False, False, False]]
    assert causal_mask(3, 3) == [
        [False, True, True], [False, False, True], [False, False, False]]
    decode = visible_mask(1, 128)
    assert len(decode) == 1 and len(decode[0]) == 128
    assert not any(decode[0])
    assert sum(row.count(True) for row in causal_mask(32, 32)) == 496
    assert sum(row.count(True) for row in causal_mask(1, 128)) == 0


def test_gqa_reference_repeats_kv_heads_and_softmaxes():
    q = [[[[1.0, 0.0]], [[0.0, 1.0]]]]
    k = [[[[1.0, 0.0]]]]
    v = [[[[3.0, 4.0]]]]
    mask = [[[False]], [[False]]]
    out = attention_ref(q, k, v, mask, 1.0)
    assert out == [[[[3.0, 4.0]], [[3.0, 4.0]]]]
    assert finite(out) and compare(out, out, 0.0, 0.0)["passed"]

    k2 = [[[[1.0, 0.0], [0.0, 1.0]]]]
    v2 = [[[[1.0, 2.0], [3.0, 4.0]]]]
    mask2 = [[[False, False]], [[False, False]]]
    out2 = attention_ref(q, k2, v2, mask2, 1.0)
    p = math.e / (1.0 + math.e)
    assert abs(out2[0][0][0][0] - (p * 1.0 + (1 - p) * 3.0)) < 1e-6
    assert abs(out2[0][1][0][1] - ((1 - p) * 2.0 + p * 4.0)) < 1e-6


def test_hq3_hkv1_q2_kv3_bottom_right_mask_and_repeated_heads():
    head = [[1.0, 0.0], [0.0, 1.0]]
    q = [[head, head, head]]
    k = [[[[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]]]
    v = [[[[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]]]
    mask = causal_mask(2, 3)
    out = attention_ref(q, k, v, mask, 1.0)
    assert mask == [[False, False, True], [False, False, False]]
    assert all(abs(out[0][h][0][0] - out[0][0][0][0]) < 1e-6 for h in range(3))
    assert out[0][0][0][0] > 9.9
    assert abs(out[0][0][1][0] - 37.0) < 1e-5
    top_left = [[False, True, True], [False, False, True]]
    wrong = attention_ref(q, k, v, top_left, 1.0)
    assert abs(wrong[0][0][0][0] - 1.0) < 1e-6
    assert abs(wrong[0][0][1][0] - 5.5) < 1e-6


def test_native_correctness_dispatch_uses_declared_stream():
    class Ctx:
        def __init__(self): self.entered = 0
        def __enter__(self): self.entered += 1; return self
        def __exit__(self, exc_type, exc, tb): return None
    class Stream:
        def __init__(self): self.syncs = 0
        def synchronize(self): self.syncs += 1
    class Npu:
        def __init__(self, ctx): self.ctx = ctx
        def stream(self, stream): return self.ctx
    class Torch:
        def __init__(self, ctx): self.npu = Npu(ctx)
    class Mixin(AttentionRecipeMixin):
        pass
    ctx = Ctx(); stream = Stream(); mixin = Mixin()
    mixin.torch = Torch(ctx); mixin.stream = stream
    def op():
        assert ctx.entered == 1
        return ("native-return",)
    assert mixin._run_native(op) == ("native-return",)
    assert ctx.entered == 1 and stream.syncs == 1


def test_cleanup_budget_is_checked_after_synchronization():
    class Clock:
        def clean(self, stage):
            if stage == "after_synchronize":
                raise BudgetExceeded(stage, 61.0, 60.0)
    class Stream:
        def __init__(self): self.syncs = 0
        def synchronize(self): self.syncs += 1
    class Npu:
        def __init__(self): self.syncs = 0
        def synchronize(self): self.syncs += 1
    class Torch:
        def __init__(self): self.npu = Npu()
    runner = Runner(); runner.torch = Torch(); runner.stream = Stream()
    info = runner.cleanup(Clock())
    assert info["status"] == "failed" and info["ok"] is False
    assert info["synchronized"] is True
    assert info["error"]["type"] == "BudgetExceeded"
    assert runner.stream.syncs == 1 and runner.torch.npu.syncs == 1


def test_half_rope_reference_uses_rotated_halves_and_rounded_planes():
    x = [[[[1.0, 2.0, 3.0, 4.0]]]]
    cos = [[[[1.0, 1.0, 1.0, 1.0]]]]
    sin = [[[[0.0, 0.0, 0.0, 1.0]]]]
    assert rope_ref(x, cos, sin) == [[[[1.0, 2.0, 3.0, 6.0]]]]
    zero = [[[[0.0, 0.0, 0.0, 0.0]]]]
    assert rope_ref(x, cos, zero) == x
    assert not finite([[float("nan")]])
    assert not compare([[1.0]], [[2.0]], 0.0, 0.0)["passed"]


def test_capability_plan_geometry_counts_and_scope():
    assert _doc()["kind"] == "a3_attention_rope_manifest"
    plan = _plan()
    assert plan["kind"] == "a3_attention_rope_capability"
    c = plan["counts"]
    assert c == {"cases": 4, "trials": 2, "samples_per_trial": 30,
                 "planned_condition_executions": 8, "planned_raw_blocks": 240,
                 "planned_warmup_events_per_trial": 10,
                 "planned_warmup_events_per_case": 20}
    assert plan["protocol"]["budgets_seconds"] == {
        "case": 24.0, "setup": 60.0, "global_work": 240.0, "cleanup": 60.0}
    assert plan["source_contract"]["sha256"] == (
        "0b2265792c52bae1cc4fc4946ea72f02fd3fd1e6f1024e43cfbe2251a04f604c")
    assert plan["source_contract"]["schema_lines"] == {
        "npu_fusion_attention": 323, "npu_rotary_mul": 613}
    cases = {case["case_id"]: case for case in plan["cases"]}
    prefill = cases["fa-prefill-b1-hq3-hkv1-q32-kv32-d256-causal"]
    decode = cases["fa-decode-b1-hq3-hkv1-q1-kv128-d256-fullvisible"]
    assert prefill["shape"]["q_heads"] == 3 and prefill["shape"]["kv_heads"] == 1
    assert prefill["shape"]["qk_dim"] == 256 and prefill["scale"] == 0.0625
    assert prefill["mask"] == "causal_bottom_right"
    assert prefill["mask_true_count_expanded"] == 3 * 496
    assert decode["mask"] == "full_visible"
    assert decode["mask_true_count_expanded"] == 0
    assert all(case["target_match"] is False and case["split"] == "train"
               for case in plan["cases"])
    rope = cases["rope-half-b1-s8-h3-d64"]
    assert rope["rotary_mode"] == "half" and rope["shape"]["dim"] == 64
    assert rope["positions"] == {"start": 0, "end": 7}
    assert "partial_rotary_factor" in rope["source"]
    assert all("primitive_scope" in case["source"] for case in plan["cases"])


def test_valid_manifest_without_source_contract_keeps_empty_or_supplied_provenance(tmp_path):
    doc = _doc()
    del doc["source_contract"]
    path = tmp_path / "no-provenance.manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    plan = dry_run_payload(path)["plan"]
    assert plan["source_contract"] == {}
    result = new_results(plan, "no-provenance", "npu:0", str(path), "0" * 64)
    assert result["manifest"]["source_contract"] == {}
    assert result["progress"]["requested_cases"] == 4

    doc["source_contract"] = {"attribution": "optional only", "note": {"k": 1}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    plan = dry_run_payload(path)["plan"]
    assert plan["source_contract"] == doc["source_contract"]
    result = new_results(plan, "partial-provenance", "npu:0", str(path), "0" * 64)
    assert result["manifest"]["source_contract"] == doc["source_contract"]
    assert "sha256" not in result["manifest"]["source_contract"]


def test_manifest_rejections_are_pre_execution():
    base = _doc()
    mutations = [
        lambda d: d.update(kind="wrong"),
        lambda d: d.pop("kind"),
        lambda d: d["cases"][0].update(target_match=True),
        lambda d: d["cases"][0].update(split="mixed"),
        lambda d: d["cases"][0].update(scale=0.1),
        lambda d: d["cases"][0]["shape"].update(kv_heads=2),
        lambda d: d["cases"][2]["shape"].update(dim=65),
        lambda d: d["cases"][0].update(layout="BSH"),
        lambda d: d["cases"][0].update(mask="window"),
        lambda d: d["expectations"].update(planned_raw_blocks=239),
    ]
    for mutation in mutations:
        doc = copy.deepcopy(base)
        mutation(doc)
        with pytest.raises(AttentionInputError):
            build_plan(doc)


def test_manifest_declared_heldout_is_preserved_not_relabelled(tmp_path):
    doc = _doc()
    heldout_id = doc["cases"][1]["case_id"]
    doc["cases"][1]["split"] = "heldout"
    path = tmp_path / "heldout.manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    plan = dry_run_payload(path)["plan"]
    splits = {case["case_id"]: case["split"] for case in plan["cases"]}
    assert splits[heldout_id] == "heldout"
    assert sum(1 for value in splits.values() if value == "heldout") == 1
    assert splits[doc["cases"][0]["case_id"]] == "train"
    result = new_results(plan, "heldout-test", "npu:0", str(path), "0" * 64)
    by_id = {case["case_id"]: case for case in result["cases"]}
    assert by_id[heldout_id]["split"] == "heldout"
    assert by_id[heldout_id]["geometry"]["split"] == "heldout"
    assert by_id[doc["cases"][0]["case_id"]]["split"] == "train"
    refresh(result)
    assert by_id[heldout_id]["split"] == "heldout"
    assert result["requested"]["cases"] == 4


def test_new_results_accounting_is_truthful_for_partial_failures():
    plan = _plan()
    doc = new_results(plan, "cpu-only", "npu:0", str(MANIFEST), "0" * 64)
    assert doc["kind"] == "a3_attention_rope_capability"
    assert doc["analytic_ms"] is None
    assert doc["progress"]["requested_cases"] == 4
    assert doc["progress"]["raw_blocks"] == 0
    doc["cases"][0]["status"] = "measured"
    doc["cases"][0]["trials"] = [{"samples": [{}, {}]}]
    doc["cases"][1]["status"] = "failed"
    doc["cases"][2]["status"] = "pending"
    doc["cases"][3]["status"] = "not_run"
    refresh(doc)
    p = doc["progress"]
    assert p["requested_cases"] == 4 and p["completed_cases"] == 1
    assert p["measured_cases"] == 1 and p["failed_cases"] == 1
    assert p["pending_cases"] == 1 and p["not_run_cases"] == 1
    assert p["raw_blocks"] == 2 and p["planned_raw_blocks"] == 240
    assert doc["status"] == "completed_with_failures"
    for case in doc["cases"]:
        assert case["analytic_ms"] is None
        assert case["target_match"] is False


def test_dry_run_imports_no_torch_and_writes_no_results(monkeypatch, tmp_path,
                                                        capsys):
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch" or name == "torch_npu" or name.startswith("torch."):
            raise AssertionError(f"dry-run imported {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    payload = dry_run_payload(MANIFEST)
    assert payload["valid"] is True
    assert payload["writes_results"] is False
    assert payload["plan"]["counts"]["planned_raw_blocks"] == 240
    results = tmp_path / "must-not-exist.json"
    code = main(["--manifest", str(MANIFEST), "--results", str(results),
                 "--run-id", "dry", "--device", "npu:0", "--dry-run"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["plan"]["cases"][1]["mask"] == "full_visible"
    assert not results.exists()
    assert main(["--manifest", str(MANIFEST), "--results", str(results),
                 "--run-id", "bad", "--device", "npu:1", "--dry-run"]) == 2
