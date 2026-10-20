"""Standalone A3 native FA/RoPE capability worker (dry-run and device run).
[Role]
  Build exact npu_fusion_attention/npu_rotary_mul capability cases, full CPU
  checks and one-stream timing. --dry-run imports no torch and writes nothing.
[Invariants]
  * FA: installed pre_tockens/next_tockens spelling, BNSD, explicit contiguous
    bool mask, keep_prob=1, sparse_mode=0, GQA head_num, scale=1/sqrt(D);
    bottom-right causal or Q1/KV128 fully visible; no torch fallback.
  * FA returns exactly 7 items; only output0 is interpreted; actual auxiliary
    types/shapes/dtypes are recorded without meaning. Dispatch errors pending;
    malformed/nonfinite/wrong output failed.
  * RoPE: BSND npu_rotary_mul(x,cos,sin,'half') with fully materialized
    contiguous BF16 cos/sin equal to x shape; cat(-x2,x1); no mRoPE/alias claim.
  * Inputs reused; native outputs allocate (no .out). Only latest output is
    held; previous cleared. Reference/copies are untimed.
  * CPU-submit/device-event/wall clocks per call; precreated events; analytic_ms
    null; no calibrated/validated performance label.
  * Budgets 24/60/240/60s; supervisor >=370s; no lease/network/queue here.
  * Manifest-declared split in {train,heldout} is preserved exactly; no split
    inference or relabelling.  Per-case partial persistence; nonzero for partial/
    failed/pending or cleanup/persistence errors; no retry.
[Timeouts & Locks]
  Cooperative checks, no locks/threads; outer supervisor bounds blocked calls.
[Side-Effect Policy]
  Dry-run: no torch, no writes. Real: torch after validation, NPU allocations,
  atomic results-path writes only.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
MANIFEST_KIND = "a3_attention_rope_manifest"
OPERATORS = ("fusion_attention", "rotary_mul")
CEILINGS = {"case": 24.0, "setup": 60.0, "global_work": 240.0, "cleanup": 60.0}
SUPERVISOR_SECONDS_MIN = 370.0
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
class AttentionInputError(ValueError):
    def __init__(self, errors, path=None):
        self.errors, self.path = list(errors), path
        super().__init__((f"{path}: " if path else "") + "; ".join(self.errors))
class PendingBackend(RuntimeError):
    """Installed native primitive unavailable or dispatch rejected."""
class BadNativeOutput(RuntimeError):
    """Malformed, non-finite or incorrect native output."""
class PersistenceFailure(RuntimeError):
    pass
class TrialFailure(RuntimeError):
    def __init__(self, trial, cause):
        self.trial, self.cause = trial, cause
        super().__init__(f"trial failed: {cause}")
class BudgetExceeded(RuntimeError):
    def __init__(self, stage, elapsed, limit):
        self.stage, self.elapsed, self.limit = stage, float(elapsed), float(limit)
        super().__init__(f"{type(self).__name__} {stage}: {elapsed:.6f}s > {limit:.6f}s")
class GlobalWorkBudgetExceeded(BudgetExceeded): pass
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
def exception_info(exc):
    return {"type": type(exc).__name__, "module": type(exc).__module__,
            "message": str(exc), "repr": repr(exc), "retry": "not_attempted",
            "traceback": "".join(traceback.format_exception(type(exc), exc,
                                                             exc.__traceback__))}
def causal_mask(q, kv):
    return [[key > kv - q + query for key in range(kv)] for query in range(q)]
def visible_mask(q, kv):
    return [[False] * kv for _ in range(q)]
def attention_ref(q, k, v, mask, scale):
    """Full FP32 CPU qk/softmax/pv from actual BF16-rounded float lists."""
    import numpy as np
    q = np.asarray(q, dtype=np.float32); k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    groups = q.shape[1] // k.shape[1]
    k = np.repeat(k, groups, axis=1); v = np.repeat(v, groups, axis=1)
    m = np.asarray(mask, dtype=bool)[None, ...]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k, dtype=np.float32) * np.float32(scale)
    scores = np.where(m, np.float32(-np.inf), scores)
    top = np.max(scores, axis=-1, keepdims=True)
    ex = np.exp(scores - top)
    probs = ex / np.sum(ex, axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", probs, v, dtype=np.float32).tolist()
def rope_ref(x, cos, sin):
    import numpy as np
    x = np.asarray(x, dtype=np.float32); cos = np.asarray(cos, dtype=np.float32)
    sin = np.asarray(sin, dtype=np.float32); half = x.shape[-1] // 2
    rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return (x * cos + rotated * sin).astype(np.float32).tolist()
def flat(v):
    if isinstance(v, (list, tuple)):
        for item in v:
            yield from flat(item)
    else:
        yield float(v)
def finite(v):
    return all(math.isfinite(x) for x in flat(v))
def compare(actual, expected, atol, rtol):
    a, e = list(flat(actual)), list(flat(expected))
    if len(a) != len(e):
        return {"status": "failed", "passed": False, "reason": "element_count"}
    nonfinite = sum(1 for x in a if not math.isfinite(x))
    diffs = [abs(x - y) for x, y in zip(a, e)]
    mismatches = sum(1 for x, y, d in zip(a, e, diffs)
                     if not math.isfinite(x) or d > atol + rtol * abs(y))
    return {"status": "passed" if not mismatches else "failed",
            "passed": not mismatches, "mismatch_count": mismatches,
            "element_count": len(a), "nonfinite_count": nonfinite,
            "max_abs_error": max(diffs) if diffs else 0.0,
            "max_rel_error": max((d / max(abs(y), 1e-12)
                                 for d, y in zip(diffs, e)), default=0.0),
            "atol": atol, "rtol": rtol,
            "reference_backend": "numpy_float32_cpu_from_actual_bf16_rounded_inputs"}
def _req(value, path, keys, errors):
    if not isinstance(value, dict): errors.append(f"{path}: object required"); return {}
    for key in keys:
        if key not in value: errors.append(f"{path}: missing {key}")
    return value
def _int(v, path, errors, low=1, high=2**31 - 1):
    if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
        errors.append(f"{path}: integer in [{low},{high}] required"); return None
    return v
def _num(v, path, errors, low=0.0, high=1e9):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errors.append(f"{path}: number required"); return None
    v = float(v)
    if not math.isfinite(v) or not low <= v <= high:
        errors.append(f"{path}: number in [{low},{high}] required"); return None
    return v
def _str(v, path, errors):
    if not isinstance(v, str) or not v:
        errors.append(f"{path}: non-empty string required"); return None
    return v
def _protocol(p, errors):
    p = _req(p, "protocol", ("warmup_iterations", "trials", "samples_per_trial",
                             "seed", "reference", "budgets_seconds"), errors)
    warm = _int(p.get("warmup_iterations"), "protocol.warmup_iterations", errors,
                1, 10000)
    trials = _int(p.get("trials"), "protocol.trials", errors, 1, 100)
    samples = _int(p.get("samples_per_trial"), "protocol.samples_per_trial", errors,
                   1, 10000)
    seed = _int(p.get("seed"), "protocol.seed", errors, 0, 2**63 - 1)
    ref = _req(p.get("reference"), "protocol.reference", ("atol", "rtol"), errors)
    atol = _num(ref.get("atol"), "protocol.reference.atol", errors, 0, .25)
    rtol = _num(ref.get("rtol"), "protocol.reference.rtol", errors, 0, .25)
    budgets = _req(p.get("budgets_seconds"), "protocol.budgets_seconds",
                   tuple(CEILINGS), errors)
    vals = {k: _num(budgets.get(k), f"protocol.budgets_seconds.{k}", errors,
                    1e-9, CEILINGS[k]) for k in CEILINGS}
    supervisor = _num(p.get("supervisor_seconds_min", 370),
                      "protocol.supervisor_seconds_min", errors, 370, 1e9)
    if None in (warm, trials, samples, seed, atol, rtol, supervisor) or None in vals.values():
        return None
    return {"warmup_iterations": warm, "trials": trials,
            "samples_per_trial": samples, "seed": seed,
            "reference": {"atol": atol, "rtol": rtol},
            "budgets_seconds": vals, "supervisor_seconds_min": supervisor}
def _case(raw, index, errors):
    path = f"cases[{index}]"; before = len(errors)
    c = _req(raw, path, ("case_id", "operator", "dtype", "layout", "shape",
                         "data", "source", "split", "target_match"), errors)
    cid = _str(c.get("case_id"), f"{path}.case_id", errors)
    if cid and not CASE_ID_RE.match(cid):
        errors.append(f"{path}.case_id: invalid")
    op = c.get("operator")
    if op not in OPERATORS:
        errors.append(f"{path}.operator: unsupported"); return None
    if c.get("target_match") is not False:
        errors.append(f"{path}.target_match: must be false")
    if c.get("split") not in ("train", "heldout"):
        errors.append(f"{path}.split: train or heldout required")
    if c.get("dtype") != "bfloat16":
        errors.append(f"{path}.dtype: bfloat16 required")
    data = _req(c.get("data"), f"{path}.data", ("seed", "low", "high"), errors)
    seed = _int(data.get("seed"), f"{path}.data.seed", errors, 0, 2**63 - 1)
    low = _num(data.get("low"), f"{path}.data.low", errors, -1e6, 1e6)
    high = _num(data.get("high"), f"{path}.data.high", errors, -1e6, 1e6)
    if low is not None and high is not None and low >= high:
        errors.append(f"{path}.data: low < high required")
    source = _req(c.get("source"), f"{path}.source",
                  ("model", "source_scope", "primitive_scope"), errors)
    for key in ("model", "source_scope", "primitive_scope"):
        _str(source.get(key), f"{path}.source.{key}", errors)
    shape = c.get("shape") if isinstance(c.get("shape"), dict) else {}
    if op == "fusion_attention":
        req = ("batch", "q_heads", "kv_heads", "q_tokens", "kv_tokens", "qk_dim", "v_dim")
        if c.get("layout") != "BNSD":
            errors.append(f"{path}.layout: BNSD required")
        dims = {k: _int(shape.get(k), f"{path}.shape.{k}", errors) for k in req}
        mask_kind = c.get("mask")
        if mask_kind not in ("causal_bottom_right", "full_visible"):
            errors.append(f"{path}.mask: causal_bottom_right/full_visible required")
            mask = []
        else:
            mask = (causal_mask(dims["q_tokens"], dims["kv_tokens"])
                    if None not in (dims["q_tokens"], dims["kv_tokens"]) and mask_kind == "causal_bottom_right"
                    else visible_mask(dims["q_tokens"], dims["kv_tokens"])
                    if None not in (dims["q_tokens"], dims["kv_tokens"]) else [])
        if None in dims.values():
            return None
        if dims["q_heads"] % dims["kv_heads"]:
            errors.append(f"{path}: q_heads/kv_heads divisibility")
        if dims["qk_dim"] != dims["v_dim"]:
            errors.append(f"{path}: qk_dim == v_dim required")
        scale = _num(c.get("scale"), f"{path}.scale", errors, 1e-9, 1.0)
        if scale is not None and abs(scale - 1 / math.sqrt(dims["qk_dim"])) > 1e-12:
            errors.append(f"{path}.scale: 1/sqrt(qk_dim) required")
        for key in ("topology", "global_q_heads", "global_kv_heads", "head_dim"):
            if key not in source:
                errors.append(f"{path}.source: missing {key}")
        nnz = sum(row.count(True) for row in mask)
        b, hq, qn, kvn = dims["batch"], dims["q_heads"], dims["q_tokens"], dims["kv_tokens"]
        qb = b * hq * qn * dims["qk_dim"] * 2
        kb = b * dims["kv_heads"] * kvn * dims["qk_dim"] * 2
        vb = b * dims["kv_heads"] * kvn * dims["v_dim"] * 2
        case = {"case_id": cid, "operator": op, "dtype": "bfloat16",
                "layout": "BNSD", "shape": dims, "mask": mask_kind,
                "mask_base_2d": mask,
                "mask_expanded_shape": [b, hq, qn, kvn],
                "mask_convention": "True=masked; key > kv-q+query (bottom-right)" if mask_kind == "causal_bottom_right" else "True=masked; fully visible",
                "mask_true_count_expanded": nnz * b * hq,
                "scale": scale, "data": {"seed": seed, "low": low, "high": high},
                "source": source, "target_match": False, "split": c.get("split"),
                "declared_bytes": {"q": qb, "k": kb, "v": vb,
                                   "mask_bool": b * hq * qn * kvn,
                                   "output0_est": b * hq * qn * dims["v_dim"] * 2}}
    else:
        if c.get("layout") != "BSND" or c.get("rotary_mode") != "half":
            errors.append(f"{path}: bfloat16/BSND/half required")
        dims = {k: _int(shape.get(k), f"{path}.shape.{k}", errors)
                for k in ("batch", "seq", "heads", "dim")}
        if None in dims.values():
            return None
        if dims["dim"] % 2:
            errors.append(f"{path}.shape.dim: even required")
        pos = _req(c.get("positions"), f"{path}.positions", ("start", "end"), errors)
        start = _int(pos.get("start"), f"{path}.positions.start", errors, 0, 2**31 - 1)
        end = _int(pos.get("end"), f"{path}.positions.end", errors, 0, 2**31 - 1)
        if start is not None and end is not None and end - start + 1 != dims["seq"]:
            errors.append(f"{path}.positions: count must equal seq")
        base = _num(c.get("inv_freq_base", 10000), f"{path}.inv_freq_base", errors, 1, 1e12)
        for key in ("partial_rotary_factor", "full_head_dim", "rotated_dim"):
            if key not in source:
                errors.append(f"{path}.source: missing {key}")
        nb = dims["batch"] * dims["seq"] * dims["heads"] * dims["dim"] * 2
        case = {"case_id": cid, "operator": op, "dtype": "bfloat16",
                "layout": "BSND", "shape": dims, "rotary_mode": "half",
                "positions": {"start": start, "end": end},
                "inv_freq_base": base,
                "data": {"seed": seed, "low": low, "high": high},
                "source": source, "target_match": False, "split": c.get("split"),
                "declared_bytes": {"x": nb, "cos": nb, "sin": nb, "output_est": nb}}
    return None if len(errors) > before else case
def validate_manifest(doc):
    errors = []; root = _req(doc, "manifest",
                             ("kind", "manifest_id", "protocol",
                              "expectations", "cases"), errors)
    if root.get("kind") != MANIFEST_KIND:
        errors.append(f"manifest.kind: must be {MANIFEST_KIND}")
    mid = _str(root.get("manifest_id"), "manifest.manifest_id", errors)
    provenance = root.get("source_contract", {})
    if not isinstance(provenance, dict):
        provenance = {}
    protocol = _protocol(root.get("protocol"), errors)
    exp = _req(root.get("expectations"), "manifest.expectations",
               ("cases", "planned_raw_blocks"), errors)
    ec = _int(exp.get("cases"), "manifest.expectations.cases", errors)
    er = _int(exp.get("planned_raw_blocks"), "manifest.expectations.planned_raw_blocks",
              errors)
    raw = root.get("cases")
    cases, seen = [], set()
    if not isinstance(raw, list) or not raw:
        errors.append("manifest.cases: non-empty list required")
    else:
        for i, item in enumerate(raw):
            case = _case(item, i, errors)
            if case and case["case_id"] in seen:
                errors.append(f"cases[{i}].case_id: duplicate")
            elif case:
                seen.add(case["case_id"]); cases.append(case)
    if errors: raise AttentionInputError(errors)
    raw_blocks = len(cases) * protocol["trials"] * protocol["samples_per_trial"]
    if ec != len(cases) or er != raw_blocks:
        raise AttentionInputError([f"expectations: cases {ec}->{len(cases)}, raw {er}->{raw_blocks}"])
    return {"kind": MANIFEST_KIND, "manifest_id": mid,
            "protocol": protocol,
            "source_contract": provenance,
            "cases": cases}
def build_plan(doc):
    m = validate_manifest(doc); p = m["protocol"]
    raw = len(m["cases"]) * p["trials"] * p["samples_per_trial"]
    return {"kind": "a3_attention_rope_capability", "manifest_id": m["manifest_id"],
            "source_contract": m["source_contract"], "protocol": p,
            "cases": m["cases"], "case_ids": [c["case_id"] for c in m["cases"]],
            "counts": {"cases": len(m["cases"]), "trials": p["trials"],
                       "samples_per_trial": p["samples_per_trial"],
                       "planned_condition_executions": len(m["cases"]) * p["trials"],
                       "planned_raw_blocks": raw,
                       "planned_warmup_events_per_trial": p["warmup_iterations"],
                       "planned_warmup_events_per_case": p["warmup_iterations"] * p["trials"]},
            "budgets": {**p["budgets_seconds"],
                        "supervisor_seconds_min": p["supervisor_seconds_min"]}}
def load_manifest(path):
    p = Path(path)
    try: raw = p.read_bytes()
    except OSError as exc: raise AttentionInputError([f"read: {exc}"], path=str(p))
    try:
        doc = json.loads(raw.decode(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite")))
    except Exception as exc: raise AttentionInputError([f"JSON: {exc}"], path=str(p))
    if not isinstance(doc, dict): raise AttentionInputError(["top-level object required"])
    return doc, hashlib.sha256(raw).hexdigest(), str(p)
def dry_run_payload(path):
    doc, digest, p = load_manifest(path)
    return {"dry_run": True, "valid": True,
            "validation_mode": "stdlib_only_no_torch_import",
            "manifest_path": p, "manifest_sha256": digest,
            "plan": build_plan(doc), "writes_results": False,
            "requires_device": False}
def new_results(plan, run_id, device, manifest_path, manifest_sha):
    cases = [{"case_id": c["case_id"], "operator": c["operator"],
              "requested": True, "status": "requested", "target_match": False,
              "split": c["split"], "source": c["source"], "geometry": c,
              "reason": None, "error": None, "trials": [], "native": None,
              "correctness": None, "summary_clocks": None,
              "analytic_ms": None} for c in plan["cases"]]
    doc = {"kind": "a3_attention_rope_capability",
           "run_id": run_id,
           "manifest": {"manifest_id": plan["manifest_id"],
                        "path": manifest_path, "sha256": manifest_sha,
                        "source_contract": plan["source_contract"]},
           "worker": {"module": "mindie_sim.a3_operator_worker_attention",
                      "source_files_sha256": {Path(__file__).name:
                      hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}},
           "device": {"requested": device, "selected": None},
           "environment": None, "protocol": plan["protocol"],
           "budgets": plan["budgets"], "analytic_ms": None,
           "requested": {"case_ids": list(plan["case_ids"]),
                         "cases": plan["counts"]["cases"],
                         "planned_raw_blocks": plan["counts"]["planned_raw_blocks"]},
           "progress": {}, "cases": cases, "failures": [],
           "cleanup": {"status": "not_started", "ok": None}, "cleanup_ok": None,
           "persistence": {"ok": None, "errors": []}, "persistence_ok": None,
           "stop_reason": None, "status": "running",
           "started_at_utc": utc_now(), "finished_at_utc": None}
    refresh(doc); return doc
def refresh(doc):
    counts = {"measured": 0, "failed": 0, "pending": 0, "not_run": 0}
    raw = 0
    for c in doc["cases"]:
        if c["status"] in counts: counts[c["status"]] += 1
        raw += sum(len(t["samples"]) for t in c["trials"])
    req = len(doc["cases"]); term = sum(counts.values())
    doc["progress"] = {"requested_cases": req, "completed_cases": counts["measured"],
                       "measured_cases": counts["measured"],
                       "failed_cases": counts["failed"], "pending_cases": counts["pending"],
                       "not_run_cases": counts["not_run"], "terminal_cases": term,
                       "raw_blocks": raw,
                       "planned_raw_blocks": doc["requested"]["planned_raw_blocks"],
                       "fraction_raw_blocks": raw / doc["requested"]["planned_raw_blocks"]}
    doc["status"] = ("partial" if term < req else "completed"
                     if counts["measured"] == req else "completed_with_failures")
class AtomicWriter:
    def __init__(self, path, allow_overwrite=False):
        self.path = Path(path); self.allow = allow_overwrite
    def preflight(self):
        if self.path.exists() and not self.allow:
            raise AttentionInputError([f"refusing overwrite: {self.path}"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def write(self, doc):
        text = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass
def tensor_facts(t, torch):
    return {"shape": [int(x) for x in t.shape], "strides": [int(x) for x in t.stride()],
            "dtype": str(t.dtype).replace("torch.", ""), "numel": int(t.numel()),
            "nbytes": int(t.numel()) * int(t.element_size()),
            "data_ptr": int(t.data_ptr()), "device": str(t.device),
            "contiguous": bool(t.is_contiguous())}
def item_facts(item, torch):
    if isinstance(item, torch.Tensor): return {"kind": "tensor", **tensor_facts(item, torch)}
    if isinstance(item, (int, float, bool)) or item is None: return {"kind": type(item).__name__, "value": item}
    return {"kind": type(item).__name__, "repr": repr(item)[:200]}
class AttentionRecipeMixin:
    """Importable builders returning the existing worker's recipe-dict shape."""
    def _ctx(self):
        s = getattr(self, "stream", None)
        return self.torch.npu.stream(s) if s is not None else nullcontext()
    def _cpu(self, shape, case):
        g = self.torch.Generator(device="cpu"); g.manual_seed(case["data"]["seed"])
        return self.torch.empty(tuple(shape), dtype=self.torch.float32).uniform_(
            case["data"]["low"], case["data"]["high"], generator=g).to(self.torch.bfloat16)
    def _dev(self, cpu):
        with self._ctx(): d = cpu.to(device=self.device)
        if getattr(self, "stream", None) is not None: self.stream.synchronize()
        return d
    def _run_native(self, op):
        stream = getattr(self, "stream", None)
        if stream is not None:
            with self.torch.npu.stream(stream): result = op()
            stream.synchronize()
        else:
            result = op(); self.torch.npu.synchronize()
        return result
    def _fa_call(self, case, q, k, v, mask):
        fn = getattr(self.torch_npu, "npu_fusion_attention", None)
        if not callable(fn): raise PendingBackend("npu_fusion_attention unavailable")
        d = case["shape"]
        try:
            return fn(q, k, v, head_num=d["q_heads"], input_layout="BNSD",
                      atten_mask=mask, scale=case["scale"], keep_prob=1.0,
                      pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)
        except Exception as exc:
            raise PendingBackend(f"npu_fusion_attention_dispatch:{type(exc).__name__}:{exc}") from exc
    def build_fusion_attention(self, case):
        d = case["shape"]; b, hq, hkv, qn, kvn = (d["batch"], d["q_heads"],
            d["kv_heads"], d["q_tokens"], d["kv_tokens"])
        qc = self._cpu((b, hq, qn, d["qk_dim"]), case)
        kc = self._cpu((b, hkv, kvn, d["qk_dim"]), case)
        vc = self._cpu((b, hkv, kvn, d["v_dim"]), case)
        mc = self.torch.tensor(case["mask_base_2d"], dtype=self.torch.bool).expand(
            b, hq, qn, kvn).contiguous()
        mask = self._dev(mc); q = self._dev(qc); k = self._dev(kc); v = self._dev(vc)
        def op(): return self._fa_call(case, q, k, v, mask)
        try:
            native = self._run_native(op)
        except PendingBackend: raise
        except Exception as exc:
            raise PendingBackend(f"npu_fusion_attention_dispatch:{type(exc).__name__}:{exc}") from exc
        if not isinstance(native, (tuple, list)) or len(native) != 7:
            raise BadNativeOutput(f"npu_fusion_attention arity != 7: {type(native).__name__}")
        out = native[0]
        want = [b, hq, qn, d["v_dim"]]
        if not isinstance(out, self.torch.Tensor) or list(out.shape) != want or str(out.dtype) != "torch.bfloat16":
            raise BadNativeOutput(f"npu_fusion_attention output0 {getattr(out,'shape',None)}/{getattr(out,'dtype',None)} != {want}/torch.bfloat16")
        expected = attention_ref(qc.float().tolist(), kc.float().tolist(),
                                 vc.float().tolist(), case["mask_base_2d"], case["scale"])
        actual = out.detach().to("cpu").float().tolist()
        check = compare(actual, expected, case["tolerance"]["atol"], case["tolerance"]["rtol"])
        if not finite(actual) or not check["passed"]:
            raise BadNativeOutput(f"npu_fusion_attention correctness: {check}")
        aux = [{"index": i, **item_facts(x, self.torch)} for i, x in enumerate(native[1:], 1)]
        facts = {"observed_api": "torch_npu.npu_fusion_attention", "return_arity": 7,
                 "output0": tensor_facts(out, self.torch), "auxiliaries": aux,
                 "auxiliary_meaning": "not interpreted", "call_kwargs":
                 {"head_num": hq, "input_layout": "BNSD", "scale": case["scale"],
                  "keep_prob": 1.0, "sparse_mode": 0, "pre_tockens": 2147483647,
                  "next_tockens": 2147483647},
                 "allocation": "native_default_allocating", "preallocated": False,
                 "out_overload": False, "retained_policy": "last_output_only"}
        details = {"geometry": d, "layout": "BNSD", "scale": case["scale"],
                   "mask": {"kind": case["mask"], "convention": case["mask_convention"],
                            "base_2d": case["mask_base_2d"],
                            "expanded_shape": [b, hq, qn, kvn],
                            "true_count": int(mc.sum().item()), "dtype": "bool",
                            "contiguous": True},
                   "inputs": {"q": tensor_facts(q, self.torch), "k": tensor_facts(k, self.torch),
                              "v": tensor_facts(v, self.torch), "mask": tensor_facts(mask, self.torch)},
                   "native": facts, "reference": "full_cpu_qk_softmax_pv_from_bf16_values",
                   "native_output_allocation": True}
        return {"op": op, "out": out, "input": (q, k, v, mask), "preallocated": False,
                "kind": "npu_fusion_attention", "correctness": check,
                "observed_api": "torch_npu.npu_fusion_attention",
                "details": details, "native": facts}
    def _rope_call(self, x, cos, sin):
        fn = getattr(self.torch_npu, "npu_rotary_mul", None)
        if not callable(fn): raise PendingBackend("npu_rotary_mul unavailable")
        try: return fn(x, cos, sin, rotary_mode="half")
        except Exception as exc:
            raise PendingBackend(f"npu_rotary_mul_dispatch:{type(exc).__name__}:{exc}") from exc
    def build_rotary_mul(self, case):
        d = case["shape"]; b, s, h, dim = d["batch"], d["seq"], d["heads"], d["dim"]
        xc = self._cpu((b, s, h, dim), case)
        pos = self.torch.arange(case["positions"]["start"], case["positions"]["end"] + 1,
                                dtype=self.torch.float32)
        inv = 1.0 / (case["inv_freq_base"] ** (self.torch.arange(0, dim, 2,
                                                                dtype=self.torch.float32) / dim))
        ang = pos[:, None] * inv[None, :]
        c2 = self.torch.cat((self.torch.cos(ang), self.torch.cos(ang)), dim=-1)
        s2 = self.torch.cat((self.torch.sin(ang), self.torch.sin(ang)), dim=-1)
        cc = c2[None, :, None, :].expand(b, s, h, dim).contiguous().to(self.torch.bfloat16)
        sc = s2[None, :, None, :].expand(b, s, h, dim).contiguous().to(self.torch.bfloat16)
        cos = self._dev(cc); sin = self._dev(sc); x = self._dev(xc)
        def op(): return self._rope_call(x, cos, sin)
        try:
            native = self._run_native(op)
        except PendingBackend: raise
        except Exception as exc:
            raise PendingBackend(f"npu_rotary_mul_dispatch:{type(exc).__name__}:{exc}") from exc
        if not isinstance(native, self.torch.Tensor) or list(native.shape) != [b, s, h, dim] or str(native.dtype) != "torch.bfloat16":
            raise BadNativeOutput(f"npu_rotary_mul output {getattr(native,'shape',None)}/{getattr(native,'dtype',None)}")
        expected = rope_ref(xc.float().tolist(), cc.float().tolist(), sc.float().tolist())
        actual = native.detach().to("cpu").float().tolist()
        check = compare(actual, expected, case["tolerance"]["atol"], case["tolerance"]["rtol"])
        if not finite(actual) or not check["passed"]:
            raise BadNativeOutput(f"npu_rotary_mul correctness: {check}")
        variation = float((cc.float()[0, -1] - cc.float()[0, 0]).abs().max().item())
        facts = {"observed_api": "torch_npu.npu_rotary_mul", "rotary_mode": "half",
                 "return_arity": 1, "output": tensor_facts(native, self.torch),
                 "allocation": "native_default_allocating", "preallocated": False,
                 "out_overload": False, "retained_policy": "last_output_only"}
        details = {"geometry": d, "layout": "BSND", "rotary_mode": "half",
                   "half_rotation": "cat(-x[...,D/2:],x[...,:D/2])",
                   "positions": case["positions"], "inv_freq_base": case["inv_freq_base"],
                   "cos_shape": list(cc.shape), "sin_shape": list(sc.shape),
                   "cos_contiguous": bool(cc.is_contiguous()),
                   "sin_contiguous": bool(sc.is_contiguous()),
                   "position_cos_variation_max": variation,
                   "full_shape_cos_sin": True, "broadcast_assumption": None,
                   "inputs": {"x": tensor_facts(x, self.torch), "cos": tensor_facts(cos, self.torch),
                              "sin": tensor_facts(sin, self.torch)},
                   "native": facts, "reference": "full_cpu_half_rotation_from_bf16_values",
                   "native_output_allocation": True}
        return {"op": op, "out": native, "input": (x, cos, sin), "preallocated": False,
                "kind": "npu_rotary_mul", "correctness": check,
                "observed_api": "torch_npu.npu_rotary_mul",
                "details": details, "native": facts}
class Clock:
    def __init__(self, budgets):
        self.b = {k: float(budgets[k]) for k in CEILINGS}
        for k, v in CEILINGS.items():
            if self.b[k] > v: raise ValueError(f"{k} budget too high")
        self.t0 = time.monotonic(); self.work0 = None; self.clean0 = None
    def setup(self, stage):
        e = time.monotonic() - self.t0
        if e > self.b["setup"]: raise BudgetExceeded(stage, e, self.b["setup"])
    def work_start(self): self.work0 = time.monotonic()
    def work(self, stage):
        if self.work0 is not None:
            e = time.monotonic() - self.work0
            if e > self.b["global_work"]: raise GlobalWorkBudgetExceeded(stage, e, self.b["global_work"])
    def case(self, start, stage):
        e = time.monotonic() - start; self.work(stage)
        if e > self.b["case"]: raise BudgetExceeded(stage, e, self.b["case"])
    def clean_start(self): self.clean0 = time.monotonic()
    def clean(self, stage):
        e = time.monotonic() - self.clean0
        if e > self.b["cleanup"]: raise BudgetExceeded(stage, e, self.b["cleanup"])
class Runner(AttentionRecipeMixin):
    def __init__(self, device="npu:0", tolerance=None):
        self.device = device; self.tolerance = tolerance or {"atol": .02, "rtol": .02}
        self.torch = None; self.torch_npu = None; self.stream = None; self.environment = {}
    def setup(self, clock):
        t0 = time.monotonic(); clock.setup("import")
        try:
            self.torch = importlib.import_module("torch"); self.torch_npu = importlib.import_module("torch_npu")
        except Exception as exc: raise RuntimeError(f"torch/torch_npu import: {exc}") from exc
        if self.device != "npu:0": raise RuntimeError("device must be npu:0")
        clock.setup("device"); self.torch.npu.set_device(0); self.torch.npu.synchronize()
        self.stream = self.torch.npu.Stream(device=0); self.torch.npu.synchronize()
        pk = {}
        for n in ("torch", "torch-npu", "torch_npu", "numpy"):
            try: pk[n] = importlib.metadata.version(n)
            except Exception: pk[n] = None
        self.environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "torch": getattr(self.torch, "__version__", None),
                            "torch_npu": getattr(self.torch_npu, "__version__", None),
                            "packages": pk, "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                            "stream": repr(self.stream),
                            "clocks": {"host": "time.perf_counter_ns",
                                       "device_event": "torch.npu.Event(enable_timing=True)"}}
        try:
            self.environment["device_name"] = str(self.torch.npu.get_device_name(0))
            self.environment["mem_get_info"] = repr(self.torch.npu.mem_get_info())
        except Exception as exc:
            self.environment["device_query_error"] = type(exc).__name__
        clock.setup("complete")
        return {"status": "ok", "duration_s": time.monotonic() - t0,
                "inference_mode": bool(self.torch.is_inference_mode_enabled())}
    def prepare(self, case, clock, start):
        clock.case(start, "prepare"); c = dict(case); c["tolerance"] = self.tolerance
        with self.torch.inference_mode():
            r = (self.build_fusion_attention(c) if c["operator"] == "fusion_attention"
                 else self.build_rotary_mul(c))
        ev = (self.torch.npu.Event(enable_timing=True), self.torch.npu.Event(enable_timing=True))
        r["out"] = None
        clock.case(start, "prepare_complete")
        return {"case": c, "recipe": r, "events": ev, "native": r["native"],
                "correctness": r["correctness"], "details": r["details"], "active": 0.0}
    def _primary(self, result):
        return result[0] if isinstance(result, (tuple, list)) else result
    def trial(self, art, idx, protocol, clock, start):
        op = art["recipe"]["op"]; se, ee = art["events"]
        tr = {"trial_index": idx, "status": "measuring", "warmup": None, "samples": [],
              "finished_at_utc": None}
        try:
            w0 = time.monotonic()
            with self.torch.npu.stream(self.stream):
                for _ in range(protocol["warmup_iterations"]):
                    result = op(); _last = self._primary(result); result = None
            self.stream.synchronize(); _last = None
            tr["warmup"] = {"iterations": protocol["warmup_iterations"], "timed": False,
                            "excluded_from_samples": True,
                            "duration_s": time.monotonic() - w0,
                            "scope": "exact_native_allocating_call_before_each_trial",
                            "output_retained": False}
            art["active"] = time.monotonic() - start
            clock.case(start, "warmup")
            for si in range(protocol["samples_per_trial"]):
                clock.case(start, "sample"); self.stream.synchronize()
                _last = None; result = None
                with self.torch.npu.stream(self.stream):
                    w = time.perf_counter_ns(); se.record(self.stream)
                    c0 = time.perf_counter_ns(); result = op()
                    c1 = time.perf_counter_ns()
                    _last = self._primary(result); result = None
                    ee.record(self.stream)
                    ee.synchronize(); w1 = time.perf_counter_ns()
                    dev = float(se.elapsed_time(ee))
                tr["samples"].append({"trial_index": idx, "sample_index": si,
                                      "api_invocations": 1,
                                      "cpu_submit_ms": (c1 - c0) / 1e6,
                                      "device_interval_ms": dev,
                                      "wall_complete_ms": (w1 - w) / 1e6,
                                      "analytic_ms": None, "events_reused": True,
                                      "cpu_submit_scope": "native_api_call_only",
                                      "auxiliary_lifetime": "native_return_alive_until_output0_extraction_after_submit_clock",
                                      "synchronize_between_samples_outside_clocks": True})
                art["active"] = time.monotonic() - start
            clock.case(start, "case_complete")
            tr["status"] = "measured"; tr["finished_at_utc"] = utc_now()
            return tr
        except Exception as exc:
            tr["status"] = "failed"; tr["finished_at_utc"] = utc_now()
            raise TrialFailure(tr, exc) from exc
    def cleanup(self, clock):
        info = {"status": "running", "ok": False}; t0 = time.monotonic()
        try:
            clock.clean("start")
            if self.torch is not None and self.stream is not None:
                self.stream.synchronize(); self.torch.npu.synchronize()
                info["synchronized"] = True
                clock.clean("after_synchronize")
                info.update({"status": "ok", "ok": True, "synchronized": True})
            else: info.update({"status": "not_needed", "ok": True})
        except Exception as exc:
            info.update({"status": "failed", "ok": False, "error": exception_info(exc)})
        info["duration_s"] = time.monotonic() - t0
        info["supervisor_seconds_min"] = SUPERVISOR_SECONDS_MIN
        return info
def run_manifest(plan, manifest_path, manifest_sha256, run_id, device, results_path,
                 allow_overwrite=False):
    if device != "npu:0": raise AttentionInputError(["device must be npu:0"])
    writer = AtomicWriter(results_path, allow_overwrite); writer.preflight()
    doc = new_results(plan, run_id, device, manifest_path, manifest_sha256)
    errors = []
    def persist(stage):
        try: writer.write(doc); return
        except Exception as exc:
            errors.append({"stage": stage, "path": str(results_path), "at_utc": utc_now(),
                           "error": exception_info(exc)})
            doc["persistence"] = {"ok": False, "errors": errors}; doc["persistence_ok"] = False
            raise PersistenceFailure(str(exc)) from exc
    try: persist("initial_requested_cases")
    except PersistenceFailure:
        print(json.dumps({"error": "persistence_failure", "durable_final_write": False,
                          "errors": errors}), file=sys.stderr)
        return 1
    clock = Clock(plan["budgets"]); runner = None; stop = None; setup_failed = False
    try:
        try:
            runner = Runner(device, plan["protocol"]["reference"])
            setup = runner.setup(clock); doc["environment"] = runner.environment
            doc["worker"]["setup"] = setup; doc["device"]["selected"] = device
            persist("setup")
        except PersistenceFailure: raise
        except Exception as exc:
            setup_failed = True; doc["setup"] = {"status": "failed", "error": exception_info(exc)}
            for c in doc["cases"]: c["status"] = "not_run"; c["reason"] = "setup_failed"
            stop = "setup_failed"; persist("setup_failure")
        if not setup_failed and stop is None:
            clock.work_start()
            for cd, cp in zip(doc["cases"], plan["cases"]):
                if cd["status"] != "requested": continue
                try: clock.work("before_case")
                except GlobalWorkBudgetExceeded as exc: stop = str(exc); break
                start = time.monotonic()
                try:
                    art = runner.prepare(cp, clock, start)
                except PendingBackend as exc:
                    cd.update({"status": "pending", "reason": str(exc),
                               "error": exception_info(exc)})
                    doc["failures"].append({"case_id": cd["case_id"], "status": "pending",
                                            "error": exception_info(exc)})
                    refresh(doc); persist("case_pending"); continue
                except Exception as exc:
                    cd.update({"status": "failed", "reason": str(exc),
                               "error": exception_info(exc)})
                    doc["failures"].append({"case_id": cd["case_id"], "status": "failed",
                                            "error": exception_info(exc)})
                    refresh(doc); persist("case_failed"); continue
                cd.update({"native": art["native"], "correctness": art["correctness"],
                           "details": art["details"], "allocation": art["recipe"].get("native"),
                           "analytic_ms": None})
                persist("case_prepared")
                try:
                    for ti in range(plan["protocol"]["trials"]):
                        tr = runner.trial(art, ti, plan["protocol"], clock, start)
                        cd["trials"].append(tr); persist(f"trial_{ti}")
                    cd["status"] = "measured"; cd["reason"] = None
                except TrialFailure as exc:
                    cd["trials"].append(exc.trial)
                    cd.update({"status": "failed", "reason": str(exc.cause),
                               "error": exception_info(exc.cause)})
                    doc["failures"].append({"case_id": cd["case_id"], "status": "failed",
                                            "error": exception_info(exc.cause)})
                    refresh(doc); persist("case_measurement_failure")
                refresh(doc); persist("case_complete")
    except PersistenceFailure:
        stop = "persistence_failure"
    except KeyboardInterrupt:
        stop = "keyboard_interrupt"
    except Exception as exc:
        doc["runtime_error"] = exception_info(exc)
        stop = f"unexpected_runtime_error:{type(exc).__name__}"
    finally:
        clock.clean_start()
        cleanup = runner.cleanup(clock) if runner else {"status": "not_started", "ok": True}
        doc["cleanup"] = cleanup; doc["cleanup_ok"] = cleanup.get("ok") is True
        if stop: doc["stop_reason"] = stop
        for c in doc["cases"]:
            if c["status"] == "requested": c["status"] = "not_run"; c["reason"] = "run_stopped"
        refresh(doc)
        if setup_failed: doc["status"] = "failed"
        elif doc["status"] == "completed" and not doc["cleanup_ok"]:
            doc["status"] = "completed_with_cleanup_failure"
        doc["persistence"] = {"ok": not errors, "errors": errors}
        doc["persistence_ok"] = not errors
        if not doc["persistence_ok"] and doc["status"] == "completed":
            doc["status"] = "completed_with_persistence_failure"
        doc["finished_at_utc"] = utc_now()
        try: writer.write(doc); durable = True
        except Exception as exc:
            durable = False; errors.append({"stage": "final", "path": str(results_path),
                                            "at_utc": utc_now(), "error": exception_info(exc)})
            doc["persistence"] = {"ok": False, "errors": errors}; doc["persistence_ok"] = False
            if doc["status"] == "completed": doc["status"] = "completed_with_persistence_failure"
        if errors:
            print(json.dumps({"error": "persistence_failure", "durable_final_write": durable,
                              "errors": errors}), file=sys.stderr)
    return 0 if (doc["status"] == "completed" and doc["progress"]["measured_cases"]
                 == doc["progress"]["requested_cases"] and doc["cleanup_ok"]
                 and doc["persistence_ok"]) else 1
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m mindie_sim.a3_operator_worker_attention")
    p.add_argument("--manifest", required=True); p.add_argument("--results", required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--device", default="npu:0")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if not a.run_id.strip(): print(json.dumps({"error": "run_id_required"}), file=sys.stderr); return 2
    if a.device != "npu:0": print(json.dumps({"error": "device_must_be_npu:0"}), file=sys.stderr); return 2
    if a.dry_run:
        try: payload = dry_run_payload(a.manifest)
        except AttentionInputError as exc:
            print(json.dumps({"error": "manifest_invalid", "errors": exc.errors}), file=sys.stderr); return 2
        payload.update({"run_id": a.run_id, "device": a.device})
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)); return 0
    try:
        doc, sha, path = load_manifest(a.manifest); plan = build_plan(doc)
    except AttentionInputError as exc:
        print(json.dumps({"error": "manifest_invalid", "errors": exc.errors}), file=sys.stderr); return 2
    try:
        return run_manifest(plan, path, sha, a.run_id, a.device, a.results,
                            allow_overwrite=a.overwrite)
    except AttentionInputError as exc:
        print(json.dumps({"error": "output_path_error", "errors": exc.errors}), file=sys.stderr); return 2
    except KeyboardInterrupt: return 130
if __name__ == "__main__":
    raise SystemExit(main())
