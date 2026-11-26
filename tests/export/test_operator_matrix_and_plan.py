"""Tier-0 tests for multi-model operator matrix, physics cliffs, HBM dual-peak gate, BenchPlanV2 & outward pure-English export.

[Role]
  Verify unseen synthetic HuggingFace ``config.json`` evaluation (AG014-clean),
  physics-derived sequence cliffs and ``R2-PA-12b-07`` ``local_kv`` ABI
  divisibility, dual-peak HBM + host DDR admissibility (Astra #4), structural
  parallel arms (first-principles ``expert_tp > 1`` vs shipping firewall),
  un-derated nominal Roofline / MFU / MBU formulas, exact ``reverse_index``
  multiset conservation and ``BenchPlanV2`` sealing + v1 compat projection,
  multi-file pure-English outward package (OOXML native line charts + live
  Excel formulas + standalone HTML viewer + ``.tar.gz``), ``assert_zero_internal_jargon``
  positive & negative gates, and ``profiler_normalizer`` ``R2-PA-15a-06`` + S1
  concurrent rank execution.
[Invariants]
  * Pure Tier-0 execution (< 5s), isolated in ``tmp_path``, zero NPU devices,
    zero host environment mutations, and zero heavy multi-model full-grid sweeps.
[Timeouts & Locks]
  Deterministic in-memory + temp-directory checks (< 5s); no locks.
[Side-Effect Policy]
  Writes exclusively inside pytest ``tmp_path``.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

from mindie_sim import api
from mindie_sim.backends.ascend import profiles as ascend_profiles
from mindie_sim.cli import _main as cli_main
from mindie_sim.compose import compose_case
from mindie_sim.export.matrix_writer import (
    HW_NOMINAL_COLUMNS,
    OUTWARD_OP_COLUMNS,
    assert_zero_internal_jargon,
    export_operator_matrix,
)
from mindie_sim.export.ooxml import ChartSeries, LineChartSpec, Workbook, column_letter
from mindie_sim.export.operator_matrix import (
    BUILDERS,
    COLLECTIVE_BUILDERS,
    build_bench_plan,
    build_operator_matrix,
    derive_physical_seq_cliffs,
    enumerate_parallel_arms,
    evaluate_hbm_admissibility,
    render_subgraph_plan_v1_compat,
)
from mindie_sim.onboarding.profiler_normalizer import (
    acquire_stage3_observations,
    normalize_profiler_records,
    resolve_profiler_builder_strict,
    run_profiler_bench_subprocess,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.bench_matrix import (
    BENCH_PLAN_V2_SCHEMA,
    FABRIC_COLLECTIVE_ABI_MAP,
    BenchPlanV2,
    HardwareNominalSpec,
    HbmAdmissibilityReceipt,
    MatrixSpec,
    ParallelArmSpec,
    assert_no_forbidden_prediction_keys,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

pytestmark = pytest.mark.tier0


def _write_synthetic_hf_config(tmp_path: Path, *, name: str = "synthetic-unseen-7b", moe: bool = False) -> Path:
    cfg: dict[str, object] = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "vocab_size": 32000,
        "max_position_embeddings": 65536,
        "torch_dtype": "bfloat16",
    }
    if moe:
        cfg.update({
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_experts": 16,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 1024,
        })
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_physics_cliffs_ag014_clean_and_local_kv_abi_divisibility(tmp_path: Path) -> None:
    """R2-PA-12b-07: every non-abi_rejection cliff obeys cp_stride * local_kv_abi_divisor."""
    hf_path = _write_synthetic_hf_config(tmp_path)
    parsed = parse_case(compose_case(hf_path, hardware="a5", tp=4, dcp=2, pcp=2, ep=1, layers=2))
    device = parsed.machine.dies[0]
    par = ParallelSpec(dp=1, tp=4, ep=1, pp=1, dcp=2, pcp=2)

    cliffs = derive_physical_seq_cliffs(
        parsed.model,
        device,
        par,
        seq_min=512,
        seq_max=32768,
        concurrency=4,
        cliff_budget_per_arm=8,
        local_kv_abi_divisor=16,
        include_abi_rejection=True,
    )
    assert len(cliffs) >= 3
    non_rej = [c for c in cliffs if c.split != "abi_rejection"]
    rej = [c for c in cliffs if c.split == "abi_rejection"]
    assert non_rej and rej
    cp_stride = par.dcp * par.pcp
    for pt in non_rej:
        assert pt.seq_len % cp_stride == 0
        local_kv = pt.seq_len // cp_stride
        assert local_kv % 16 == 0, (pt.seq_len, cp_stride, local_kv)
        assert pt.formula_expr
    assert any(pt.seq_len % (cp_stride * 16) != 0 for pt in rej)
    kinds = {c.cliff_kind for c in non_rej}
    assert "anchor" in kinds and len(kinds) >= 2


def test_parallel_arm_first_principles_vs_shipping_firewall() -> None:
    """Physics/target arms allow routed_expert_tp > 1; shipping vllm-ascend rejects it."""
    moe_model = ModelSpec.from_mapping({
        "name": "syn-moe-arm",
        "layers": 2,
        "hidden_size": 1024,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "intermediate_size": 2048,
        "vocab_size": 8192,
        "attention": "gqa",
        "ffn": "moe",
        "num_experts": 16,
        "top_k": 2,
        "moe_intermediate_size": 512,
    })
    etp_arm = ParallelArmSpec(
        arm_id="etp2_ep4", dp=1, tp=8, ep=4, routed_expert_tp=2, shared_expert_tp=8, num_devices=8,
    )
    ship_valid, ship_skip = enumerate_parallel_arms(
        moe_model, "Ascend950DT_9582", [etp_arm], feature_track="shipping",
    )
    assert any(s["reason"] == "shipping_vllm_ascend_requires_routed_expert_tp_1" for s in ship_skip)
    assert all(a.arm_id != "etp2_ep4" for a in ship_valid)

    tgt_valid, tgt_skip = enumerate_parallel_arms(
        moe_model, "Ascend950DT_9582", [etp_arm], feature_track="target",
    )
    assert any(a.arm_id == "etp2_ep4" and a.routed_expert_tp == 2 for a in tgt_valid)
    assert not any(s["arm_id"] == "etp2_ep4" for s in tgt_skip)


def test_dual_peak_hbm_and_host_ddr_admissibility_gate(tmp_path: Path) -> None:
    """Astra #4: load_peak > usable or runtime_peak > usable or host_ddr > cap rejects cell."""
    # Schema-level dual-peak invariant: runtime <= usable but load > usable MUST NOT be admissible.
    with pytest.raises(ModelingError, match="hbm_admissibility_dual_peak_violation"):
        HbmAdmissibilityReceipt(
            admissible=True,
            reason="ADMISSIBLE",
            usable_hbm_bytes_per_die=64 * (1024 ** 3),
            runtime_peak_hbm_bytes_by_die=(50 * (1024 ** 3),),
            load_peak_hbm_bytes_by_die=(70 * (1024 ** 3),),
        )
    with pytest.raises(ModelingError, match="hbm_admissibility_host_ddr_violation"):
        HbmAdmissibilityReceipt(
            admissible=True,
            reason="ADMISSIBLE",
            usable_hbm_bytes_per_die=64 * (1024 ** 3),
            runtime_peak_hbm_bytes_by_die=(20 * (1024 ** 3),),
            load_peak_hbm_bytes_by_die=(20 * (1024 ** 3),),
            host_ddr_bytes=200 * (1024 ** 3),
            usable_host_ddr_bytes=128 * (1024 ** 3),
        )

    # Real case evaluation: tight HBM override triggers inadmissible cell with zero fake latency.
    hf_path = _write_synthetic_hf_config(tmp_path)
    tight_case = parse_case(compose_case(
        hf_path, hardware="a2", tp=1, dp=1, ep=1, layers=2,
        memory_bytes=64 * (1024 ** 2), prompt_tokens=2048, output_tokens=64, concurrency=32,
    ))
    receipt, _ledger = evaluate_hbm_admissibility(
        tight_case, phase="decode", seq_len=2048, concurrency=32, cell_id="c-tight",
    )
    assert receipt.admissible is False
    assert "HBM" in receipt.reason
    assert receipt.max_runtime_peak_hbm_bytes > receipt.usable_hbm_bytes_per_die or (
        receipt.max_load_peak_hbm_bytes > receipt.usable_hbm_bytes_per_die
    )

    # End-to-end matrix gate: massive KV demand marks cell inadmissible with 0.0 step time & zero fake op rows.
    oom_spec = MatrixSpec(
        models=(str(hf_path),),
        skus=("a2",),
        phases=("decode",),
        seq_min=32768,
        seq_max=32768,
        anchor_ladder=(32768,),
        auto_physical_cliffs=False,
        concurrencies=(2048,),
        parallel_arms=(ParallelArmSpec(arm_id="tp1", dp=1, tp=1, ep=1, num_devices=1),),
        stages=("before",),
    )
    oom_manifest = build_operator_matrix(oom_spec, layers=1)
    assert len(oom_manifest.step_summaries) == 1
    assert oom_manifest.step_summaries[0].admissible is False
    assert oom_manifest.step_summaries[0].sim_step_us == 0.0
    assert len(oom_manifest.rows) == 0


def test_unseen_hf_matrix_roofline_reverse_index_ooxml_charts_and_sanitization(tmp_path: Path) -> None:
    """Full Tier-0 E2E: unseen HF JSON -> matrix -> BenchPlanV2 -> pure-English package + .tar.gz."""
    hf_path = _write_synthetic_hf_config(tmp_path, name="unseen-hf-demo")
    spec = MatrixSpec(
        models=(str(hf_path),),
        skus=("a5",),
        phases=("decode",),
        seq_min=512,
        seq_max=2048,
        anchor_ladder=(512, 2048),
        auto_physical_cliffs=True,
        cliff_budget_per_arm=2,
        concurrencies=(1, 4),
        parallel_arms=(ParallelArmSpec(arm_id="tp1", dp=1, tp=1, ep=1, num_devices=1),),
        stages=("before",),
    )
    manifest = build_operator_matrix(spec, layers=1)
    assert manifest.rows and manifest.step_summaries and manifest.hbm_receipts
    assert all(s.admissible for s in manifest.step_summaries)

    # 1. Verify strictly un-derated nominal datasheet specs (utilization == 1.0, launch_us == 0.0)
    sku_key = next(iter(manifest.nominal_specs))
    nom_spec = manifest.nominal_specs[sku_key]
    assert nom_spec.utilization == 1.0 and nom_spec.launch_us == 0.0
    raw_prof = ascend_profiles()[sku_key]
    assert nom_spec.hbm_peak_bytes_per_s == pytest.approx(float(raw_prof["device"]["rates"]["hbm"]))

    # 2. Verify operator Roofline / MFU / MBU math & white-box formulas (unclamped raw prediction)
    for row in manifest.rows:
        assert row.formula_symbolic and row.formula_terms
        assert row.hbm_total_bytes == row.hbm_read_bytes + row.hbm_write_bytes
        expected_theo = nom_spec.nominal_theo_us(
            cube_flops=float(row.cube_flops), cube_dtype=row.cube_dtype,
            vector_ops=float(row.vector_ops),
            hbm_read_bytes=float(row.hbm_read_bytes),
            hbm_write_bytes=float(row.hbm_write_bytes),
            link_bytes=float(row.link_bytes), link_tier=row.link_tier,
        )
        assert row.nominal_theo_us == pytest.approx(expected_theo, abs=1e-4)
        assert row.per_call_stage0_us > 0.0

    # 3. Verify deduplicated BenchPlanV2 seal, zero predicted_* keys, exact reverse_index multiset conservation & compat-v1
    plan_v2 = build_bench_plan(manifest, include_fabric=False)
    assert isinstance(plan_v2, BenchPlanV2)
    assert plan_v2.schema == BENCH_PLAN_V2_SCHEMA
    plan_dict = plan_v2.to_dict(include_seal=True)
    assert_no_forbidden_prediction_keys(plan_dict)
    reloaded = BenchPlanV2.from_mapping(plan_dict, verify_seal=True)
    assert reloaded.plan_sha256 == plan_v2.plan_sha256

    plan_occ: Counter[tuple[object, ...]] = Counter()
    for cid, entries in plan_v2.reverse_index.items():
        for e in entries:
            plan_occ[(cid, *e.occurrence_key())] += e.occurrence_count
    manifest_occ: Counter[tuple[object, ...]] = Counter()
    for cid, entries in manifest.reverse_index.items():
        for e in entries:
            manifest_occ[(cid, *e.occurrence_key())] += e.occurrence_count
    assert plan_occ == manifest_occ and sum(plan_occ.values()) > 0
    assert sum(plan_occ.values()) == sum(r.occurrence_count for r in manifest.rows)

    v1_plan = render_subgraph_plan_v1_compat(plan_v2)
    assert v1_plan["schema"] == "mindie-bench-subgraph-plan/1"
    assert len(v1_plan["cases"]) == len(plan_v2.cases)
    assert all(c["execution_modes"] == [c["execution_mode"]] for c in v1_plan["cases"])

    # 4. Export structured multi-file pure-English outward package + .tar.gz + separate internal bench plan
    bundle_dir = tmp_path / "outward_pkg"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stale_note = bundle_dir / "stale_notes.txt"
    stale_note.write_text("stale customer file must not enter tar.gz\n", encoding="utf-8")
    plan_dir = tmp_path / "internal_plans"
    receipt = export_operator_matrix(
        manifest, bundle_dir, emit_bench_plan_dir=plan_dir, compat_v1=True, tar_gz=True, include_fabric=False,
    )
    assert receipt.sanitization_passed is True
    assert receipt.tar_gz_path and Path(receipt.tar_gz_path).is_file()
    assert len(receipt.bench_plan_paths) == 1 and Path(receipt.bench_plan_paths[0]).is_file()
    assert len(receipt.compat_v1_paths) == 1 and Path(receipt.compat_v1_paths[0]).is_file()
    assert_zero_internal_jargon(bundle_dir)

    # Verify exact outward file layout, verbatim CANN msprof Block Num headers & zero stale/bench_plan files in tar.gz
    model_dir = next((bundle_dir / "models").iterdir())
    assert (bundle_dir / "README.md").is_file()
    assert (bundle_dir / "hardware_nominal_specs.csv").is_file()
    assert (bundle_dir / "summary" / "Executive_Summary.xlsx").is_file()
    assert (bundle_dir / "summary" / "model_step_summary.csv").is_file()
    assert (bundle_dir / "operators" / "unique_kernel_catalog.csv").is_file()
    assert (bundle_dir / "charts" / "scaling_explorer.html").is_file()
    op_csv = model_dir / "op_breakdown_matrix.csv"
    op_xlsx = model_dir / "Operator_Breakdown.xlsx"
    assert op_csv.is_file() and op_xlsx.is_file() and (model_dir / "scaling_curves.csv").is_file()
    with op_csv.open(encoding="utf-8") as fh:
        headers = next(csv.reader(fh))
    assert tuple(headers) == OUTWARD_OP_COLUMNS
    assert "Block Num" in headers and "Mix Block Num" in headers

    # Verify OOXML workbook has 4 sheets, live Excel formulas referencing Hardware_Nominal_Specs, and native charts!
    with zipfile.ZipFile(op_xlsx, "r") as zf:
        names = set(zf.namelist())
        assert "xl/charts/chart1.xml" in names
        assert "xl/charts/chart2.xml" in names
        assert "xl/drawings/drawing1.xml" in names
        wb_xml = zf.read("xl/workbook.xml").decode("utf-8")
        for sheet_name in ("Hardware_Nominal_Specs", "Step_Summary", "Op_Breakdown", "Scaling_Curves"):
            assert sheet_name in wb_xml
        op_sheet_xml = zf.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "Hardware_Nominal_Specs!" in op_sheet_xml and "<f>" in op_sheet_xml

    # Verify quoted chart title produces valid XML (audit #22)
    import xml.etree.ElementTree as ET
    quoted_xlsx = tmp_path / "quoted_chart.xlsx"
    qwb = Workbook()
    qws = qwb.create_sheet('Sheet "A"')
    qws.set_cell(1, 1, "x")
    qws.set_cell(2, 1, 1)
    qws.set_cell(1, 2, "y")
    qws.set_cell(2, 2, 2)
    qws.add_line_chart(LineChartSpec(
        title='Latency at "target"', x_title="X", y_title="Y", anchor_from=(3, 1), anchor_to=(8, 10),
        series=(ChartSeries("S", "Sheet!$A$2:$A$2", "Sheet!$B$2:$B$2"),),
    ))
    qwb.save(quoted_xlsx)
    with zipfile.ZipFile(quoted_xlsx, "r") as zf:
        for member in zf.namelist():
            if member.endswith((".xml", ".rels")):
                ET.fromstring(zf.read(member))

    with tarfile.open(receipt.tar_gz_path, "r:gz") as tf:
        tar_members = tf.getnames()
        assert all("bench_plan" not in m.lower() for m in tar_members)
        assert all("stale_notes.txt" not in m for m in tar_members)

    # 5. Negative sanitization gates: nested bench_plan (even under plan_only), forbidden token, CJK, Win path, IPv6 & XML!
    with pytest.raises(ModelingError, match="bench_plan_dir_must_be_outside_outward_bundle"):
        export_operator_matrix(manifest, bundle_dir, emit_bench_plan_dir=bundle_dir / "bench_plan")
    with pytest.raises(ModelingError, match="bench_plan_dir_must_be_outside_outward_bundle"):
        export_operator_matrix(manifest, bundle_dir, plan_only=True, emit_bench_plan_dir=bundle_dir / "internal")

    for tag, content, err_pat in (
        ("bad_tok", "case,k2-deadbeef01234567\n", "outward_internal_token_leak"),
        ("bad_cjk", "Latency breakdown 中文 leak\n", "outward_cjk_character_leak"),
        ("bad_win", r"model,C:\Users\alice\Private\model" + "\n", "outward_private_path_leak"),
        ("bad_v6", "host,fd12:3456:789a::1\n", "outward_private_ip_leak"),
    ):
        bdir = tmp_path / tag
        bdir.mkdir()
        (bdir / "leak.csv").write_text(content, encoding="utf-8")
        with pytest.raises(ModelingError, match=err_pat):
            assert_zero_internal_jargon(bdir)

    bad_xlsx_dir = tmp_path / "bad_xlsx"
    bad_xlsx_dir.mkdir()
    with zipfile.ZipFile(bad_xlsx_dir / "bad.xlsx", "w") as zf:
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet><t>Stage 0 internal</t></worksheet>")
    with pytest.raises(ModelingError, match="outward_internal_token_leak"):
        assert_zero_internal_jargon(bad_xlsx_dir)

    # 6. CLI mindie-sim matrix --plan-only round-trip + SDK exports
    cli_out = tmp_path / "cli_plan_out"
    rc = cli_main([
        "matrix", "--models", str(hf_path), "--skus", "a5", "--phases", "decode",
        "--seq-range", "512..512", "--no-auto-cliffs", "--concurrencies", "1",
        "--parallel-arms", "tp1", "--layers", "1", "--plan-only", "--no-fabric",
        "--out-dir", str(cli_out),
    ])
    assert rc == 0
    assert any(p.name.endswith(".bench_plan_v2.json") for p in cli_out.iterdir())
    assert callable(api.build_operator_matrix) and callable(api.export_operator_matrix)
    assert callable(api.build_bench_plan) and callable(api.reconcile_campaign)


def test_profiler_normalizer_qualified_survival_and_concurrent_s1_execution(tmp_path: Path) -> None:
    """R2-PA-15a-06 qualified survival, rank validation, and deadlock-free concurrent S1 launch."""
    qual_row = {
        "schema": "mindie-bench-observation/1", "record_type": "observation",
        "case_id": "gemm", "run_id": "run.stream_saturated.0", "split": "train",
        "timing_mode": "stream_saturated", "duration_ms": 1.0, "qualified": True,
        "rank": 0, "trial": 0, "repeat_index": 0, "sample_id": "s-qual",
        "source": "capture://bench-subgraph/gemm",
    }
    unqual_row = {
        "schema": "mindie-bench-observation/1", "record_type": "observation",
        "case_id": "gemm", "run_id": "run.host_fallback.0", "split": "train",
        "timing_mode": "host_fallback", "duration_ms": 10.0, "qualified": False,
        "rank": 0, "trial": 0, "repeat_index": 0, "sample_id": "s-host",
        "source": "capture://bench-subgraph/gemm",
    }
    out_mixed, _ = normalize_profiler_records([qual_row, unqual_row], plan_doc={}, sku="a3")
    assert len(out_mixed) == 1
    assert out_mixed[0]["measured_ns"] == 1_000_000
    assert out_mixed[0]["clock_operator"] == "busy_union"

    with pytest.raises(ModelingError, match="profiler_observation_unqualified"):
        normalize_profiler_records([unqual_row], plan_doc={}, sku="a3")
    with pytest.raises(ModelingError, match="profiler_observation_rank_invalid"):
        normalize_profiler_records([{**qual_row, "rank": -1}], plan_doc={}, sku="a3")

    # Audit #30: incompatible eager vs stream_saturated timing modes stay partitioned; partial rank set rejected!
    eager_row = {**qual_row, "timing_mode": "eager", "duration_ms": 1.0, "sample_id": "s-eager"}
    sat_row = {**qual_row, "timing_mode": "stream_saturated", "duration_ms": 9.0, "sample_id": "s-sat"}
    split_rows, _ = normalize_profiler_records([eager_row, sat_row], plan_doc={}, sku="a3")
    assert len(split_rows) == 2 and {r["clock_operator"] for r in split_rows} == {"elapsed_span", "busy_union"}
    with pytest.raises(ModelingError, match="profiler_observation_rank_"):
        normalize_profiler_records(
            [{**qual_row, "case_id": "c0", "rank": 2}],
            plan_doc={"world_size": 2, "cases": [{"case_id": "c0", "group_ranks": [0, 1]}]},
            sku="a3",
        )

    # Audit #31: relative imports inside external profiler_root mindie_profiler/bench/operators.py work cleanly!
    ext_prof = tmp_path / "ext_prof_pkg"
    (ext_prof / "mindie_profiler" / "bench").mkdir(parents=True)
    (ext_prof / "mindie_profiler" / "__init__.py").write_text("", encoding="utf-8")
    (ext_prof / "mindie_profiler" / "bench" / "__init__.py").write_text("", encoding="utf-8")
    (ext_prof / "mindie_profiler" / "bench" / "_helper.py").write_text("TAG = 'aclnnMatmul'\n", encoding="utf-8")
    (ext_prof / "mindie_profiler" / "bench" / "operators.py").write_text(
        "from ._helper import TAG\n"
        "def resolve_builder_strict(f, r, i, *, dtypes=(), meta=None): return ('matmul', TAG)\n",
        encoding="utf-8",
    )
    assert resolve_profiler_builder_strict("gemm", "linear", "", profiler_root=ext_prof) == ("matmul", "aclnnMatmul")

    # Concurrent S1 proof: rank 0 waits for rank 1's rendezvous marker file; sequential execution would deadlock!
    marker_dir = tmp_path / "rendezvous"
    marker_dir.mkdir()
    fake_prof = tmp_path / "fake_s1_prof"
    (fake_prof / "mindie_profiler").mkdir(parents=True)
    (fake_prof / "mindie_profiler" / "__init__.py").write_text("", encoding="utf-8")
    (fake_prof / "mindie_profiler" / "__main__.py").write_text(
        "import json, os, sys, time\nfrom pathlib import Path\nargs = sys.argv[1:]\n"
        "plan = json.loads(Path(args[args.index('--plan') + 1]).read_text(encoding='utf-8'))\n"
        "obs_p, out_p = Path(args[args.index('--observations') + 1]), Path(args[args.index('--out') + 1])\n"
        "rk, ws = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])\n"
        "assert os.environ.get('MASTER_ADDR') == '127.0.0.1' and int(os.environ.get('MASTER_PORT', '0')) > 0\n"
        f"mdir = Path({str(marker_dir)!r})\n"
        "(mdir / f'rank{rk}.ready').write_text('ok', encoding='utf-8')\n"
        "t0 = time.monotonic()\n"
        "while any(not (mdir / f'rank{i}.ready').exists() for i in range(ws)):\n"
        "    if time.monotonic() - t0 > 4.0:\n"
        "        raise SystemExit(9)\n"
        "    time.sleep(0.02)\n"
        "obs_p.write_text(json.dumps({'schema':'mindie-bench-observation/1','record_type':'observation','case_id':'c0','run_id':f'r{rk}.eager.t0','trial':0,'repeat_index':0,'rank':rk,'split':'train','timing_mode':'eager','duration_ms':0.5 + 0.1 * rk,'sku':plan['sku'],'event_ids':[f'e{rk}']}) + '\\n', encoding='utf-8')\n"
        "Path(f'{out_p}.rank{rk}.json').write_text(json.dumps({'schema':'mindie-bench-subgraph-run/1','status':'ok','run_id':f'r{rk}','records':[{'case_id':'c0','status':'ok','actual_shape':{'tokens':8,'width':64},'timed_nodes':['n0']}]}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    s1_plan = {
        "schema": "mindie-bench-subgraph-plan/1",
        "plan_id": "p-s1",
        "sku": "Ascend910B4",
        "stage": "s1_joint",
        "world_size": 2,
        "execution_modes": ["eager"],
        "cases": [{"case_id": "c0", "split": "train", "stage": "s1_joint", "nodes": [{"node_id": "n0"}]}],
    }
    rows_s1, _ = run_profiler_bench_subprocess(
        s1_plan, sku="Ascend910B4", profiler_root=fake_prof, timeout_s=10,
    )
    assert len(rows_s1) == 1 and rows_s1[0]["device_ranks"] == [0, 1]
    assert rows_s1[0]["measured_ns"] == 600_000


def test_nominal_cube_vector_peak_columns_ooxml_formulas_and_builder_lock(tmp_path: Path) -> None:
    """Verify Cube+Vector nominal peak columns, shifted OOXML formula refs, retired tools & profiler builder lock."""
    import xml.etree.ElementTree as ET

    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "tools" / "bench_packets.py").exists()
    assert not (repo_root / "tools" / "export_subgraph_bench_plan.py").exists()

    hf_path = _write_synthetic_hf_config(tmp_path, name="nominal-peak-demo")
    out_dir = tmp_path / "matrix_outward"
    rc = cli_main([
        "matrix", "--models", str(hf_path), "--skus", "Ascend950DT_9582,Ascend910_9362",
        "--phases", "decode", "--seq-range", "512..512", "--no-auto-cliffs",
        "--concurrencies", "1", "--parallel-arms", "tp1", "--layers", "1",
        "--no-fabric", "--out-dir", str(out_dir),
    ])
    assert rc == 0
    assert_zero_internal_jargon(out_dir)

    with (out_dir / "hardware_nominal_specs.csv").open(encoding="utf-8") as fh:
        csv_rows = list(csv.reader(fh))
    assert tuple(csv_rows[0]) == HW_NOMINAL_COLUMNS
    cube_i = HW_NOMINAL_COLUMNS.index("Nominal Cube Peak(TFLOPS)")
    vec_i = HW_NOMINAL_COLUMNS.index("Nominal Vector Peak(TFLOPS)")
    tot_i = HW_NOMINAL_COLUMNS.index("Nominal Total Peak Cube+Vector(TFLOPS)")
    assert vec_i == cube_i + 1 and tot_i == vec_i + 1
    by_sku = {r[0]: dict(zip(HW_NOMINAL_COLUMNS, r)) for r in csv_rows[1:]}
    for sku, exp_cube, exp_vec, exp_tot in (
        ("Ascend950DT_9582", 432.54, 54.07, 486.60),
        ("Ascend910_9362", 270.34, 16.90, 287.23),
    ):
        rec = by_sku[sku]
        c_val, v_val, t_val = (
            float(rec["Nominal Cube Peak(TFLOPS)"]),
            float(rec["Nominal Vector Peak(TFLOPS)"]),
            float(rec["Nominal Total Peak Cube+Vector(TFLOPS)"]),
        )
        assert round(c_val, 2) == exp_cube and round(v_val, 2) == exp_vec and round(t_val, 2) == exp_tot
        assert c_val + v_val == pytest.approx(t_val, abs=1e-6)
    assert float(by_sku["Ascend910_9362"]["Intra-Pkg SIO Peak(Bytes/s)"]) > 0.0

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    for token in ("Nominal Cube Peak", "Nominal Vector Peak", "Nominal Total Peak Cube+Vector", "486.60", "287.23"):
        assert token in readme

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cube_col, vec_col, tot_col = column_letter(cube_i + 1), column_letter(vec_i + 1), column_letter(tot_i + 1)
    bf16_raw_col = column_letter(HW_NOMINAL_COLUMNS.index("BF16 Cube Peak(FLOPs/s)") + 1)
    hbm_raw_col = column_letter(HW_NOMINAL_COLUMNS.index("HBM Peak(Bytes/s)") + 1)
    vec_raw_col = column_letter(HW_NOMINAL_COLUMNS.index("Vector Peak(Ops/s)") + 1)
    model_dir = next((out_dir / "models").iterdir())
    for xlsx_path, hw_sheet in (
        (out_dir / "summary" / "Executive_Summary.xlsx", "xl/worksheets/sheet2.xml"),
        (model_dir / "Operator_Breakdown.xlsx", "xl/worksheets/sheet1.xml"),
    ):
        with zipfile.ZipFile(xlsx_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith((".xml", ".rels")):
                    ET.fromstring(zf.read(member))
            hw_root = ET.fromstring(zf.read(hw_sheet))
            cells = {c.attrib["r"]: c for c in hw_root.findall(".//m:c", ns)}
            for r_idx in (2, 3):
                tot_cell = cells[f"{tot_col}{r_idx}"]
                f_node = tot_cell.find("m:f", ns)
                assert f_node is not None and f_node.text == f"{cube_col}{r_idx}+{vec_col}{r_idx}"
            if xlsx_path.name == "Operator_Breakdown.xlsx":
                op_xml = zf.read("xl/worksheets/sheet3.xml").decode("utf-8")
                assert f"Hardware_Nominal_Specs!${bf16_raw_col}$2:${bf16_raw_col}$3" in op_xml
                assert f"Hardware_Nominal_Specs!${hbm_raw_col}$2:${hbm_raw_col}$3" in op_xml
                assert f"Hardware_Nominal_Specs!${tot_col}$2:${tot_col}$3" not in op_xml
                if f"Hardware_Nominal_Specs!${vec_raw_col}$2:${vec_raw_col}$3" in op_xml:
                    assert vec_raw_col != tot_col

    assert COLLECTIVE_BUILDERS <= BUILDERS
    assert {v[0] for v in FABRIC_COLLECTIVE_ABI_MAP.values()} <= BUILDERS
    try:
        from mindie_profiler.bench.operators import COLLECTIVE_OPS, OPERATOR_BUILDERS
    except ImportError:
        OPERATOR_BUILDERS = COLLECTIVE_OPS = None  # type: ignore[assignment]
    if OPERATOR_BUILDERS is not None and COLLECTIVE_OPS is not None:
        profiler_vocab = set(OPERATOR_BUILDERS) | set(COLLECTIVE_OPS)
        assert not sorted(set(BUILDERS) - profiler_vocab)

    # Verify HF32 & canonical d2d_intra_package SIO rows bind to exact shifted Hardware_Nominal_Specs columns.
    from dataclasses import replace as dc_replace
    from mindie_sim.export.matrix_writer import _nominal_spec_row, _write_model_operator_breakdown_xlsx
    base_manifest = build_operator_matrix(MatrixSpec(
        models=(str(hf_path),), skus=("Ascend950DT_9582", "Ascend910_9362"), phases=("decode",),
        seq_min=512, seq_max=512, anchor_ladder=(512,), auto_physical_cliffs=False,
        concurrencies=(1,), parallel_arms=(ParallelArmSpec(arm_id="tp1", dp=1, tp=1, ep=1, num_devices=1),),
    ), layers=1)
    r_a5 = next(r for r in base_manifest.rows if r.sku == "Ascend950DT_9582")
    r_a3 = next(r for r in base_manifest.rows if r.sku == "Ascend910_9362")
    probe_rows = [
        dc_replace(r_a5, cube_dtype="hf32", cube_flops=216_268_800, per_call_stage0_us=2.0),
        dc_replace(r_a3, link_tier="d2d_intra_package", link_bytes=270_000, per_call_stage0_us=1.0),
    ]
    probe_xlsx = tmp_path / "hf32_sio_probe.xlsx"
    nom_rows = [_nominal_spec_row(s) for _, s in sorted(base_manifest.nominal_specs.items())]
    _write_model_operator_breakdown_xlsx(probe_xlsx, nom_rows, [], probe_rows, [])
    hf32_col = column_letter(HW_NOMINAL_COLUMNS.index("HF32 Cube Peak(FLOPs/s)") + 1)
    sio_col = column_letter(HW_NOMINAL_COLUMNS.index("Intra-Pkg SIO Peak(Bytes/s)") + 1)
    with zipfile.ZipFile(probe_xlsx, "r") as zf:
        op_probe_xml = zf.read("xl/worksheets/sheet3.xml").decode("utf-8")
    assert f"Hardware_Nominal_Specs!${hf32_col}$2:${hf32_col}$3" in op_probe_xml
    assert f"Hardware_Nominal_Specs!${sio_col}$2:${sio_col}$3" in op_probe_xml
