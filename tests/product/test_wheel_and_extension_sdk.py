"""Installed-wheel isolation and the external extension SDK contract.

[Role] Exercise workspace providers, installed entry points, and profiler replay.
[Invariants] A shared declaration keeps one program and one charge per call.
  Wheel-mode execution uses installed distributions outside the source checkout.
[Timeouts & Locks] Bounded CPU-only execution and tiny CSV capture replay.
[Side-Effect Policy] Temporary extensions and captures live under tmp_path.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path

import pytest

import mindie_sim

pytestmark = pytest.mark.tier1

SKU = "Ascend950DT_9582"
M, K, N = 4, 65, 32
FLOPS = 2.0 * M * K * N
A_BYTES = M * K * 2
B_BYTES = K * N * 2
C_BYTES = M * N * 2
HBM_BYTES = A_BYTES + B_BYTES + C_BYTES
MTE2_BYTES = A_BYTES + B_BYTES
MTE3_BYTES = C_BYTES
ACLNN_ENTRY = "aclnnMatmul"
PROFILER_RULE = "aclnnMatmul_AcmeMatmulV3"


EXTENSION_SOURCE = '''"""External provider of a matrix operation and its profiler replay."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

from mindie_sim.public import Extension, Provider, simulate_model, simulate_service
from mindie_sim.schema.operators import OpSpec

M, K, N = 4, 65, 32
FLOPS = 2.0 * M * K * N
A_BYTES = M * K * 2
B_BYTES = K * N * 2
C_BYTES = M * N * 2
HBM_BYTES = A_BYTES + B_BYTES + C_BYTES
METADATA = (
    ("extension_id", "acme.wheel_demo"),
    ("semantic_op", "acme.matmul"),
    ("program_kind", "matmul"),
    ("physical_program", "SteadyMatmul"),
    ("activation_format", "bf16"),
    ("output_format", "bf16"),
    ("matrix_groups", 1),
    ("aclnn_api", "aclnnMatmul"),
    ("aclnn_symbol", "aclnnMatmul"),
    ("profiler_rule", "aclnnMatmul_AcmeMatmulV3"),
)


def make_op(op_id: str = "acme.matmul") -> OpSpec:
    return OpSpec(
        op_id, "gemm", "decode", 0, 0, (),
        (M, K, N), (M, K, N), (M, K, N), (M, K, N),
        dtype="bf16", flops=FLOPS, hbm_bytes=HBM_BYTES,
        mte2_bytes=A_BYTES + B_BYTES, mte3_bytes=C_BYTES,
        recipe="gemm", metadata=METADATA,
    )


def _declared(ops):
    values = tuple(ops) if ops is not None else ()
    return values if values else (make_op(),)


def op_builder(case, ops, **kwargs):
    return {"ops": _declared(ops)}


def model_provider(case, extensions=None, **kwargs):
    kwargs.pop("scope", None)
    report = dict(simulate_model(case, provider="core", **kwargs))
    report["extension_provider"] = "acme.wheel_demo"
    report["extension_fingerprint"] = extensions.fingerprint if extensions else None
    return report


def service_provider(case, extensions=None, **kwargs):
    kwargs.pop("scope", None)
    report = dict(simulate_service(case, provider="core", **kwargs))
    report["extension_provider"] = "acme.wheel_demo"
    report["extension_fingerprint"] = extensions.fingerprint if extensions else None
    return report


def _capture_rows(source):
    root = Path(source)
    if root.is_dir():
        root = root / "kernel_details.csv"
    rows = []
    with root.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "name": row.get("Name") or "",
                "duration_us": float(row.get("Duration(us)") or 0.0),
                "dtype": row.get("Input Data Types") or "",
            })
    return rows


def profile_provider(source, extensions=None, **kwargs):
    rows = _capture_rows(source)
    rule = "aclnnMatmul_AcmeMatmulV3"
    selected = [row for row in rows if row["name"] == rule]
    total = round(sum(row["duration_us"] for row in selected), 6)
    return {
        "schema_version": "mindie-profile-summary/1",
        "step_accounting": {
            "step_status": "not_provided",
            "kernel_sum_ms_p50": total / 1000.0,
            "adapter": "acme.wheel_demo",
        },
        "source_identity": {
            "schema": "acme.profiler-replay/1",
            "rule": rule,
            "rows": len(selected),
            "fingerprint": sha256(json.dumps(
                selected, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "observed_total_us": total,
        "kernel_names": [row["name"] for row in selected],
        "extension_fingerprint": extensions.fingerprint if extensions else None,
    }


def compare_provider(baseline, candidate, extensions=None, **kwargs):
    predicted = float(baseline.get("predicted_us", 0.0))
    observed = float(candidate.get("observed_us", 0.0))
    ratio = (predicted / observed) if observed else None
    return {
        "schema_version": "mindie-profile-comparison/1",
        "status": "pass" if ratio is not None and abs(ratio - 1.0) <= 0.10
                  else "fail",
        "predicted_us": predicted,
        "observed_us": observed,
        "ratio": ratio,
        "items": [{"name": "acme.matmul", "ratio": ratio}],
    }


EXTENSION = Extension(
    id="acme.wheel_demo", version="1.0.0",
    metadata={"owner": "wheel-only-e2e"},
    providers=(
        Provider("operator", "acme_demo", op_builder, purpose="builder",
                 default=True),
        Provider("native", "acme_demo", op_builder, purpose="builder",
                 default=True),
        Provider("graph", "acme_demo", op_builder, purpose="builder",
                 default=True),
        Provider("model", "acme_demo", model_provider, purpose="report",
                 default=True),
        Provider("service", "acme_demo", service_provider, purpose="report",
                 default=True),
        Provider("profile", "acme_demo", profile_provider, purpose="report",
                 default=True),
        Provider("compare", "acme_demo", compare_provider, purpose="report",
                 default=True),
    ),
)
'''


def _install_is_wheel(distribution: str) -> None:
    dist = metadata.distribution(distribution)
    assert dist.version and dist.read_text("WHEEL")
    raw = dist.read_text("direct_url.json")
    direct = json.loads(raw) if raw else {}
    assert not direct.get("dir_info", {}).get("editable", False), (
        distribution + " is an editable/source install, not a wheel")


def _tree_manifest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts
        and path.suffix != ".pyc"}


def _environment():
    from mindie_sim.backends.ascend import materialize
    from mindie_sim.public import ExecutionEnvironment
    from mindie_sim.schema.hardware import Hardware
    return ExecutionEnvironment.from_hardware(Hardware.from_mapping(materialize({"sku": SKU})))


def _tiny_case(output_tokens=1):
    """One legal 3-layer projection; machine resources come from the SKU."""
    from mindie_sim.backends.ascend import materialize
    machine = materialize({"sku": SKU})
    machine["num_devices"] = 1
    machine["ranks_per_node"] = 1
    return {"name": "wheel-only-tiny", "machine": machine, "model": {"catalog_key": "deepseek-v3.2-prune-3l"},
        "workload": {"requests": [{"id": "r0", "arrival_ms": 0.0, "prompt_tokens": 16,
            "output_tokens": output_tokens}]}, "runtime": {"tp": 1, "dp": 1, "ep": 1, "pp": 1,
            "max_model_len": 8192, "max_num_seqs": 2, "max_num_batched_tokens": 1024,
            "gpu_memory_utilization": 0.9, "enable_chunked_prefill": True, "kv_block_size": 128,
            "graph_mode": "none", "async_scheduling": False,},}


def _write_extension(tmp_path: Path) -> Path:
    root = tmp_path / "acme_extension"
    root.mkdir()
    (root / "mindie_ext.py").write_text(EXTENSION_SOURCE, encoding="utf-8")
    (root / "helper_rule.py").write_text("RULE = 'aclnnMatmul_AcmeMatmulV3'\n", encoding="utf-8")
    return root / "mindie_ext.py"


def _capture(tmp_path: Path) -> Path:
    root = tmp_path / "capture_r0"
    root.mkdir()
    header = ["Device_id", "Model ID", "Task ID", "Stream ID", "Name", "Type",
        "OP State", "Accelerator Core", "Start Time(us)", "Duration(us)",
        "Wait Time(us)", "Block Num", "Mix Block Num", "HF32 Eligible",
        "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
        "Output Data Types", "Output Formats", "Context ID",]
    rows = [[0, 1, 1, 1, PROFILER_RULE, "MatMulV3", "static", "AI_CORE",
         "1000.000", "10.000", "0.000", 16, 0, "NO", '"4,65;65,32"',
         "DT_BF16;DT_BF16", "ND;ND", '"4,32"', "DT_BF16", "ND", "N/A"],
        [0, 1, 2, 1, PROFILER_RULE, "MatMulV3", "static", "AI_CORE",
         "1010.000", "20.000", "0.000", 16, 0, "NO", '"4,65;65,32"',
         "DT_BF16;DT_BF16", "ND;ND", '"4,32"', "DT_BF16", "ND", "N/A"],]
    with (root / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return root


def _site_package(module):
    return any(part in {"site-packages", "dist-packages"} for part in Path(module.__file__).resolve().parts)


def test_runtime_package_origin():
    if os.environ.get("MINDIE_WHEEL_ONLY") == "1" or _site_package(mindie_sim):
        _assert_wheel_runtime()
    else:
        assert Path(mindie_sim.__file__).resolve() == Path(__file__).resolve().parents[2].joinpath(
            "mindie_sim", "__init__.py")


def _assert_wheel_runtime():
    assert _site_package(mindie_sim)
    _install_is_wheel("mindie-sim")
    assert not os.environ.get("PYTHONPATH")
    assert not (Path.cwd() / "mindie_sim" / "__init__.py").is_file()
    assert "tests" not in str(Path(mindie_sim.__file__).resolve())


def test_workspace_extension_registers_capabilities_without_touching_site_packages(tmp_path):
    from mindie_sim.public import CAPABILITIES, load_extensions
    package_root = Path(mindie_sim.__file__).resolve().parent
    before = _tree_manifest(package_root)
    entry = _write_extension(tmp_path)
    context = load_extensions(paths=[entry.parent])
    assert len(context) == 1
    extension = context.extension("acme.wheel_demo")
    assert extension.source_kind == "workspace"
    assert extension.code_sha256
    assert Path(extension.source).is_relative_to(tmp_path)
    assert not Path(extension.source).is_relative_to(Path(mindie_sim.__file__).resolve().parent)
    providers = {(item.capability, item.name) for item in extension.providers}
    assert providers == {(capability, "acme_demo") for capability in (
        "operator", "native", "graph", "model", "service", "profile", "compare")}
    assert {item.capability for item in extension.providers} <= CAPABILITIES
    assert context.fingerprint.startswith("sha256:")
    assert context.scoped_fingerprint(capabilities=["operator"]) != context.scoped_fingerprint(
        capabilities=["service"])
    after = _tree_manifest(package_root)
    assert before == after, "extension loading modified the runtime package"
    listing = context.describe()
    assert listing["schema"] == "mindie-sim.extensions/1"
    assert listing["extensions"][0]["source_kind"] == "workspace"
    again = load_extensions(paths=[entry.parent])
    assert again.fingerprint == context.fingerprint


ENTRY_MODULE = '''"""A module shipped by an installed distribution, not a workspace file."""
from mindie_sim.public import Extension, Provider


def factory(case=None, ops=None, **kwargs):
    return {"ops": tuple(ops or ())}


def extension():
    return Extension(
        id="acme.entry_point", version="1.0.0",
        providers=(Provider("operator", "entry_demo", factory,
                            purpose="builder", default=True),))


EXTENSION = extension()
'''


def test_installed_entry_point_package_registers_extension(tmp_path, monkeypatch):
    from mindie_sim.public import load_extensions
    site = tmp_path / "site"
    site.mkdir()
    (site / "acme_entry_ext.py").write_text(ENTRY_MODULE, encoding="utf-8")
    dist = site / "acme_entry_ext-1.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: acme-entry-ext\nVersion: 1.0.0\n",
        encoding="utf-8")
    (dist / "entry_points.txt").write_text("[mindie_sim.extensions]\nacme-entry = acme_entry_ext:extension\n",
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    context = load_extensions(packages=["acme-entry-ext"])
    assert len(context) == 1
    extension = context.extension("acme.entry_point")
    assert extension.source_kind == "entry_point"
    assert "mindie_sim.extensions:acme-entry" in extension.source
    assert len(extension.code_sha256) == 64
    assert context.provider("operator", "entry_demo").name == "entry_demo"
    assert context.fingerprint.startswith("sha256:")



def test_extension_entries_share_one_physical_program_and_one_charge(tmp_path):
    from mindie_sim.public import (load_extensions, simulate_graph, simulate_native_call, simulate_operator,)
    context = load_extensions(paths=[_write_extension(tmp_path)])
    environment = _environment()
    op = _extension_op()
    operator = simulate_operator(environment, (op,), extensions=context)
    native = simulate_native_call(environment, (op,), extensions=context)
    graph = simulate_graph(environment, (op,), extensions=context)
    for report in (operator, native, graph):
        assert report["schema"] in {"mindie-sim.operators/1", "mindie-sim.call-costs/1", "mindie-sim.graph/1"}
    durations = {"operator": operator["operator_sum_ms"], "native": native["duration_ms"],
        "graph": graph["duration_ms"],}
    assert len(set(round(value, 12) for value in durations.values())) == 1, durations
    assert durations["operator"] > 0
    for report in (operator, native, graph):
        row = report["timeline"][0]
        assert row["program"] == "SteadyMatmul"
        assert row["operator"]["recipe"] == "gemm"
        assert row["operator"]["metadata"]["aclnn_api"] == ACLNN_ENTRY
        assert row["operator"]["metadata"]["profiler_rule"] == PROFILER_RULE
    assert operator["operators"][0]["occurrences"] == 1
    assert operator["operators"][0]["charge"] == "call"
    assert operator["operators"][0]["serviced"]["memory@r0:s0"] == pytest.approx(HBM_BYTES)
    assert native["physical_operators"] == 1
    assert native["expanded_operators"] == 1
    assert native["operator_occurrences"] == 1
    twice = simulate_native_call(environment, (op,), extensions=context, invocations=2, mode="eager")
    assert twice["physical_operators"] == 1
    assert twice["expanded_operators"] == 2
    assert twice["operator_occurrences"] == 2
    assert twice["duration_ms"] == pytest.approx(2 * native["duration_ms"])
    assert twice["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(2 * HBM_BYTES)
    assert context.select("operator").name == "acme_demo"



def test_missing_capability_reports_a_replayable_gap(tmp_path):
    from mindie_sim.public import (CapabilityGap, GAP_SCHEMA, load_extensions, simulate_operator,)
    context = load_extensions(paths=[_write_extension(tmp_path)])
    environment = _environment()
    unmodeled = replace(_extension_op(), dtype="int4_custom", matrix_flops=(("int4_custom", FLOPS),))
    with pytest.raises(CapabilityGap, match="int4_custom") as caught:
        simulate_operator(environment, (unmodeled,), extensions=context)
    gap = caught.value.gap
    assert gap["schema"] == GAP_SCHEMA
    assert gap["status"] == "gap"
    assert gap["capability"] == "operator"
    assert gap["entry"] == "simulate_operator"
    assert gap["kind"] == "unmodeled_capability"
    assert gap["code"] == "native_compute_mode_unsupported"
    assert "int4_custom" in gap["message"]
    assert gap["fallback"] == "not_attempted"
    assert gap["result_available"] is False
    assert gap["replayable"] is True
    assert "operator_sum_ms" in gap["affected_results"]
    replay = gap["minimal_replay_input"]
    assert replay["entry"] == "simulate_operator"
    assert replay["ops"][0]["dtype"] == "int4_custom"
    assert replay["environment"]["sku"] == SKU
    ok = simulate_operator(environment, (_extension_op(),), extensions=context)
    assert ok["operator_sum_ms"] > 0
    assert ok["operators"][0]["operator"]["dtype"] == "bf16"


def test_pure_model_forward_bills_each_declared_occurrence_once(tmp_path):
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.public import (load_extensions, simulate_model, simulate_operator,)
    context = load_extensions(paths=[_write_extension(tmp_path)])
    case = _tiny_case()
    report = simulate_model(case, extensions=context, phase="prefill", tokens=16, context=0,
        sequences=1, event_limit=8)
    assert report["schema"] == "mindie-sim.body-forward/1"
    assert report["status"] == "modeled"
    assert report["extension_provider"] == "acme.wheel_demo"
    assert report["operator_occurrences"] > 0
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    assert report["execution"]["completed_tasks"] > 0
    for resource in report["resources"]:
        if resource["unit"] == "peak_ms" or not resource["capacity_per_ms"]:
            continue
        assert resource["capacity_bound_ms"] == pytest.approx(
            resource["work"] / resource["capacity_per_ms"], rel=1e-9)
        assert resource["busy_ms"] + 1e-9 >= resource["capacity_bound_ms"]
    rows = report["timeline"]
    assert rows
    assert any("Matmul" in row["program"] for row in rows)
    assert all(row["native_call"]["members"]["charge"] == "call" for row in rows)
    parsed = parse_case(case)
    graphs = target_graphs(parsed, phase="prefill", tokens=16, context=0, sequences=1,
        scope="body", apply_body_defaults=True)
    declared = {op.op_id: op for op in graphs[0]}
    environment = _environment()
    matched = 0
    for row in rows:
        if row["program"] != "AIVTiledVectorAndGatherRecipe":
            continue
        op = declared.get(row["op_id"])
        if op is None or not op.dtype:
            continue
        standalone = simulate_operator(environment, (replace(op, deps=()),), extensions=context,
            provider="core")
        assert standalone["operator_sum_ms"] == pytest.approx(row["elapsed_ms"], rel=1e-12, abs=1e-12)
        assert standalone["timeline"][0]["program"] == row["program"]
        matched += 1
    assert matched >= 5


def test_service_entry_commits_declared_batches_and_leaves_no_live_task(tmp_path):
    from mindie_sim.public import load_extensions, simulate_service
    context = load_extensions(paths=[_write_extension(tmp_path)])
    case = _tiny_case()
    report = simulate_service(case, extensions=context, event_limit=8)
    assert report["schema"] == "mindie-sim.predict/2"
    assert report["status"] == "modeled"
    assert report["execution_status"] == "completed"
    assert report["production_entry"] == "simulate_service"
    assert report["extension_provider"] == "acme.wheel_demo"
    assert report["metrics"]["ttft_ms"]["count"] == 1
    assert report["metrics"]["total_requests"] == 1
    assert report["metrics"]["served_requests"] == 1
    execution = report["execution"]
    assert execution["live_tasks"] == 0
    assert execution["live_programs"] == 0
    assert execution["completed_tasks"] > 0
    assert execution["assembly"]["declared_commits"] >= 1
    assert execution["assembly"]["fallback_commits"] == 0
    assert execution["model_steps"] >= 1
    assert report["plan"]["model"]["name"] == "deepseek-v3.2-prune-3l"
    assert execution["serviced"]["memory@r0:s0"] > 0
    assert execution["serviced"].get("matrix@r0:s0", 0) > 0
    assert report["plan"]["runtime"]["parallel"]["tp"] == 1


@pytest.mark.requires_profiler
def test_profiler_replays_raw_capture_and_extension_rule_maps_it(tmp_path):
    from mindie_profiler import public
    from mindie_profiler.public import analyze_profile, compare_profiles
    from mindie_sim.public import load_extensions

    if os.environ.get("MINDIE_WHEEL_ONLY") == "1" or _site_package(mindie_sim):
        assert _site_package(public)
        _install_is_wheel("mindie-profiler")
    capture = _capture(tmp_path)
    core = analyze_profile(capture, full_range=True)
    assert core["schema_version"] == "mindie-profile-summary/1"
    top = core["top_hotspot_kernels"]
    assert top, "profiler did not expose a hotspot inventory"
    row = next(item for item in top if item["kernel_name"] == PROFILER_RULE)
    assert row["call_count"] == 2
    assert row["measured_device_ms_total"] == pytest.approx(0.030)
    assert row["measured_device_ms_max"] == pytest.approx(0.020)
    assert core["step_accounting"]["rank_scope"] == ["capture_r0"]
    context = load_extensions(paths=[_write_extension(tmp_path)])
    replayed = analyze_profile(capture, extensions=context, full_range=True)
    assert replayed["schema_version"] == "mindie-profile-summary/1"
    assert replayed["step_accounting"]["adapter"] == "acme.wheel_demo"
    assert replayed["source_identity"]["rule"] == PROFILER_RULE
    assert replayed["observed_total_us"] == pytest.approx(30.0)
    assert replayed["kernel_names"] == [PROFILER_RULE, PROFILER_RULE]
    assert replayed["extension_fingerprint"] == context.fingerprint
    comparison = compare_profiles({"predicted_us": 30.0}, {"observed_us": replayed["observed_total_us"]},
        extensions=context)
    assert comparison["schema_version"] == "mindie-profile-comparison/1"
    assert comparison["status"] == "pass"
    assert comparison["ratio"] == pytest.approx(1.0)
    assert comparison["items"][0]["name"] == "acme.matmul"


def _extension_op():
    """The same declaration the workspace extension makes (test-side copy)."""
    from mindie_sim.schema.operators import OpSpec
    return OpSpec("acme.matmul", "gemm", "decode", 0, 0, (), (M, K, N), (M, K, N), (M, K, N), (M, K, N),
        dtype="bf16", flops=FLOPS, hbm_bytes=HBM_BYTES, mte2_bytes=MTE2_BYTES, mte3_bytes=MTE3_BYTES,
        recipe="gemm", metadata=(("extension_id", "acme.wheel_demo"), ("semantic_op", "acme.matmul"),
            ("program_kind", "matmul"), ("physical_program", "SteadyMatmul"), ("activation_format", "bf16"),
            ("output_format", "bf16"), ("matrix_groups", 1), ("aclnn_api", ACLNN_ENTRY),
            ("aclnn_symbol", ACLNN_ENTRY), ("profiler_rule", PROFILER_RULE),),)


def test_two_layer_operator_sdk_simulation_honesty_lock_and_layer2_binding():
    from mindie_sim.public import (
        CapabilityGap,
        ExecutionEnvironment,
        ExtensionError,
        FormulaKernelSpec,
        FormulaTensorSpec,
        bind_kernel_implementation,
        load_extensions,
        simulate_aclnn_call,
        simulate_formula_kernel,
        simulate_graph,
        simulate_model,
        simulate_native_call,
        simulate_operator,
        simulate_service,
    )
    from mindie_sim.schema.hardware import Hardware
    from mindie_sim.schema.operators import OpSpec

    env = ExecutionEnvironment.from_hardware(Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582"}))
    spec = FormulaKernelSpec(
        name="FusedNormGateCustom",
        equations=("y = rmsnorm(x) * sigmoid(g)",),
        inputs=(
            FormulaTensorSpec("x", (16, 4096), dtype="bf16"),
            FormulaTensorSpec("g", (16, 4096), dtype="bf16"),
        ),
        outputs=(FormulaTensorSpec("y", (16, 4096), dtype="bf16", role="output"),),
        vector_ops_fp32=float(16 * 4096 * 8),
        row_reduction=True,
    )

    # Layer 1 (Modeling Mode): evaluates hardware SRAM & pipes, stamped unbound.
    eval_a = simulate_formula_kernel(spec, env)
    eval_b = simulate_formula_kernel(env, spec)
    assert eval_a.projected_us == pytest.approx(eval_b.projected_us)
    assert eval_a.projected_us >= eval_a.l_star_us > 0.0
    assert eval_a.sram_verified is True and eval_a.ub_peak_bytes <= eval_a.ub_budget_bytes
    assert eval_a.binding_state == "unbound_formula_projection" and eval_a.has_unbound_formula_ops is True

    # Simulation Mode Honesty Lock: rejects unbound FormulaKernelSpec and unbound OpSpec.
    unbound_op = OpSpec(
        "custom.fused_norm_gate", "vector", "decode", 0, 0, (), (65536,), (65536,), (65536,), (65536,),
        dtype="bf16", hbm_bytes=262144, vector_ops=524288,
        metadata=(("binding_state", "unbound_formula_projection"), ("unbound_formula_op", True)),
    )
    for call_fn, arg in (
        (simulate_operator, spec),
        (simulate_operator, unbound_op),
        (simulate_native_call, spec),
        (simulate_graph, (unbound_op,)),
    ):
        with pytest.raises(CapabilityGap) as exc_info:
            call_fn(env, arg)
        assert exc_info.value.code == "unbound_operator_implementation"
        assert exc_info.value.kind == "unmodeled_capability"

    # Simulation Mode also forbids passing allow_unbound_formula=True.
    for fn, args in (
        (simulate_operator, (env, ())),
        (simulate_native_call, (env, ())),
        (simulate_aclnn_call, (env,)),
        (simulate_graph, (env, ())),
        (simulate_model, (_tiny_case(),)),
        (simulate_service, (_tiny_case(),)),
    ):
        with pytest.raises(ExtensionError, match="simulation_mode_forbids_unbound_formula_ops"):
            fn(*args, allow_unbound_formula=True)

    # Layer 2: empty binding fails fast; real binding promotes op into passing Simulation Mode.
    with pytest.raises(ExtensionError, match="bind_kernel_implementation_requires_real_binding"):
        bind_kernel_implementation(spec)

    bound_op = OpSpec(
        "custom.fused_norm_gate", "vector", "decode", 0, 0, (), (65536,), (65536,), (65536,), (65536,),
        dtype="bf16", hbm_bytes=262144, vector_ops=524288,
    )
    ext = bind_kernel_implementation(spec, implementation={"ops": (bound_op,)}, capability="operator")
    ctx = load_extensions(extensions=[ext])
    sim_res = simulate_operator(env, (unbound_op,), extensions=ctx)
    assert sim_res["schema"] == "mindie-sim.operators/1" and sim_res["operator_sum_ms"] > 0.0

