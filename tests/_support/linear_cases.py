"""Shared builders for linear-attention, causal-conv, and template tests.

[Role] Provide compact, single-definition case, caller, execution, and
  planning-context builders used across ``tests/operators/`` linear-attention,
  causal-conv, and template suites, plus retargeted legacy callers.
[Invariants] Zero cross-test imports, zero host-absolute path literals, zero
  mutated global state, and deterministic in-memory construction.
[Timeouts & Locks] Pure in-memory construction and bounded single-op execution;
  no sleeps, locks, or network access. Provenance fixture Git calls time out.
[Side-Effect Policy] Read packaged fixtures and write provenance test data under temporary roots.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from .hardware import machine
from mindie_sim import target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operators.kda.family_kda_recurrent import IMPLEMENTATION as KDA_REC_IMPL
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.linear_attention._gdn_decode_programs import CONV, GATING
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.program import RecipeContext, Stage
from mindie_sim.prediction.operators.recurrent.recurrent import linear
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.runtime.memory_physics import AdmissionMemoryView
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Software
from mindie_sim.schema.compute_arch import NativeComputeMode, RepresentationCapability
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.templates import (
    EdgeRouteBinding, EvgVisitorNode, TemplateSourceRef, compile_fuse_region, enumerate_catlass_specs, enumerate_dsl_specs,
    lower_dsl_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_KDA_CASE_DIR = "examples/cases/high-concurrency-20261030/kimi-source-paged-host"


def kda_hardware_case(
    *,
    units: int = 2,
    split_copy: bool = False,
    rates: dict[str, float] | None = None,
    compute_bf16: float | None = None,
) -> dict[str, Any]:
    """Return a minimal single-device mapping for KDA source controls."""
    base: dict[str, float] = {
        "memory": 1e12,
        "vector": 2e12,
        "mte1": 1e12,
        "fixpipe": 1e12,
        "mte3": 1e12,
    }
    resource_units: dict[str, int] = {"vector": 2 * units}
    if split_copy:
        resource_units.update(mte2_aic=units, mte2_aiv=2 * units)
        base.update(mte2_aic=1e12, mte2_aiv=2e12)
    else:
        base["mte2"] = 1e12
    if rates:
        base.update(rates)
    return {
        "machine": {
            "name": "kda-source-control",
            "num_devices": 1,
            "device": {
                "memory_bytes": 64 * 1024 ** 3,
                "compute_units": units,
                "compute_peaks": {
                    "bf16": compute_bf16 or units * 1e14,
                    "fp32": units * 2.5e13,
                },
                "rates": base,
                "resource_units": resource_units,
            },
        },
        "model": {
            "name": "geometry",
            "layers": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 8,
            "intermediate_size": 16,
            "vocab_size": 16,
        },
        "workload": {
            "requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}],
        },
        "runtime": {},
    }


def _base_recurrent_case():
    parsed = parse_case({
        "machine": {
            "name": "declared-gpu",
            "architecture": "declared",
            "num_devices": 1,
            "device": {
                "memory_bytes": 1024 ** 3,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                "rates": {"memory": 1e9, "vector": 1e9},
                "compute_units": 4,
            },
            "links": [{
                "name": "fabric",
                "capacity_bytes_per_ms": 1000,
                "members": [0],
                "semantics": "shared",
            }],
        },
        "model": {
            "name": "geometry",
            "layers": 2,
            "hidden_size": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
        },
        "workload": {
            "requests": [{"id": "r", "prompt_tokens": 4, "output_tokens": 3}],
        },
        "runtime": {
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "ep": 1,
            "max_num_batched_tokens": 16,
        },
    })
    layer = replace(
        parsed.model.layer_plans[0],
        attention="gdn",
        linear_key_heads=1,
        linear_value_heads=2,
        linear_heads=2,
        linear_key_head_dim=3,
        linear_head_dim=2,
        conv_kernel=3,
        recurrent_state_dtype="fp32",
    )
    return replace(
        parsed,
        model=replace(parsed.model, layers=1, layer_plans=(layer,)),
        runtime=replace(parsed.runtime, pp_layer_partition=(1,)),
    )


def gdn_decode_source_case(
    lengths: tuple[int, ...] = (4, 4, 4, 4),
    *,
    keys: int = 4,
    heads: int = 12,
    units: int = 64,
    queue: bool = False,
):
    """Build a GDN decode body case and op sequence."""
    parsed = _base_recurrent_case()
    layer = replace(
        parsed.model.layer_plans[0],
        linear_key_heads=keys,
        linear_value_heads=heads,
        linear_heads=heads,
        linear_key_head_dim=128,
        linear_head_dim=128,
        conv_kernel=4,
    )
    kernels: dict[str, Any] = {"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}
    rates = {
        "hbm": 4e12,
        "vector": 1e13,
        "mte2": 4e12,
        "mte2_aic": 4e12,
        "mte2_aiv": 4e12,
        "mte3": 4e12,
    }
    if queue:
        rates["scalar_store_issue"] = 6.4e9
        kernels["lowerings"]["gdn_decode"] = {
            "implementation": "vllm_ascend_gdn_decode",
            "options": {
                "gating_beta_store": {
                    "resource": "scalar_store_issue",
                    "queue": {
                        "outstanding": 8,
                        "granule_bytes": 128,
                        "response_ms": 0.0001,
                    },
                },
            },
        }
    device = replace(
        parsed.machine.dies[0],
        rates=tuple(rates.items()),
        compute_units=units // 2,
        resource_units=tuple((r, units) for r in rates),
    )
    parsed = replace(
        parsed,
        model=replace(parsed.model, layer_plans=(layer,)),
        runtime=replace(parsed.runtime, kernels=kernels),
        machine=replace(parsed.machine, dies=(device,)),
    )
    views = tuple(
        RequestView(str(i), q, 4096, False, "decode")
        for i, q in enumerate(lengths)
    )
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    return parsed, views, ops


def execute_one(parsed, views, op):
    """Execute one op behind a delayed ready event and assert zero leaks."""
    op = replace(op, deps=("ready",))
    execution = Execution(
        parsed, CostModel(parsed), include_submission=False, trace_limit=10000,
    )
    ready = execution.engine.submit("ready", release=2)
    complete = execution.add(
        IterationWork((op,), IterationPlan(views), 1, 0, 0, None, "body", "test"),
        0,
        0,
        external_inputs={"ready": ready},
    ).completion
    execution.engine.advance(horizon=1)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    assert not execution.engine._tasks and not execution.engine._promises
    return execution.engine, complete.time - 2


def caller(
    lengths: tuple[int, ...] = (1, 4),
    *,
    padded: int | None = None,
    dv: int = 128,
    options: dict[str, Any] | None = None,
    phase: str = "decode",
):
    """Build an ordinary KDA caller case and lowered linear op list."""
    model = load_model("kimi-k3")
    plan = next(p for p in model.layer_plans if p.attention == "kda")
    plan = replace(
        plan,
        linear_key_heads=1,
        linear_value_heads=2,
        linear_heads=2,
        linear_head_dim=dv,
        linear_key_head_dim=128,
    )
    model = replace(
        model,
        layers=1,
        hidden_size=512,
        layer_plans=(plan,),
        quant=quant_preset("bf16"),
    )
    runtime = Software.from_mapping({
        "kernels": {
            "lowerings": {
                "kda": {
                    "implementation": KDA_REC_IMPL,
                    "options": options or {},
                },
            },
        },
    })
    views = tuple(
        RequestView(str(i), q, 2048, False, phase)
        for i, q in enumerate(lengths)
    )
    ops = linear(
        prefix="kda",
        plan=plan,
        views=views,
        stage=0,
        deps="input",
        model=model,
        parallel=ParallelSpec(),
        quant=model.quant,
        mode=phase,
        padded_tokens=padded or sum(lengths),
        policy=AttentionPolicy(),
        runtime=runtime,
    ).ops
    parsed = parse_case(kda_hardware_case(units=4, split_copy=True))
    die = replace(
        parsed.machine.dies[0],
        buffers=(("ub", 256 * 1024),),
        buffer_units=(("ub", 8),),
    )
    return (
        replace(
            parsed,
            model=model,
            runtime=runtime,
            machine=replace(parsed.machine, dies=(die,)),
        ),
        ops,
    )


def packed_roles(ops):
    """Return packed KDA input projection op and its declared role slices."""
    packed = next(op for op in ops if op.recipe == "linear_qkvbfg_a")
    slices = {
        row["role"]: row["shape"]
        for row in dict(packed.metadata)["packed_projection_slices"]
    }
    return packed, slices


def case(batch: int, *, native: bool = True) -> dict[str, Any]:
    """Load the paged KDA decode example case with optional native conv selector."""
    path = REPO_ROOT / _KDA_CASE_DIR / f"a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if native:
        raw["runtime"]["kernels"]["lowerings"]["causal_conv_update"] = CONV
    return raw


def conv_graph(raw: Any, batch: int):
    """Build target decode body ops for a KDA conv update case."""
    return target_graphs(
        raw,
        scope="body",
        phase="decode",
        tokens=1,
        context=4096,
        sequences=batch,
        apply_body_defaults=True,
    )[0]


def sku_planning_context(
    sku: str,
    op_id: str = "op0",
    *,
    requested_abi: str | None = None,
) -> tuple[Geometry, OperatorPlanningContext]:
    """Build a real Geometry and OperatorPlanningContext from a packaged SKU."""
    geom = Geometry.from_devices(machine(sku).dies)
    dp = geom.datapath
    ctx = OperatorPlanningContext(
        occurrence=OpOccurrence(replica=0, stage=0, rank=0, op_id=op_id),
        geometry=geom,
        datapath=dp,
        requested_abi=requested_abi,
    )
    return geom, ctx


def recipe_context(geometry: Geometry, op) -> RecipeContext:
    """Bind an isolated program to the real geometry with readable resource names."""
    rates = {**geometry.rates, "cube": 1000}
    return RecipeContext(
        op, {}, {role: role for role in rates}, rates, geometry.compute_units,
        resource_units=geometry.resource_units, compute_peaks=geometry.compute_peaks,
        pool_names={pool: pool for pool in geometry.buffers}, pool_units=geometry.buffer_units,
        datapath=geometry.datapath,
    )


__all__ = [
    "CONV",
    "GATING",
    "caller",
    "case",
    "conv_graph",
    "execute_one",
    "gdn_decode_source_case",
    "kda_hardware_case",
    "packed_roles",
    "recipe_context",
    "sku_planning_context",
]


def lower_reference(name, p, dp, *, live_rates=None, **options):
    spec, = enumerate_dsl_specs(name, p, dp, **{"aic_budget": 8, "aiv_budget": 16, **options})
    bindings = template_live_bindings(dp, p, **live_rates) if live_rates is not None else {}
    return spec, lower_dsl_spec(spec, p, dp, **bindings)

def stages_by_name(result):
    return {stage.name: stage for stage in result.program.stages}

def stage_ancestors(stages, name):
    seen, pending = set(), list(stages[name].after)
    while pending:
        parent = pending.pop()
        if parent not in seen:
            seen.add(parent)
            pending.extend(stages[parent].after)
    return seen

def stage_bytes(stages, suffix, pipe):
    return sum(stage.work.get(pipe, 0) for name, stage in stages.items() if name.endswith(suffix))

def template_context(dp, p, **extra_rates):
    units = dict.fromkeys(("cube", "mte1", "mte2_aic", "fixpipe"), dp.aic_count)
    units.update(dict.fromkeys(("mte2_aiv", "mte3", "vec_reduce"), dp.aiv_count), hbm=1, l2=1)
    units.update(dict.fromkeys(extra_rates.keys() - units.keys(), dp.aiv_count))
    rates = dict.fromkeys(units, 10**6) | extra_rates
    return RecipeContext(p, {}, dict(zip(units, units)), rates, dp.aic_count, resource_units=units,
                         compute_peaks=dict.fromkeys(("bf16", "fp16"), 1000 * rates["cube"]),
                         pool_names={pool: pool for pool, _ in dp.buffers})

def run_template(dp, context, result, *, trace_limit=0):
    resources = [Resource(role, context.rates[role], units=units)
                 for role, units in context.resource_units.items() if role != "hbm"]
    pools = dict(dp.buffers)
    engine = Engine(resources, pools=pools, backend="python", trace_limit=trace_limit, pool_units={
        pool: 1 if pool == "l2" else dp.aiv_count if pool == "ub" else dp.aic_count for pool in pools
    })
    done = result.program.start(engine, "matrix", engine.submit("ready", release=1))
    engine.advance(until=(done,))
    assert done.time > 1 and not engine.allocations
    return engine

def tile_edge(route="l0c->ub", axes=("aic", "aiv"), dst="take", view="tile"):
    return EdgeRouteBinding("make", 0, view, dst, "tile_pipelined", route, *axes, "fp32", "RowMajor", 2)

def compile_tile_region(dp, edge, **overrides):
    return compile_fuse_region("fused", ("make", "take"), dp, **{
        "dependencies": {"make": (), "take": ("make",)}, "edge_routes": (edge,), "edge_bytes": {edge.view_id: 64},
        "op_stages": {name: Stage(name) for name in ("make", "take")}, **overrides,
    })

def load_template_provenance(monkeypatch):
    spec = spec_from_file_location("template_provenance", REPO_ROOT / "scripts" / "check_template_provenance.py")
    audit = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, audit)
    spec.loader.exec_module(audit)
    return audit


def provenance_git_fixture(root, relative_path, blob, symbol, line_range):
    repo = root / "upstream"
    path = repo / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(blob)

    def git(*args):
        return subprocess.check_output(("git", "-C", str(repo), *args), text=True, timeout=5).strip()

    git("init", "--quiet")
    git("add", "--", relative_path)
    git("-c", "user.name=Provenance Test", "-c", "user.email=fixture@example.test",
        "commit", "--quiet", "--no-gpg-sign", "-m", "Source fixture")
    bare = root / "bare.git"
    git("clone", "--quiet", "--bare", str(repo), str(bare))
    ref = TemplateSourceRef("test/upstream", git("rev-parse", "HEAD"), relative_path,
                            sha256(blob).hexdigest(), symbol, line_range)
    return repo, bare, ref, git("rev-parse", "HEAD^{tree}")


def verify_template_declaration(audit, root, ref, repo=None):
    path = root / "mindie_sim" / "templates" / "fixture.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"TemplateSourceRef{tuple(ref.values())!r}\n", encoding="utf-8")
    repo_map = {ref["repository"]: repo} if repo is not None else {}
    _, findings = audit.verify(root, runtime=False, repo_map=repo_map)
    return [f.reason.split(":")[0] for f in findings]


def enumerate_matrix(name, p, dp, *, tile=(32, 32, 32), **options):
    return enumerate_catlass_specs(p, dp, template_ids=("catlass." + name,), l1_tiles=(tile,),
                                   **{"aic_budget": 4, "aiv_budget": 8, **options})


def with_mx_modes(direct):
    arch = direct.compute_arch.declared
    formats = (("mxfp8", 8), ("mxfp4", 4), ("a8w4_mx", 4))
    arch = replace(arch, representations=(*arch.representations, *(RepresentationCapability(name, bits, ("operand",))
                   for name, bits in formats)), native_modes=(*arch.native_modes, *(
                       NativeComputeMode(name, "matrix", (name,), "cube", ("fp32",)) for name, _ in formats)))
    return replace(direct, compute_arch=replace(direct.compute_arch, declared=arch))


def visitor_graph():
    roots = (EvgVisitorNode("acc", "acc_load", "fp32"), *(EvgVisitorNode(kind, kind, "bf16", aux_tensor_id=kind)
              for kind in ("aux_load", "row_broadcast", "col_broadcast")))
    unary = ("exp", "relu", "sqrt", "rsqrt_fast", "leaky_relu", "silu", "cast", "muls")
    chain = tuple(EvgVisitorNode(kind, kind, "fp32", inputs=(parent,), scalar_param=0.25)
                  for kind, parent in zip(unary, ("acc", *unary)))
    tree = (*roots, *chain, EvgVisitorNode("mul", "mul", "fp32", inputs=("muls", "row_broadcast")),
            EvgVisitorNode("add", "add", "fp32", inputs=("mul", "col_broadcast")),
            EvgVisitorNode("store", "aux_store", "bf16", inputs=("add",), aux_tensor_id="aux_output"))
    return tree, chain[0]


def serpentine(major, minor):
    bands = [range(start, min(start + 2, major)) for start in range(0, major, 2)]
    return [(i, j) for band, strip in enumerate(bands)
            for j in (range(minor) if band % 2 == 0 else reversed(range(minor))) for i in strip]


def flash_kda_op(problem):
    metadata = dict(lengths=problem.lengths, hv=problem.value_heads, hq=problem.query_heads,
                    device_name=problem.device_name, has_initial_state=problem.has_initial_state,
                    return_final_state=problem.return_final_state, save_chunk_states=problem.save_chunk_states)
    return OpSpec("kda", "kda_prefill", "prefill", 0, 0, (), (), (), (), (), dtype="bf16", layout="BNSD",
                  recipe="kda_prefill", metadata=tuple(metadata.items()))


def template_datapath(sku):
    return datapath_for_die(machine(sku).dies[0])


def template_live_bindings(dp, problem, **rates):
    return dict(context=template_context(dp, problem, **rates),
                memory_view=AdmissionMemoryView({0: dp.usable_bytes("l2")}), live_admission=True)


def mounted_catalog_sources(root, refs):
    repositories = {ref.repository.split("/")[-1] for ref in refs}
    candidates = (parent / "refs" for parent in root.parents)
    return next((path for path in candidates if all((path / repo).is_dir() for repo in repositories)), None)


def recurrent_scan_context(tokens=10, chunk=4, *, setup=0, minimum=0):
    chunks = (tokens + chunk - 1) // chunk
    op = OpSpec("scan", "recurrent_scan", "prefill", 0, 0, (), (tokens,),
                (tokens,), (tokens,), (tokens,), metadata=(
                    ("chunks", chunks), ("chunk", chunk), ("tokens", tokens),
                    ("heads", 4), ("chunk_vector_ops", 100.0)))
    return SimpleNamespace(
        op=op, work={"cube": 1000.0, "vector": 500.0, "hbm": 300.0},
        resource_names={"cube": "cube", "vector": "vector"},
        rates={"cube": 1000.0, "vector": 1000.0}, core_count=4,
        minimum_ms=minimum, setup_ms=setup, minimum_provenance={},
        native_work={}, service_identity={},
    )
