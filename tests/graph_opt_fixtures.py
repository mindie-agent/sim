"""Shared immutable fixtures for the R01-R16 optimizer regressions.

[Role] Build typed projections, real Case/ModelSpec fixtures and hardware
  profiles once for the behavioral suites.
[Invariants] Fixtures are synthetic structural declarations, not product
  measurements; the canonical fixtures use the real catalog/compose path.
[Timeouts & Locks] Pure in-memory construction; bounded.
[Side-Effect Policy] Returns new records; no filesystem or environment changes.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.engine_plan import HardwareProfile
from mindie_sim.prediction.fusion_feasibility import CubeResources
from mindie_sim.prediction.fusion_graph import OpNode, TensorValue, TypedGraph

CUBE = CubeResources(l1=524288, l0a=65536, l0b=65536, l0c=131072)
CAPS = (("fixpipe_epilogue", True), ("cube_prologue", True),
        ("aclgraph_capture", True), ("aclgraph_prefetch", True),
        ("weight_prefetch", True))


def val(value_id, dtype, shape, *, role="activation", producer=None,
        storage=None, public=False, sharded=False, layout="contiguous",
        version=None, state_key="", alias_of=None):
    return TensorValue(value_id, dtype, shape, role=role, producer=producer,
                       storage_bytes=storage, public=public, sharded=sharded,
                       layout=layout, state_version=version, state_key=state_key,
                       alias_of=alias_of)


def op(op_id, kind, resource, inputs, outputs, *, work=(), params=(),
       deps=(), collective=False, state_effect="", capture_boundary="",
       stream="main", owner=0, numerical_policy="exact", phase="decode"):
    return OpNode(op_id, kind, resource, tuple(inputs), tuple(outputs),
                  deps=tuple(deps), work=tuple(work), params=tuple(params),
                  collective=collective, state_effect=state_effect,
                  capture_boundary=capture_boundary, stream=stream, owner=owner,
                  numerical_policy=numerical_policy, phase=phase)


def hw(**overrides):
    fields = dict(hbm_bytes_per_ms=1e9, vector_ops_per_ms=1e9,
                  cube_flops_per_ms=1e12, onchip_bytes_per_ms=4e9,
                  ub_bytes_per_aiv=262144, prefetch_capacity_bytes=16 * 1024 * 1024,
                  max_streams=2, launch_ms=0.0, device_min_ms=0.0,
                  tensor_parallel=1, capabilities=CAPS, cube_capacities=CUBE)
    fields.update(overrides)
    return HardwareProfile(**fields)


def external_graph(residual=False, mte1=0.0):
    rows = 16
    values = {
        "x1": val("x1", "bf16", (rows, 1), storage=32),
        "x2": val("x2", "bf16", (rows, 1), storage=32),
        "h": val("h", "bf16", (rows, 1), producer="add", storage=32),
        "g": val("g", "bf16", (1,), role="weight", storage=8),
        "n": val("n", "bf16", (rows, 1), producer="norm", storage=32),
        "q": val("q", "int8", (rows, 1), producer="quant", storage=32),
        "s": val("s", "fp32", (rows, 1), role="scale", producer="quant", storage=8),
    }
    add_outputs = ("h",)
    public = ("q", "s")
    if residual:
        values["r"] = val("r", "fp32", (rows, 1), role="residual",
                          producer="add", storage=64)
        add_outputs = ("h", "r")
        public = ("q", "s", "r")
    ops = {
        "add": op("add", "add", "vector", ("x1", "x2"), add_outputs,
                  work=(("vector", 100.0), ("mte1", mte1))),
        "norm": op("norm", "rms_norm", "vector", ("h", "g"), ("n",),
                   work=(("vector", 200.0),),
                   params=(("epsilon", 1e-6), ("reduction_order", "declared"))),
        "quant": op("quant", "dynamic_quant", "vector", ("n",), ("q", "s"),
                    work=(("vector", 50.0),),
                    params=(("axis", -1), ("group_size", 8),
                            ("rounding", "rne"), ("saturation", "sat"))),
    }
    return TypedGraph(rows, values, ops, public)


def hybrid_graph():
    rows, hidden, out = 16, 4096, 4096
    values = {
        "x1": val("x1", "bf16", (rows, hidden)),
        "x2": val("x2", "bf16", (rows, hidden)),
        "h": val("h", "bf16", (rows, hidden), role="residual", producer="add"),
        "g": val("g", "bf16", (hidden,), role="weight"),
        "n": val("n", "bf16", (rows, hidden), producer="norm"),
        "q": val("q", "int8", (rows, hidden), producer="quant"),
        "s": val("s", "fp32", (rows, 1), role="scale", producer="quant"),
        "W_qb": val("W_qb", "bf16", (out, hidden), role="weight"),
        "proj": val("proj", "bf16", (rows, out), producer="matmul"),
        "proj_b": val("proj_b", "fp16", (rows, out), producer="cast"),
        "idx": val("idx", "int32", (rows, 128), role="index", producer="indexer"),
        "out": val("out", "bf16", (rows, out), role="output", producer="join"),
    }
    ops = {
        "add": op("add", "add", "vector", ("x1", "x2"), ("h",),
                  work=(("vector", 65536),)),
        "norm": op("norm", "rms_norm", "vector", ("h", "g"), ("n",),
                   work=(("vector", 393216),),
                   params=(("epsilon", 1e-6), ("reduction_order", "declared"))),
        "quant": op("quant", "dynamic_quant", "vector", ("n",), ("q", "s"),
                    work=(("vector", 196608),),
                    params=(("axis", -1), ("group_size", 128),
                            ("rounding", "rne"), ("saturation", "sat"))),
        "matmul": op("matmul", "matmul", "cube", ("q", "W_qb"), ("proj",),
                     work=(("cube", 2 * rows * out * hidden),)),
        "cast": op("cast", "cast", "vector", ("proj",), ("proj_b",),
                   work=(("vector", 65536),),
                   params=(("target_dtype", "fp16"), ("rounding", "rne"))),
        "indexer": op("indexer", "topk", "vector", ("q",), ("idx",),
                      work=(("vector", 1e6),)),
        "join": op("join", "elementwise", "vector", ("proj_b", "idx"), ("out",),
                   work=(("vector", 65536),)),
    }
    return TypedGraph(rows, values, ops, ("h", "q", "s", "out"))


def cast_chain_graph(prefix, *, work_a=5.0, work_b=5.0):
    values = {
        f"{prefix}x": val(f"{prefix}x", "bf16", (1, 2048), storage=4096),
        f"{prefix}y": val(f"{prefix}y", "bf16", (1, 2048), producer=f"{prefix}a",
                          storage=4096),
        f"{prefix}z": val(f"{prefix}z", "bf16", (1, 2048), producer=f"{prefix}b",
                          storage=4096),
    }
    ops = {
        f"{prefix}a": op(f"{prefix}a", "cast", "vector", (f"{prefix}x",),
                         (f"{prefix}y",), work=(("vector", work_a),),
                         params=(("target_dtype", "bf16"), ("rounding", "rne"))),
        f"{prefix}b": op(f"{prefix}b", "cast", "vector", (f"{prefix}y",),
                         (f"{prefix}z",), work=(("vector", work_b),),
                         params=(("target_dtype", "bf16"), ("rounding", "rne"))),
    }
    return values, ops


def ub_contention_graph():
    v1, o1 = cast_chain_graph("a", work_a=5.0, work_b=5.0)
    v2, o2 = cast_chain_graph("b", work_a=0.1, work_b=0.1)
    values = {**v1, **v2,
              "out": val("out", "bf16", (1, 2048), storage=4096,
                         producer="join")}
    ops = {**o1, **o2,
           "join": op("join", "add", "vector", ("az", "bz"), ("out",),
                      work=(("vector", 0.1),))}
    return TypedGraph(1, values, ops, ("out",))


def stream_graph(compute_heavy=True):
    weight = 1.2e12 if compute_heavy else 0.1e12
    vector = 0.6e9 if compute_heavy else 0.1e9
    values = {
        "ia": val("ia", "bf16", (1,), storage=int(0.8e9)),
        "w": val("w", "bf16", (1,), role="weight", storage=8),
        "oa": val("oa", "bf16", (1,), producer="a", storage=8),
        "ib": val("ib", "bf16", (1,), storage=int(0.8e9)),
        "ob": val("ob", "bf16", (1,), producer="b", storage=8),
        "out": val("out", "bf16", (1,), producer="join", storage=8),
    }
    ops = {
        "a": op("a", "matmul", "cube", ("ia", "w"), ("oa",),
                work=(("cube", weight),)),
        "b": op("b", "silu", "vector", ("ib",), ("ob",),
                work=(("vector", vector),)),
        "join": op("join", "elementwise", "vector", ("oa", "ob"), ("out",),
                   work=(("vector", 0.1),)),
    }
    return TypedGraph(1, values, ops, ("out",))


def prefetch_graph(*, weight_bytes=8192, predecessor=10.0):
    values = {
        "x": val("x", "bf16", (1,), storage=8),
        "h": val("h", "bf16", (1,), producer="pre", storage=8),
        "W": val("W", "bf16", (1,), role="weight", storage=weight_bytes),
        "y": val("y", "bf16", (1,), producer="mm", storage=8),
    }
    ops = {
        "pre": op("pre", "cast", "vector", ("x",), ("h",),
                  work=(("vector", predecessor),),
                  params=(("target_dtype", "bf16"), ("rounding", "rne"))),
        "mm": op("mm", "matmul", "cube", ("h", "W"), ("y",),
                 work=(("cube", 1.0),)),
    }
    return TypedGraph(1, values, ops, ("y",))


def quant_graph(axis=-1, group=8, target="bf16", layout="contiguous"):
    values = {
        "x": val("x", "bf16", (16, 64), storage=2048),
        "q": val("q", "int8", (16, 64), producer="quant", storage=1024),
        "s": val("s", "fp32", (16, 1), role="scale", producer="quant", storage=64),
        "y": val("y", target, (16, 64), producer="cast", storage=2048,
                 layout=layout),
    }
    ops = {
        "quant": op("quant", "dynamic_quant", "vector", ("x",), ("q", "s"),
                    work=(("vector", 1.0),),
                    params=(("axis", axis), ("group_size", group),
                            ("rounding", "rne"), ("saturation", "sat"))),
        "cast": op("cast", "cast", "vector", ("q",), ("y",),
                   work=(("vector", 1.0),),
                   params=(("target_dtype", target), ("rounding", "rne"))),
    }
    return TypedGraph(16, values, ops, ("y",))


def scalar_cube_graph():
    values = {
        "a": val("a", "bf16", (1, 1), storage=2),
        "w": val("w", "bf16", (1, 1), role="weight", storage=2),
        "p": val("p", "bf16", (1, 1), producer="mm", storage=2),
        "c": val("c", "fp16", (1, 1), producer="cast", storage=2),
    }
    ops = {
        "mm": op("mm", "matmul", "cube", ("a", "w"), ("p",),
                 work=(("cube", 2.0),)),
        "cast": op("cast", "cast", "vector", ("p",), ("c",),
                   work=(("vector", 1.0),),
                   params=(("target_dtype", "fp16"), ("rounding", "rne"))),
    }
    return TypedGraph(1, values, ops, ("c",))


def state_graph(*, missing_effect=False, unordered=False):
    values = {
        "s0": val("s0", "fp32", (1,), role="state", storage=4,
                  version=0, state_key="k"),
        "s1": val("s1", "fp32", (1,), role="state", producer="upd",
                  storage=4, version=1, state_key="k"),
        "s2": val("s2", "fp32", (1,), role="state", producer="upd2",
                  storage=4, version=2, state_key="k"),
        "x": val("x", "bf16", (16, 8), storage=256),
        "y": val("y", "bf16", (16, 8), producer="upd", storage=256),
        "z": val("z", "bf16", (16, 8), producer="upd2", storage=256),
    }
    ops = {
        "upd": op("upd", "state_update", "vector", ("x", "s0"), ("y", "s1"),
                  state_effect="" if missing_effect else "read_write",
                  params=(("state_key", "k"),)),
        "upd2": op("upd2", "state_update", "vector", ("y",), ("z", "s2"),
                   deps=() if unordered else ("upd",),
                   state_effect="write", params=(("state_key", "k"),)),
    }
    return TypedGraph(16, values, ops, ("z", "s2"))


def budget_graph(chains=12):
    values, ops = {}, {}
    for index in range(chains):
        v, o = cast_chain_graph(f"c{index}")
        values.update(v)
        ops.update(o)
    return TypedGraph(1, values, ops, tuple(f"c{index}z" for index in range(chains)))


def canonical_qwen():
    from mindie_sim import compose_case, target_graphs
    raw = compose_case("Qwen3.8-27B", hardware="a3", tp=4, layers=1,
                       quant="bf16", va_recommended=True)
    ops = target_graphs(raw, scope="body", phase="decode", tokens=16)[0]
    return raw, ops


def canonical_mlapo():
    from mindie_sim import compose_case, target_graphs
    raw = compose_case("deepseek-v3.2", hardware="a3", tp=4, layers=1,
                       va_recommended=True)
    raw["runtime"]["kernels"]["lowerings"] = {
        "mla_prolog": {"implementation": "vllm_ascend_mla_prolog_v3"},
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["mla_prolog"]}},
    }
    ops = target_graphs(raw, scope="body", phase="decode", tokens=6,
                        sequences=8, context=4104,
                        apply_body_defaults=False)[0]
    return raw, ops


def replace_state(op_node, **fields):
    return replace(op_node, **fields)
