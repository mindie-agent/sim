"""Homogeneous layer runs cut at every mandatory boundary and preserve work."""
from dataclasses import replace

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.layer_runs import (
    LayerRunProgram, homogeneous_layer_runs, layer_templates, run_summary,
)
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


def _graph(model, layers=8):
    case = parse_case(compose_case(
        model, "a3", tp=1, layers=layers, prompt_tokens=16,
        output_tokens=2, concurrency=1))
    return case, target_graphs(case, phase="decode", tokens=1, context=16,
                               sequences=1, scope="body")[0]


class _DelayProgram:
    def __init__(self, latency_ms):
        self.latency_ms = float(latency_ms)

    def start(self, engine, name, ready, *, owner=""):
        task = engine.submit(name, {"host": self.latency_ms}, after=(ready,),
                             owner=owner)
        return task


def test_homogeneous_dense_layers_fold_into_one_run():
    _case, ops = _graph("qwen3-32b", layers=8)
    runs = homogeneous_layer_runs(ops)
    assert runs
    largest = max(runs, key=lambda run: run.count)
    assert largest.count >= 2
    templates = layer_templates(ops)
    for run in runs:
        assert templates[run.start:run.start + run.count]
    summary = run_summary(runs)
    assert summary["basis"] == "structural_layer_template_reuse_not_duration_multiplication"
    assert summary["folded_layers"] >= 2


def test_hybrid_attention_schedule_cuts_runs():
    _case, ops = _graph("qwen3.6-35b-a3b", layers=8)
    runs = homogeneous_layer_runs(ops)
    # A GDN/KDA layer is never folded with a full-attention layer.
    for run in runs:
        attentions = {
            int(op.layer) for op in ops
            if int(op.layer) in run.layers and op.family in {"attention", "recurrent_scan"}
        }
        assert len(attentions) <= 1


class _SyntheticOp:
    """Minimal physical operator declaration for run identification."""

    def __init__(self, layer):
        self.layer = int(layer)
        self.stage = 0
        self.stream = "main"
        self.dtype = "bf16"
        self.metadata = (("structure_block", "attention"),
                         ("structure_role", "residual"),
                         ("payload_producer", f"s0.l{layer}.residual"))

    def to_dict(self):
        return {"op_id": f"s0.l{self.layer}.residual", "deps": (),
                "stage": 0, "layer": self.layer, "requests": (),
                "stream": "main", "notes": (), "family": "elementwise",
                "recipe": "residual_add", "dtype": self.dtype,
                "local_shape": (1, 16), "metadata": dict(self.metadata)}


def test_run_program_sequences_occurrences_without_multiplying_time():
    from mindie_sim.prediction.runtime.engine import Engine
    from mindie_sim.prediction.runtime.engine_types import Resource

    engine = Engine([Resource("host", 1.0, "ms", "fifo")])
    run = homogeneous_layer_runs([_SyntheticOp(index) for index in range(3)])[0]
    programs = tuple(_DelayProgram(2.0) for _ in run.layers)
    ready = engine.submit("ready", release=0.0)
    completion = LayerRunProgram(run=run, occurrences=programs).start(
        engine, "run", ready)
    engine.advance(until=(completion,))
    assert completion.time == pytest.approx(2.0 * len(run.layers))
    with pytest.raises(ModelingError, match="engine_certification"):
        LayerRunProgram(run=run, occurrences=programs, skip_authorized=True)
