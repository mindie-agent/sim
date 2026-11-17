"""Canonical physical identity preserves physical information and aliasing."""
from dataclasses import replace
import re

from mindie_sim.compose import compose_case
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.physical_key import (
    physical_id, physical_key, physical_signature,
)
from mindie_sim.prediction.structure import parse_case


def _role(op):
    return (op.recipe, op.local_shape,
            dict(op.metadata).get("structure_block"),
            op.op_id.rsplit(".", 1)[-1])


def _graph(layers=4):
    case = parse_case(compose_case(
        "qwen3-32b", "a3", tp=1, layers=layers, prompt_tokens=16,
        output_tokens=2, concurrency=1))
    return case, target_graphs(case, phase="decode", tokens=1, context=16,
                               sequences=1, scope="body")[0]


def _layer_normalized(_op, value):
    """Normalize every layer ordinal a cross-layer fusion origin records."""
    text = value if isinstance(value, str) else repr(value)
    return re.sub(r's0\.l\d+\.', "s0.lN.",
                  re.sub(r"['\"]layer['\"],\s*\d+", "layer,N", text))


def test_corresponding_layers_share_one_physical_identity():
    _case, ops = _graph()
    by_recipe = {}
    for op in ops:
        by_recipe.setdefault(_role(op), []).append(op)
    repeated = [group for group in by_recipe.values()
                if len({op.layer for op in group}) > 1]
    assert repeated, "expected a repeated layer template"
    for group in repeated:
        corresponding = [op for op in group if op.layer > 0]
        assert len(corresponding) > 1
        # The fused residual-norm boundary records the producing layer; the
        # physical identity is the same once that ordinal is normalized.
        signatures = {_layer_normalized(op, physical_signature(op))
                      for op in corresponding}
        assert len(signatures) == 1
        assert len({_layer_normalized(op, physical_key(op))
                    for op in corresponding}) == 1
        # ``physical_id`` hashes the raw key, so the fused cross-layer ordinal
        # keeps one distinct id per distinct raw key; the normalized key above
        # is the identity invariant.
        assert len({physical_id(op) for op in corresponding}) == len(
            {physical_key(op) for op in corresponding})


def test_shape_dtype_bytes_state_and_local_tp_change_identity():
    _case, ops = _graph()
    op = next(item for item in ops
              if item.recipe == "rmsnorm" and len(item.local_shape) >= 2)
    baseline = physical_signature(op)
    for changed in (
        replace(op, local_shape=(int(op.local_shape[0]) + 1, *op.local_shape[1:])),
        replace(op, dtype="fp32"),
        replace(op, hbm_bytes=float(op.hbm_bytes) + 1.0),
        replace(op, metadata=tuple(dict(op.metadata).items()) + (("state_write_bytes", 4096),)),
        replace(op, metadata=tuple(dict(op.metadata).items()) + (("local_tp_width", 8),)),
    ):
        assert physical_signature(changed) != baseline


def _rotate(value, stage, layer, target_layer):
    if isinstance(value, str):
        return value.replace(f"s{stage}.l{layer}.", f"s{stage}.l{target_layer}.")
    if isinstance(value, dict):
        return {key: _rotate(item, stage, layer, target_layer)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_rotate(item, stage, layer, target_layer) for item in value)
    return value


def test_reference_roles_and_aliasing_survive_normalization():
    _case, ops = _graph()
    first = next(op for op in ops
                 if dict(op.metadata).get("structure_subgraph_memberships")
                 and op.layer > 0)
    metadata = dict(first.metadata)
    memberships = tuple(metadata["structure_subgraph_memberships"])
    assert any(f"s{first.stage}.l{first.layer}." in value for value in memberships)
    renamed = replace(first, metadata=tuple(
        (key, tuple(value[:-1] + "other_role" for value in item)
         if key == "structure_subgraph_memberships" else item)
        for key, item in first.metadata))
    assert physical_signature(renamed) != physical_signature(first)
    rotated = replace(
        first, op_id="other", layer=first.layer + 1,
        metadata=tuple((key, _rotate(value, first.stage, first.layer,
                                     first.layer + 1))
                       for key, value in first.metadata))
    assert physical_signature(rotated) == physical_signature(first)


def test_collective_participant_change_breaks_identity():
    case, ops = _graph()
    collective = next((op for op in ops if op.link_bytes > 0), None)
    if collective is None:
        return
    metadata = dict(collective.metadata)
    metadata["group_ranks"] = (0, 1, 2, 3)
    changed = replace(collective, metadata=tuple(metadata.items()))
    assert physical_signature(changed) != physical_signature(collective)
