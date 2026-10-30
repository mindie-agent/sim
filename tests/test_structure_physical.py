"""Resource identity separates repeat instances from different physical work."""
from dataclasses import replace

from mindie_sim.catalog.ops import vector_op
from mindie_sim.structure import export_op_structure
from mindie_sim.structure_physical import node_physical


def test_occurrence_annotations_do_not_split_resource_identical_layers():
    first = vector_op(
        op_id="layer0.norm", deps=("input0",), stage=0, layer=0, requests=("r0",),
        elements=256, ops_per_element=6, dtype="bf16", family="rmsnorm",
        phase="prefill", recipe="rmsnorm", shape=(8, 32),
        metadata=(("norm_has_weight", True), ("structure_block", "attention"),
                  ("structure_subgraph_memberships", ("layer0.prolog",))),
    )
    repeated = replace(
        first, op_id="layer1.norm", deps=("input1",), layer=1, requests=("r1",),
        metadata=tuple({**dict(first.metadata),
                        "structure_subgraph_memberships": ("layer1.prolog",)}.items()),
    )
    one, two = node_physical(first), node_physical(repeated)
    assert one == two
    assert one["physical_work"]["launches"] == 1
    assert one["physical_work"]["vector_ops"] == 256 * 6
    changed = replace(repeated, mte2_bytes=repeated.mte2_bytes + 128,
                      hbm_bytes=repeated.hbm_bytes + 128)
    assert node_physical(changed)["physical_signature"] != one["physical_signature"]
    manifest = export_op_structure(
        [first, repeated], model={"name": "declared-repeated-norm"},
        execution={"phase": "prefill"},
    )
    assert len({node["physical_signature"] for node in manifest["nodes"]}) == 1
    assert len({node["node_id"] for node in manifest["nodes"]}) == 2


def test_capture_segment_identity_does_not_change_work_but_capture_mode_does():
    first = vector_op(
        op_id="layer0.norm", deps=("input",), stage=0, layer=0, requests=(),
        elements=256, ops_per_element=6, dtype="bf16", family="rmsnorm",
        phase="prefill", recipe="rmsnorm", shape=(8, 32),
        metadata=(("graph_captured", True), ("graph_segment", 0)),
    )
    repeated = replace(first, op_id="layer1.norm", layer=1,
                       metadata=tuple({**dict(first.metadata), "graph_segment": 1}.items()))
    assert node_physical(first) == node_physical(repeated)
    eager = replace(repeated, metadata=tuple({
        **dict(repeated.metadata), "graph_captured": False}.items()))
    assert node_physical(eager)["physical_signature"] != node_physical(first)["physical_signature"]


def test_routing_producer_occurrences_preserve_aliases_without_splitting_layers():
    import json
    from pathlib import Path

    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case

    root = Path(__file__).resolve().parents[1]
    case = parse_case(json.loads((root / "examples/cases/qwen35-a3-tp4.json").read_text()))
    graphs = target_graphs(case, phase="prefill", tokens=8192, scope="body")
    for recipe in ("route_pack", "route_unpack"):
        ops = [op for op in graphs[0] if op.recipe == recipe]
        assert len(ops) == 40
        assert len({node_physical(op)["physical_signature"] for op in ops}) == 1
        original = ops[0]
        meta = dict(original.metadata)
        if recipe == "route_pack":
            meta["topk_ids_producer"] = meta["payload_producer"]
        else:
            meta["topk_weights_producer"] = None
        changed = replace(original, metadata=tuple(meta.items()))
        assert node_physical(changed)["physical_signature"] != node_physical(original)["physical_signature"]


def test_semantic_recipe_labels_share_fallback_but_explicit_invocations_do_not():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from test_prediction_system import case

    first = vector_op(
        op_id="norm_a", deps=(), stage=0, layer=0, requests=(),
        elements=256, ops_per_element=6, dtype="bf16", family="rmsnorm",
        phase="prefill", recipe="attention_norm", shape=(8, 32),
    )
    second = replace(first, op_id="norm_b", layer=1, recipe="ffn_norm")
    assert node_physical(first) == node_physical(second)
    baseline = operator_costs(case(), [first, second])["operators"]
    assert baseline[0]["duration_ms"] == baseline[1]["duration_ms"]
    declared = case()
    declared["runtime"]["kernels"] = {"invocations": [{
        "match": {"recipe": "attention_norm"}, "startup_ms": 1,
    }]}
    bindings = declared["runtime"]["kernels"]
    assert node_physical(first, recipe_bindings=bindings) != node_physical(
        second, recipe_bindings=bindings)
    changed = operator_costs(declared, [first, second])["operators"]
    assert changed[0]["duration_ms"] == changed[1]["duration_ms"] + 1


def test_family_service_identity_still_distinguishes_kernel_recipe_names():
    from mindie_sim.prediction.services import service_identity
    from mindie_sim.prediction.structure import parse_case
    from test_mechanism_contract import hardware_case
    from test_resource_services import copy_op, curve

    case = parse_case(hardware_case())
    first = copy_op()
    second = replace(first, recipe="other_copy")
    bindings = {"families": {"copy": {"kind": "resource_service", "model": curve().to_dict()}}}
    assert service_identity(case, first) != service_identity(case, second)
    assert node_physical(first, recipe_bindings=bindings) != node_physical(
        second, recipe_bindings=bindings)


def test_catalog_matrix_roles_share_only_an_unbound_selected_program():
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.program import RecipeBook
    from test_prediction_system import case

    first = next(op for op in target_graphs(
        case(), phase="prefill", tokens=4, scope="body")[0]
        if dict(op.metadata).get("program_kind") == "matmul")
    second = replace(
        first, op_id="other_projection", family="gate_up_projection", recipe="dense_gate_up",
        metadata=tuple({**dict(first.metadata), "semantic_family": "gate_up_projection",
                        "quant_category": "shared_expert", "catalog_category": "shared_expert"}.items()))
    assert node_physical(first) == node_physical(second)
    explicit = {"families": {"gate_up_projection": {"kind": "envelope"}}}
    book = RecipeBook.from_mapping(explicit)
    assert type(book.select(first)) is not type(book.select(second))
    assert node_physical(first, recipe_bindings=explicit) != node_physical(
        second, recipe_bindings=explicit)
    invocation = {"invocations": [{
        "match": {"family": "gate_up_projection"}, "startup_ms": 1,
    }]}
    assert node_physical(first, recipe_bindings=invocation) != node_physical(
        second, recipe_bindings=invocation)


def test_glm_msd_member_ids_are_local_but_recipe_shape_and_order_remain_physical():
    from copy import deepcopy

    from mindie_sim.prediction.operator_analysis import target_graphs
    from test_subgraph_prediction_coverage import _glm_block

    case = _glm_block(msd=True, layers=2)
    ops = [op for op in target_graphs(case, phase="decode", tokens=1, scope="body")[0]
           if "fusion_semantic_recipes" in dict(op.metadata)]
    assert len(ops) == 2
    first, second = ops
    original = node_physical(first, recipe_bindings=case.runtime.kernels)
    assert node_physical(second, recipe_bindings=case.runtime.kernels) == original
    first_entries = dict(first.metadata)["fusion_semantic_recipes"]
    second_entries = dict(second.metadata)["fusion_semantic_recipes"]
    assert [entry["op_id"] for entry in first_entries] != [
        entry["op_id"] for entry in second_entries]
    for changed_field, value in (
        ("recipe", "distinct_explicit_member_recipe"),
        ("local_shape", [1, 2, 3]),
        ("family", "distinct_physical_member_family"),
    ):
        metadata = deepcopy(dict(second.metadata))
        metadata["fusion_semantic_recipes"][0][changed_field] = value
        changed = replace(second, metadata=tuple(metadata.items()))
        assert node_physical(changed, recipe_bindings=case.runtime.kernels) != original
    reordered = deepcopy(dict(second.metadata))
    reordered["fusion_semantic_recipes"].reverse()
    assert node_physical(replace(second, metadata=tuple(reordered.items())),
                         recipe_bindings=case.runtime.kernels) != original
    # Unknown nested IDs are deliberately not part of this narrow normalization.
    left = replace(first, metadata=(*first.metadata, ("custom_source", {"op_id": "left"})))
    right = replace(second, metadata=(*second.metadata, ("custom_source", {"op_id": "right"})))
    assert node_physical(left, recipe_bindings=case.runtime.kernels) != node_physical(
        right, recipe_bindings=case.runtime.kernels)


def test_mx_dispatch_producer_ids_preserve_scale_payload_aliasing():
    import json
    from pathlib import Path
    from mindie_sim.prediction.operator_analysis import target_graphs
    root = Path(__file__).resolve().parents[1]
    path = next((root/'examples/cases/high-concurrency-20261030/v41-composed-arch35-conditional').glob('*c32*'))
    raw = json.loads(path.read_text())
    ops = target_graphs(raw, phase='decode', tokens=1, context=2048,
                        sequences=[4]*8, scope='body')[0]
    up = [op for op in ops if op.op_id.endswith('.expert_up')]
    assert len(up) == 40
    assert len({node_physical(op)['physical_signature'] for op in up}) == 1
    original = up[0]
    meta = dict(original.metadata)
    assert meta['activation_scale_producer'] == meta['activation_payload_producer']
    meta['activation_scale_producer'] = 'independent_scale_kernel'
    other = replace(original, metadata=tuple(meta.items()))
    assert node_physical(other)['physical_signature'] != node_physical(original)['physical_signature']
