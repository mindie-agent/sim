"""R31 GDN external q/k L2-normalization source adapter.

The source facts are retained, not fitted:

* ``vllm_ascend/ops/gdn.py`` sha256 ``5e3b0fb0...`` lines 34 and 739/740
  import and call ``vllm.third_party.flash_linear_attention.ops.l2norm``.
* The retained verified vllm ``l2norm.py`` sha256 ``520f4513...`` shows the
  public ``l2norm_fwd(x, eps=1e-6, output_dtype=None)`` wrapper and its
  default ``l2norm_fwd_kernel2`` launch ``(ceil(M/32),)`` with ``MBLOCK=32``.
* Pure prefill keeps q/k normalization inside the signed
  ``aclnnChunkGatedDeltaRuleFwd`` parent; only the non-spec/mixed-decode
  external calls are admitted here.
"""
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import native_l2norm_fwd_kernel2 as l2n
from mindie_sim.prediction.piecewise.compiler import (
    CompiledOperator, ImplementationPolicy, PipeSupply)
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.subgraphs.builders_attention import _build_a3
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.schema.subgraphs import BoundOperator

_IMAGE_ID = ("sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
_META = (("image_id", _IMAGE_ID), ("source_lowerings", ("gdn_decode",)))
_PIPES = ("vector", "mte2", "mte3", "hbm", "mte1", "mte2_aiv", "mte2_aic",
          "fixpipe", "cube", "link")
_SUPPLY = PipeSupply(tuple((pipe, 1e9) for pipe in _PIPES),
                     resource_units=tuple((pipe, 64) for pipe in _PIPES))
_POLICY = ImplementationPolicy(mt=128, nt=16, cores=32, moe_row_tile=4,
                               split_target=1000, max_splits=4)


def _built(metadata=_META):
    model = load_model("qwen3.6-27b")
    layer = next(plan for plan in model.layer_plans if plan.linear_heads)
    features = FeatureConfig.from_model(model, tp=2, dp=4, ep=1,
                                        metadata=metadata)
    built = _build_a3(_Ctx(model, layer, features, stage=0,
                           layers=len(model.layer_plans)))
    operators = tuple(BoundOperator(template, built.signature(template.name))
                      for template in built.template.operators)
    compiled = {}
    for operator in operators:
        compiled[operator.name] = CompiledOperator(
            operator, _SUPPLY, _POLICY, kind="A3", siblings=operators,
            variant="recurrent_decode", feature_metadata=features.metadata)
    return compiled


def _binding(name, metadata=_META, phase="decode"):
    compiled = _built(metadata)[name]
    spec = l2n.binding_spec(compiled, phase=phase)
    if spec is None:
        return compiled, None, None
    declared = declare_native_call(
        _surface(compiled, spec, source_boundary="kernel"),
        api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return compiled, spec, bind_native_call(declared)


def _decode(count):
    return WorkloadGeometry(
        tuple(RequestGeometry(1, 4096, phase="decode")
              for _ in range(count)), phase="decode")


def test_r31_q_and_k_share_the_real_wrapper_and_kernel_identity():
    bindings = {}
    for name, axis in (("q_l2norm", "q"), ("k_l2norm", "k")):
        _compiled, spec, binding = _binding(name)
        assert spec is not None and binding is not None
        assert spec.api["namespace"] == "triton"
        assert spec.api["entry"] == l2n.ENTRY
        assert tuple(kernel["name"] for kernel in spec.kernels) == (
            "l2norm_fwd_kernel2",)
        assert spec.kernels[0]["source_sha256"] == l2n.L2NORM_SOURCE["sha256"]
        assert spec.source["path"] == l2n.L2NORM_SOURCE["path"]
        assert spec.source["sha256"] == l2n.L2NORM_SOURCE["sha256"]
        assert binding.source["revision"] is None
        assert spec.owner == name
        assert "qk_axis" not in spec.typed["attrs"]
        assert spec.typed["attrs"]["non_spec_call_lines"]["mixed"] == [739, 740]
        assert spec.typed["attrs"]["non_spec_call_lines"]["pure"] == [864, 865]
        assert spec.typed["attrs"]["caller_lines"] == l2n.GDN_LINES
        assert spec.typed["attrs"]["caller_sha256"] == l2n.GDN_SOURCE["sha256"]
        assert spec.typed["attrs"]["deployment_image_id"] == l2n.R31_IMAGE_SHA
        assert spec.typed["shapes"] == {"x": [8, 128], "y": [8, 128]}
        assert spec.typed["dtype"] == "bf16"
        assert "deployed_vllm_commit_not_extracted" in binding.unresolved
        bindings[name] = binding
    # One wrapper/kernel interface, two real q/k launches: the shared static
    # identity has the same call_id while each occurrence owns its own charge.
    assert bindings["q_l2norm"].call_id == bindings["k_l2norm"].call_id
    assert bindings["q_l2norm"].timing_owner["op_id"] != (
        bindings["k_l2norm"].timing_owner["op_id"])
    # The shared identity must not make the k occurrence inherit q's cached
    # prepare branch/line.
    for name, line in (("q_l2norm", 864), ("k_l2norm", 865)):
        compiled = _built()[name]
        envelope = l2n.dynamic_call(bindings[name], compiled, _decode(1),
                                    validate=validate_native_occurrence)
        assert envelope["prepare"]["call_line"] == line
        assert envelope["prepare"]["match"]["operator"] == name
        assert envelope["prepare"]["shapes"]["x"] == [1, 1, 8, 128]


def test_one_decode_token_is_real_head_by_dimension_geometry():
    compiled, _spec, binding = _binding("q_l2norm")
    envelope = l2n.dynamic_call(binding, compiled, _decode(1),
                                validate=validate_native_occurrence)
    assert envelope is not None
    prepare = envelope["prepare"]
    assert prepare["shapes"] == {"x": [1, 1, 8, 128], "y": [1, 1, 8, 128]}
    assert prepare["kernel_arguments"] == {
        "X": "x", "Y": "y", "eps": 1e-6, "M": 8, "N": 128,
        "BD": 128, "MBLOCK": 32,
    }
    assert prepare["launch"]["grid"] == [1]
    assert prepare["descriptors"]["x"]["shape"] == [1, 1, 8, 128]
    assert prepare["descriptors"]["y"]["allocation"] == (
        "callee_allocated_by_wrapper")
    assert prepare["source_branch"] == "pure_non_spec_decode"
    assert prepare["call_line"] == l2n.GDN_LINES["pure_q_call"]
    assert prepare["stage"] == "interface_prepared"
    assert prepare["runtime_executed"] is False
    assert prepare["complete"] is False
    assert "deployed_vllm_commit_not_extracted" in prepare["partial_reasons"]
    assert envelope["prepare_args"]["shapes"]["x"] == [1, 1, 8, 128]


def test_mblock_grid_follows_the_source_ceil_m_over_32_rule():
    compiled, _spec, binding = _binding("k_l2norm")
    four = l2n.dynamic_call(binding, compiled, _decode(4),
                            validate=validate_native_occurrence)
    five = l2n.dynamic_call(binding, compiled, _decode(5),
                            validate=validate_native_occurrence)
    assert four["prepare"]["kernel_arguments"]["M"] == 32
    assert four["prepare"]["launch"]["grid"] == [1]
    assert five["prepare"]["kernel_arguments"]["M"] == 40
    assert five["prepare"]["launch"]["grid"] == [2]
    assert five["prepare"]["shapes"]["x"] == [1, 5, 8, 128]


def test_mixed_decode_only_counts_the_real_decode_rows():
    compiled, spec, binding = _binding("q_l2norm")
    mixed = WorkloadGeometry(
        (RequestGeometry(1, 4096, phase="decode"),
         RequestGeometry(4096, 0, phase="prefill")), phase="decode")
    assert spec is not None
    envelope = l2n.dynamic_call(binding, compiled, mixed,
                                validate=validate_native_occurrence)
    assert envelope is not None
    assert envelope["prepare"]["shapes"]["x"] == [1, 1, 8, 128]
    assert envelope["prepare"]["kernel_arguments"]["M"] == 8
    assert envelope["prepare"]["source_branch"] == "mixed_non_spec_decode"
    assert envelope["prepare"]["call_line"] == l2n.GDN_LINES["mixed_q_call"]
    assert envelope["prepare"]["deployment"]["caller_sha256"] == (
        l2n.GDN_SOURCE["sha256"])


def test_pure_prefill_is_a_member_of_the_genuine_fused_parent():
    compiled, _spec, _binding_obj = _binding("q_l2norm")
    pure = WorkloadGeometry((RequestGeometry(4096, 0, phase="prefill"),),
                            phase="prefill")
    parent = l2n.binding_spec(compiled, phase="prefill")
    assert parent is not None
    assert parent.owner == "A3:recurrent:gated_delta"
    assert {"q_l2norm", "k_l2norm"} <= set(parent.members)
    assert parent.api["aclnn_symbol"] == "aclnnChunkGatedDeltaRuleFwd"
    assert parent.typed["attrs"]["use_qk_l2norm_in_kernel"] is True
    # The common _gdn_chunk_spec currently carries the legacy c3d00a record;
    # the leaf marks that explicitly rather than shadowing it.  After the
    # requested producer/source hunk selects R31/5e3b0fb0, the marker must
    # disappear and the source hash must be the authoritative one.
    if parent.source.get("revision") != "r31/9e869e814846":
        assert "parent_source_revision_legacy_not_r31_5e3b0fb0" in parent.unresolved
    else:
        assert parent.source["sha256"] == l2n.GDN_SOURCE["sha256"]
    # The l2norm leaf does not own the parent call's dynamic prepare; the
    # shared dynamic_gdn path does.
    assert l2n.dynamic_call(_binding_obj, compiled, pure,
                            validate=validate_native_occurrence) is None


def test_no_normalize_parent_branch_is_not_shadowed(monkeypatch):
    compiled, _spec, _binding_obj = _binding("q_l2norm")
    from mindie_sim.prediction.piecewise import native_calls as nc
    source = {"repository": "vllm-ascend", "path": "vllm_ascend/ops/gdn.py",
              "sha256": "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed",
              "symbol": "AscendGatedDeltaNetAttention._forward_core",
              "revision": "r31/9e869e814846", "state": "task_artifact"}
    typed_no_norm = {
        "dtype": "bf16", "layout": "BSND", "phase": "prefill",
        "shapes": {"q": [8, 128]},
        "attrs": {"use_qk_l2norm_in_kernel": False},
        "rank_width": 3,
    }
    no_norm = nc._call_spec(
        nc._api("aclnn", "aclnnChunkGatedDeltaRuleFwd",
                aclnn_symbol="aclnnChunkGatedDeltaRuleFwd"),
        source, typed_no_norm, (nc._kernel("Cast", "copy"),),
        ("prepare", "gated_delta"), "A3:recurrent:gated_delta")
    # A different entry is also a disjoint branch, even if an attr is missing.
    other_entry = nc._call_spec(
        nc._api("torch", "torch_npu.npu_chunk_gated_delta_rule"),
        source, {**typed_no_norm, "attrs": {"use_qk_l2norm_in_kernel": True}},
        (nc._kernel("ChunkGatedDeltaRuleFwd", "mix"),),
        ("prepare", "gated_delta"), "A3:recurrent:gated_delta")
    for fake in (no_norm, other_entry):
        monkeypatch.setattr(nc, "_gdn_chunk_spec",
                            lambda compiled, phase="", _fake=fake: _fake)
        assert l2n.binding_spec(compiled, phase="prefill") is None
        assert l2n.binding_spec(compiled, phase="prefill") is None
    # The producer/source selector can stamp the disjoint branch directly;
    # the leaf must return None without even building the parent spec.
    external = _built((("image_id", _IMAGE_ID),
                       ("source_lowerings", ("gdn_decode",)),
                       ("gdn_prefill_qk_l2norm_owner", "external")))
    assert l2n.binding_spec(external["q_l2norm"], phase="prefill") is None


def test_non_r31_sources_and_other_operators_are_not_bound():
    r64 = (("image_id", "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),
           ("source_lowerings", ("gdn_decode",)))
    compiled = _built(r64)
    assert l2n.binding_spec(compiled["q_l2norm"], phase="decode") is None
    assert l2n.binding_spec(compiled["k_l2norm"], phase="decode") is None
    other = _built()["gdn_gating"]
    assert l2n.binding_spec(other, phase="decode") is None
    assert l2n.binding_spec(_built()["q_l2norm"], phase="draft") is None


def test_speculative_multitoken_decode_is_a_different_call_site():
    compiled, _spec, binding = _binding("q_l2norm")
    speculative = WorkloadGeometry(
        (RequestGeometry(2, 4096, phase="decode"),), phase="decode")
    assert l2n.dynamic_call(binding, compiled, speculative,
                            validate=validate_native_occurrence) is None
    verify = WorkloadGeometry(
        (RequestGeometry(1, 4096, phase="verify"),), phase="decode")
    assert l2n.dynamic_call(binding, compiled, verify,
                            validate=validate_native_occurrence) is None
