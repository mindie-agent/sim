"""DynamicMxQuant producer contract and materialization.

[Role]
  Check the source group-32 E8M0 producer selected for an ordinary MXFP8
  projection, not a generic QuantV2 or a fused SwiGLU quantizer.

[Invariants]
  * One BF16 row of 16384 values materializes 16384 E4M3 payload bytes plus
    512 E8M0 scale bytes; 1024 rows pay 17,301,504 bytes and 524,288 scale
    bytes.
  * The E8M0 scale is a separate tensor with shape (K/64, 2).
  * The virtual-instruction work is the source max-exponent / scale / clamp-cast
    ledger, not a generic per-token integer quant.
"""
import pytest

from mindie_sim.prediction.mx_dynamic_quant import group32_work, producer_contract
from types import SimpleNamespace

from mindie_sim.prediction.piecewise.families_matrix import _require_mx_scale_producer
from mindie_sim.prediction.mx_source_provenance import source_revision_from_oci
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _guard_low_act
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import StaticSignature


@pytest.mark.parametrize("width", (32, 64, 96, 128, 256, 100, 16384))
def test_dynamic_mx_quant_widths_and_odd_scale_padding(width):
    c = producer_contract(width)
    blocks = (width + 31) // 32
    padded = (blocks + 1) // 2 * 2
    assert c["payload_bytes"] == width
    assert c["scale_bytes"] == padded
    assert c["scale_shape"] == (padded // 2, 2)
    assert c["write_bytes"] == width + padded


def test_oci_source_is_not_global_r64_and_955_keeps_own_tree():
    assert source_revision_from_oci("", "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e") == "r64"
    assert source_revision_from_oci("r31:9e869e81484", "") == "unpinned"
    assert source_revision_from_oci("r35:5461a1a8253", "") == "unpinned"
    assert source_revision_from_oci("", "") == "unpinned"
    assert producer_contract(16384, source_revision="955f")["source"]["sha256"] == "31cf4eb673be3cbe38601a90b210e88cc2ed48692c2d84ce8e5cc8bf5c19ad03"
    assert producer_contract(16384)["source"]["kernel"] == "DynamicMxQuantSourceUnpinned"


def test_dynamic_mx_quant_fp16_input_and_odd_group_key():
    c = producer_contract(96, input_dtype="fp16")
    assert c["input_format"] == "fp16"
    assert c["input_bytes"] == 192
    assert c["scale_bytes"] == 4
    assert c["scale_shape"] == (2, 2)
    assert c["tiling_key"] == 33
    assert producer_contract(16384, scale_alg=1)["tiling_key"] == 65
    assert producer_contract(16384)["source_revision"] == "unpinned"
    assert producer_contract(16384, source_revision="r64")["source_revision"] == "r64"
    assert producer_contract(16384, source_revision="955f")["source_revision"] == "955f"
    assert producer_contract(16384, source_revision="missing")["source_revision"] == "unpinned"
    assert producer_contract(16384, scale_alg=1)["work_known"] is False
    with pytest.raises(ModelingError, match="scale_alg_2_rejected_for_fp8"):
        producer_contract(16384, scale_alg=2)


def test_dynamic_mx_quant_glm_materialization():
    c = producer_contract(16384)
    assert c["input_bytes"] == 2 * 16384
    assert c["payload_bytes"] == 16384
    assert c["scale_bytes"] == 512
    assert c["scale_shape"] == (256, 2)
    assert c["write_bytes"] == 16896
    assert c["write_bytes"] * 1024 == 17_301_504
    assert c["scale_bytes"] * 1024 == 524_288
    assert c["quant_op"] == "DynamicMxQuant"


def test_dynamic_mx_quant_source_work_is_typed():
    w = group32_work(16384)
    assert w["groups"] == 512
    assert w["full_loops"] == 64
    assert w["register_calls"] == {"max_exp": 384, "scale": 60, "data": 1024,
                                   "total": 1468}
    assert w["lane_ops"]["total"] == 204_800
    assert w["issue_cycles_known"] is False
    fp16 = group32_work(16384, "fp16")
    assert fp16["register_calls"]["max_exp"] == 896
    assert fp16["lane_ops"]["total"] == 282_624
    # Same padded 8 groups / one VF: K255 and K256 must be close, not 2x.
    assert group32_work(255)["lane_ops"]["total"] == 3193
    assert group32_work(256)["lane_ops"]["total"] == 3200


class _Producer:
    def __init__(self, name="o_projection_QuantV2", scale=512, kind="DynamicMxQuant"):
        self.name = name
        self.family = "quantize"
        self.theta = StaticSignature(
            role="quantize", quant=(("native_kind", kind),
                                    ("activation_scale_format", "fp8_e8m0"),
                                    ("activation_scale_bytes", scale)))


class _Consumer:
    name = "o_projection"


class _FakeFeatures:
    activation_format = "fp8_e4m3"
    activation_dtype = "fp16"
    quant_activation_format = ""
    pp = dp = cp = pcp = 1
    world_size = 1
    token_tile = 16
    kv_tile = 16
    cores = 2
    moe_row_tile = 4
    split_target = 1000
    max_splits = 4
    graph_buckets = ()
    metadata = {}


class _FakeQuant:
    activation_format = "fp8_e4m3"

    def for_category(self, category):
        return self

    def format_for(self, category):
        return "fp8_e4m3"

    def linear_method(self, category):
        return "mx_fp8"


def test_guard_emits_fp16_odd_group_producer_for_real_shape():
    ctx = SimpleNamespace(quant=_FakeQuant(), features=_FakeFeatures(),
                          tp=1, ep=1, etp=1)
    specs, signatures = [], {}
    _guard_low_act(specs, signatures, ctx, "o_projection", elements=96,
                   category="attn.o")
    assert specs[0][0] == "o_projection_QuantV2"
    theta = signatures["o_projection_QuantV2"]
    assert theta.find("input_format") == "fp16"
    assert theta.find("tiling_key") == 33
    assert theta.find("activation_scale_bytes") == 4
    assert theta.find("activation_scale_shape") == (2, 2)
    assert theta.find("read_bytes") == 192
    assert theta.find("write_bytes") == 100
    assert theta.find("mx_vector_ops_per_row") == 2783


def test_feature_runtime_image_selects_source_without_model_label():
    from types import SimpleNamespace
    r64 = FeatureConfig.from_runtime(SimpleNamespace(
        metadata=(("image", "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),)), None)
    assert dict(r64.metadata)["mx_quant_source_revision"] == "r64"
    r35 = FeatureConfig.from_runtime(SimpleNamespace(
        metadata=(("image", "sha256:5461a1a8253"),)), None)
    assert dict(r35.metadata)["mx_quant_source_revision"] == "unpinned"


def test_mx_consumer_requires_matching_separate_scale_producer():
    _require_mx_scale_producer(_Consumer(), (_Producer(),), 512)
    _require_mx_scale_producer(_Consumer(), (_Producer(name="custom_quant"),),
                               512, producer_name="custom_quant")
    with pytest.raises(ModelingError, match="requires_dynamic_mx_quant_scale_producer"):
        _require_mx_scale_producer(_Consumer(), (), 512)
    with pytest.raises(ModelingError, match="activation_scale_producer_mismatch"):
        _require_mx_scale_producer(_Consumer(), (_Producer(scale=256),), 512)
