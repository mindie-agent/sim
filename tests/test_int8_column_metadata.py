"""Explicit scale-only INT8 inputs agree across residency and matrix service."""
from dataclasses import replace
import json

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.matrix_epilogue import MatrixEpilogue
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.schema.common import ModelingError


def matrix(quant, *, category='dense_mlp', rows=5, columns=7, width=11, groups=1):
    return gemm_op(op_id='projection', deps=('input',), stage=0, layer=0,
                   requests=(), m=rows, n=columns, k=width, weight_count=groups,
                   quant=quant, category=category, phase='prefill')


def test_bf16_scale_only_is_one_column_input_and_fp32_row_scales_stay_separate():
    quant = QuantSpec('declared-native', weight_format='int8', scale_format='bf16',
                      scale_bytes=2, int8_weight_metadata='scale_only')
    op = matrix(quant)
    meta = dict(op.metadata)
    epilogue = MatrixEpilogue.from_op(op)
    assert quant.matrix_payload_bytes(7, 11, 'dense_mlp') == 77
    assert quant.matrix_bytes(7, 11, 'dense_mlp') == 77 + 14
    assert epilogue.column_scale_bytes == 14
    assert epilogue.row_scale_bytes == 5 * 4
    assert epilogue.operands(op) == (5 * 11, 7 * 11)
    assert op.mte2_bytes == 5 * 11 + 7 * 11 + 7 * 2 + 5 * 4
    assert op.mte3_bytes == 5 * 7 * 2
    assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes
    assert op.flops == 2 * 5 * 7 * 11
    assert op.vector_ops == 4 * 5 * 7
    assert meta['weight_scale_format'] == 'bf16'
    producer, consumer = activation_quantization((op,), 'bf16')
    assert producer.mte3_bytes == 5 * 11 + 5 * 4
    assert consumer.deps == (producer.op_id,)
    # One complete output tile consumes all row/column scales once after K.
    assert epilogue.scales(5, 7, 5, 7, 1, reuse_inputs=True, reuse_weights=True) == (34, 34)


def test_explicit_scale_only_does_not_change_other_quantization_categories():
    legacy = QuantSpec('int8', weight_format='int8')
    selected = replace(legacy, category_overrides={
        'dense_mlp': {'scale_format': 'bf16', 'scale_bytes': 2,
                      'int8_weight_metadata': 'scale_only'}})
    restored = QuantSpec(**json.loads(json.dumps(selected.to_dict())))
    for quant in (selected, restored):
        assert matrix(quant, category='mla.q_b') == matrix(legacy, category='mla.q_b')
        assert matrix(quant).mte2_bytes == 166
        assert quant.matrix_bytes(7, 11, 'dense_mlp') == 91
        assert quant.matrix_bytes(7, 11, 'mla.q_b') == 77 + 7 * 8
    assert MatrixEpilogue.from_op(matrix(legacy)).column_scale_bytes == 7 * 8


@pytest.mark.parametrize('groups', [1, 3])
def test_grouped_scale_inputs_and_partial_tiles_do_not_duplicate_column_metadata(groups):
    quant = QuantSpec('native', weight_format='int8', scale_bytes=2,
                      scale_format='bf16', int8_weight_metadata='scale_only')
    op = matrix(quant, rows=5 * groups, groups=groups)
    epilogue = MatrixEpilogue.from_op(op)
    assert epilogue.column_scale_bytes == groups * 7 * 2
    assert epilogue.operands(op) == (groups * 5 * 11, groups * 7 * 11)
    # Two N tails cover one group's columns; different output groups have
    # independent weight scales, while each row scale is reused across N.
    tiles = [(5, 3), (5, 4)] * groups
    traffic = sum(epilogue.scales(a, b, 5 * groups, 7, groups,
                                 reuse_inputs=True, reuse_weights=True)[1]
                  for a, b in tiles)
    assert traffic == pytest.approx(epilogue.row_scale_bytes + epilogue.column_scale_bytes)


def test_unknown_column_metadata_cannot_silently_choose_scale_offset():
    with pytest.raises(ModelingError, match='int8_weight_metadata'):
        QuantSpec('invalid', int8_weight_metadata='infer_from_name')
