"""Ordinary fusion changes physical service while preserving live outputs."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from test_mechanism_contract import hardware_case


@pytest.mark.parametrize("rows,width", ((2, 8), (3, 65)))
def test_fused_public_graph_eliminates_only_private_intermediates(rows, width):
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 2e6
    parsed = parse_case(raw)
    quant = quant_preset("mxfp8")
    parsed = replace(parsed, model=replace(parsed.model, quant=quant))
    payload = rows * width * 2
    residual = vector_op(
        op_id="residual", deps=("input", "saved"), stage=0, layer=0, requests=("r",),
        elements=rows * width, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="prefill", recipe="residual_add",
        read_bytes_override=2 * payload, write_bytes_override=payload)
    normalized = norm(
        name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
        tokens=rows, width=width, model=parsed.model, mode="prefill")
    matrix = gemm_op(
        op_id="projection", deps=("norm",), stage=0, layer=0, requests=("r",),
        m=rows, n=width, k=width, quant=quant, category="dense_mlp", phase="prefill")
    consumer = replace(residual, op_id="consumer", deps=("projection", "residual"))
    reports, versions = [], []
    for passes in (("activation_quantization",),
                   ("activation_quantization", "norm_quant", "residual_norm")):
        runtime = replace(parsed.runtime, lowering_passes=passes)
        ops = lower_implementations((residual, normalized, matrix, consumer), runtime, "bf16")
        versions.append(ops)
        reports.append(graph_costs(
            replace(parsed, runtime=runtime), ops,
            external_inputs={"input": 0, "saved": 0}, isolated=False))
    before, after = (report["execution"]["serviced"] for report in reports)
    # Remove norm-output write/read and only the residual read into norm.
    # The residual is a live output, so its write and later consumer survive.
    assert before["memory@r0:s0"] - after["memory@r0:s0"] == 3 * payload
    assert before["matrix@r0:s0"] == after["matrix@r0:s0"]
    assert before["vector@r0:s0"] == after["vector@r0:s0"]
    fused = next(op for op in versions[1] if op.family == "add_rmsnorm_quant")
    assert fused.returns == ("activation_scales", "residual")
    assert versions[1][-1].deps == ("projection", fused.op_id)
    assert len(versions[0]) - len(versions[1]) == 2
    assert all(report["execution"]["live_tasks"] == 0 for report in reports)
