"""Aligned BF16 MX32 sender work from the shared arch35 quantization helper.

[Role] Supply the ordinary MC2 producer's explicit vector work ledger.
[Invariants] Count each routed row, not each original token. Source whole-vector
  operations are counted as element work, never inferred latency. Register
  loads/stores and scalar control have no invented service rate.
[Timeouts & Locks] Constant-time construction, no locks or source I/O.
[Side-Effect Policy] Pure work descriptions only.
"""
from mindie_sim.schema.common import ModelingError


SOURCE = {
    "repository": "cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "path": "mc2/common/op_kernel/quantize_functions.h",
    "sha256": "cecceda01dc86aeeb28728316c41749fcaa5c7a9b3a56826fdea2ab8dd0acc78",
    "symbols": ("ComputeMaxExp", "ComputeScale", "ComputeFp8Data"),
    "lines": ((80, 154), (198, 269), (362, 455)),
}


def mx32_work(width, input_dtype="bf16", payload_dtype="fp8_e4m3"):
    """Non-Clip BF16→E4M3, full 256-value iterations with group-32 E8M0.

    ComputeMaxExp ANDs two 128-lane registers, pairwise-maxes them, then
    reduces each 16-lane block (128 - 8 comparisons). ComputeScale has seven
    full-register constants and 13 masked element operations per scale.
    ComputeFp8Data performs one multiply, two rearrangements and two casts
    per input element. This is typed work, not exact instruction issue time.
    """
    if (input_dtype != "bf16" or payload_dtype != "fp8_e4m3"
            or width < 256 or width % 256):
        raise ModelingError("mc2_mx32_requires_bf16_e4m3_and_width_multiple_256")
    iterations, scales = width // 256, width // 32
    return {
        # The FP16-only invalid exponent constant has no BF16 consumer.
        "max_exponent": 128 + iterations * (256 + 128 + 120),
        "scale_construction": 7 * 128 + 13 * scales,
        "rescale_cast": 5 * width,
    }
