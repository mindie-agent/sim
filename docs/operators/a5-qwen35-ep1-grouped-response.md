# A5 EP1 grouped native response

This conditional packet describes the old TP4/EP1 Qwen3.5 capture domain:
256 declared expert banks, INT8 activations and FRACTAL_NZ INT8 weights,
BF16 output and column scales, FP32 token scales, 32 AIC / 64 AIV blocks.
The two projections are `(N,K)=(256,2048)` and `(2048,128)`. The independent
EP4 rules require 64 banks and different shapes/output types; they are not
reused. The old short-context probe also uses FP32 column scales and complete
eager/graph events, so it supplies no matching frozen native-kernel response.

Only `p65536_c1_hold` and `p65536_c4_hold` assign parameters, using original
single-kernel means from complete q4 Body windows. Their total routed rows are
32 and 128. Original observation roles remain `unassigned`; this is a
retrospective source split, with previously viewed 126K captures held fixed for
validation. Neither Body duration nor internal gap enters fitting.

For `x = routed_rows / 256`, the continuous responses in microseconds are:

| Projection | Response | Domain |
| --- | --- | --- |
| gate/up | `27.1486914928 + 16.3555803028*x` | `0.125 ≤ x ≤ 0.5` |
| down | `26.9353653827 + 19.8162534688*x` | `0.125 ≤ x ≤ 0.5` |

The divisor denotes declared bank count, not a fabricated balanced histogram.
Actual group-list values and the loaded binary revision remain unknown. The
intercept is a joint response coefficient, not a separately measured startup.
No opaque duration identifies HBM/MTE/Cube bandwidth or per-pipe occupancy.

An explicit `native_grouped_call` resource serves the inclusive kernel response
through the common Engine. Original analytical work remains visible as a
reference. Fixed hardware rates, compute peaks, service modes and owner counts
are guarded; changed HBM, MTE2, Vector or INT8 supply selects the analytical
program again. Routed rows outside 32–128 also retain the analytical program.
The separate case preserves EP1 and declares BF16/2-byte routed-expert column
scales from the native ABI. Other quantization categories remain unchanged.
Column-scale dtype is retained in this case/protocol; the current generic
regime matcher does not expose it as a separate match axis, so this packet is
not an automatic cross-caller precision policy.

The four 126K sources contain eight primitive comparisons; errors range from
−3.073% to +1.967%. Extra 64K non-assignment data include a real B2/M64 Forward
inside the file named `c4_o32`; its two errors are −11.398% and −10.503% and were
not used to refit. Every source's original ABI, hash, extraction identity and
CSV locators remain in the packet. Means pool correlated layer calls, not
independent hardware trials. These primitive results do not establish complete
Body or serving accuracy.

Files live in `data/observations/a5-qwen35-ep1-grouped-20261030/`. `native.py`
streams the eight original compressed captures; `fit.py` creates parameters,
the explicit `examples/cases/longctx-decode-20261030/qwen35-a5-tp4-ep1-grouped-joint.json`
case and actual public-engine primitive validation. `reuse-audit.json` records
why the existing short-context/EP4 observations were not imported.
