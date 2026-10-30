# Expected expert-bank scheduling

Probability routing supplies expected routed rows and expected active experts.
The old aggregate matrix used the total routed rows as one M axis while its B
operand contained every active bank. Each M tile then copied the entire bank
set through the owner read pipe. This was incorrect even when ideal HBM reuse
kept off-chip bytes unchanged.

The generic matrix adapter now assigns `ceil(expected_active_experts)` bank
slots and `ceil(expected_rows / expected_active_experts)` rows per slot. Each
slot receives only its share of the weight banks. The scheduling grid is a
conditional-mean approximation: it is neither a sampled histogram nor a claim
of balanced routing. Its rounded dimensions size buffers and owner tiles;
original expected FLOPs, HBM payloads, row/column scales and epilogue work stay
unchanged. Fractional expected banks do not become extra real weight reads.
Private copy service, compute-width MTE1 lower bounds and owner allocation
follow the rounded grid; they are not unchanged aggregate services. Predictions
can therefore have scheduling steps at a grid boundary.

Exact integer routing continues through its existing per-expert row classes.
Explicit source planners retain precedence. Ordinary grouped matrices use the
compact steady pipeline unless a family/program/recipe is explicitly selected;
explicit tiled and hierarchical programs also receive the corrected bank grid.
No hardware rate, measured duration or model label selects this mechanism.

For the retained Qwen3.5-397B TP16 probability-routing input, layer-zero gate/up
and down controls give:

| Routed rows | Old aggregate gate/up + down ms | Bank grid + steady ms |
| ---: | ---: | ---: |
| 2560 | 7.604440 | 0.471319 |
| 6400 | 21.661383 | 0.539608 |

These are predicted operator controls, not measured body accuracy. The bank
partition correction is separately tested on the same fixed TiledMatmul to
isolate it from steady-pipeline selection: when N and each bank's row count fit
one tile, MTE2 reads each bank once instead of20 or50 aggregate-M repetitions.
Integer equal-bank controls agree with an independently constructed grouped
program. Tail shapes, expected fractional occupancy, output-scale conservation,
input readiness and bandwidth response run on the actual shared Engine.
