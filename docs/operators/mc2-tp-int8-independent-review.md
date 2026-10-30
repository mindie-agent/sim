# Independent MC2 TP caller and INT8 producer review

Reviewed caller `e01bb79d`, INT8 selector `89d90bb1`, and receiver-output correction `ec4c5352`. No blocking finding remains in the tested domain. This is a source/work/dependency review, not measured timing acceptance.

The pinned vLLM caller prepares TP token views before TopK and gathers the routed result afterwards. Router and shared-expert work retain full rows. The wrapper preserves the global integer expert ledger; it changes source ownership, not total expert computation. Arbitrary captured token identities, graph masks and heterogeneous DP/SP preparation are outside this explicit DP1/EP=TP/ETP1 branch.

The independent producer runs 24 real Engine controls across TP2/4/8, q1/q4, two batch sizes, optional shared experts and view versus explicit per-rank gather copies. It checks exact framed wire totals, full/local row contracts, one INT8 source conversion, retained down conversion, and dependency propagation from logits delayed to 19ms. Every graph completes with zero live tasks/programs. Two independently committed modules were composed only in the review process; no production tree was edited.

The actual Qwen3.5 conditional inputs exposed a missing *typed output declaration*: native dispatch already wrote payload/scales, but its completion node did not declare the received FP32 scale tensor required by the INT32 grouped caller. `ec4c5352` declares it at native kernel exit, preserving the consumer guard. Independent one-layer controls use the real c4/c64/c128 cases with q4:

| Contract | c4 | c64 | c128 |
|---|---:|---:|---:|
| Full router rows | 16 | 256 | 512 |
| Local TopK/MC2 source rows | 4 | 64 | 128 |
| Received expert rows | 32 | 512 | 1024 |
| Received FP32 row scales, bytes | 128 | 2048 | 4096 |
| INT32 grouped output, bytes | 131072 | 2097152 | 4194304 |

The grouped matrix retains INT32 output and has no duplicate matrix epilogue. DequantSwigluQuant waits for both matrix output and dispatch scales; its INT8 payload and new scales feed the down matrix. The removed generic down quantizer is replaced by that explicit activation producer, not discarded. Source bytes, metadata and actual late-input dependencies were checked. The reports' duration fields include an artificial readiness wait and must not be used as predicted-versus-observed latency.

Evidence and reproducible producers:

- `data/verification/mc2-tp-int8-independent-review-20261030.json` and `.producer.py`.
- `data/verification/mc2-int32-scale-independent-review-20261030.json` and `.producer.py`.

Each producer accepts `--root CHECKOUT --output REPORT`; the first optionally accepts `--int8-root` to reproduce the isolated module composition. The default root is the containing repository once both changes are integrated. Final focused checks passed 24 tests; earlier commands passed 7 wrapper and 32 INT8/MX/native tests, with overlaps, so these counts are not additive. Scalar reciprocal timing, extra polling, actual native tiling and binary identity remain separate evidence limitations; no observed duration was used as work or a floor.
