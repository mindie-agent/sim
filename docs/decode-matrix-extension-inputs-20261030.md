# Decode matrix input extension

Seven further inputs are in
`examples/cases/high-concurrency-20261030/matrix-extension/`.

- V4.1 A5 TP1/DP8/EP8: c4/c16/c32, prompt2048, no MTP, APC disabled.
  The per-cell summary carries the same engine declaration as the earlier
  DP8 campaign. Its output request length is512, not the earlier768. No
  sampled rank0 geometry from the earlier three captures is copied into these
  files. Uniform per-replica batch is a graph-construction scenario only.
- GLM Flash A3 TP16/DP1/EP16: ACLGraph c8/c16/c32/c64, 45 layers, W8A8 and
  INT8 KV, configured MTP5 and forced balance. Capture buckets are
  `[6,12,24,36,48,72,96,144,192,288,384]` in **tokens**, not requests. The
  largest declared q6/B64 body therefore fits the384-token capture.

The GLM metadata files are retained in the campaign deliverable directory,
although the top-level data copy contains only profiling CSVs. Their exact
paths and hashes are in each input. The actual retained profile-cell prefix
hit ratios are recorded; enabled APC is not silently promoted to100% reuse.
The launch script and report capture excerpt support the configuration, while
resolved force-EPLB startup remains unconfirmed for this A3 family.

Both families retain explicit partial coverage: native mixed-quant KV/Engram
representation for V4.1, and native INT8 sparse KV packing/scales for GLM.
No BF16-only source attention recipe is forced onto either representation.
Client TPOT remains outside the body prediction boundary. All seven inputs
parse and construct their complete40/45-layer target graphs, including all
required EP participants. This producer does not run body simulations or
repeat CSV extraction.

`data/verification/decode-matrix-extension-inputs-20261030.json` records the
per-input graph checks. The six additional Kimi p16k cases requested in the
same extension are in the separate Kimi variant receipt, preserving their
E16/BF16/93-layer contract.
