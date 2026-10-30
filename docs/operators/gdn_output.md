# Native token-first GDN output

`gdn_output.py` implements the arch35 token-first ChunkFwdO path from FLA
revision `a4a795857868729fbdbb0118c1e06c8655a7b31b`. Sources and hashes are retained
under `data/observations/sources/fla-a4a7958/` and the capture-image archive.
The selected geometry is BF16 K=V=128, chunk=64, scalar FP32 gates, exp2 and
V,K state layout. The planner supports its explicit alternate gate/layout
flags; it does not substitute the Catlass BNSD implementation.

## Program and work

Host tiling forms groups of three value heads for GQA=3 and four otherwise.
A group keeps Q/K loads once per unique key head, while every value head still
executes QK and QH. Token chunks and groups are distributed across physical
AICs, with paired AIVs for vector work. Inactive units cannot supply the work.

The source program preserves gate preparation, Q/K/H loads, L1-to-L0 operands,
QK and QH MMAD, Fixpipe publication, A-prime construction, direct UB-to-L1
A-prime transfer, AV, and final gated output. A-prime never visits global
memory. QK/QH use full 64-row tiles. AV receives the actual tail extent, then
issues 16x16x16 BF16 fractals; a one-row tail is neither one scalar multiply
nor a full 64-row AV tile.

A two-chunk window retains L0 bank reuse and cross-head publication edges.
Each active AIC reserves L1=352 KiB, L0A=L0B=64 KiB and L0C=256 KiB; each
paired AIV reserves UB=248 KiB. AIC and AIV read resources are distinct when
declared, while HBM remains global.

Output address is `(token * heads + head) * 128 + channel`. The native output
is already token-first, so the frontend does not add an output transpose.
Requested normalized Q/K and cumsum exports belong to Prepare and retain
separate tensor output identities. Mixed batches concatenate decode and
prefill outputs before the caller-owned prefix copy.

## Independent verification and limits

`tests/operators/test_gdn_output.py` uses numerical causal attention, token/head
address enumeration, task ownership, tail instruction work and stage-order
controls. Equal ABI shapes do not imply equal execution: `(1,127)` and
`(17,111)` both contain 128 tokens, two sequences and three chunks, but their
AV tail work differs. `tests/test_native_operator_comparison.py` preserves that
counterexample rather than claiming CSV shapes identify sequence values.

The retained 16K ABI replay predicts 347.008 us versus 512.657 us observed
(-32.312%). See `data/verification/gdn-native-20261028-integrated-caller.json`.
Vector instruction latency, scalar/flag work, caches and physical read service
remain unresolved. This result is not hardware accuracy or full-model acceptance.
