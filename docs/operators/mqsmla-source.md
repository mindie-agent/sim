# Mixed-quant sparse MLA: explicit conditional source path

The v41-mqsmla-source-control.json example binds
cannbot_mixed_quant_sparse_flash_mla through ordinary MLA construction for
Decode. It retains the full 40-layer TP1/DP8/EP8 scenario and the independent
HC profiles. Prefill keeps its analytic path. This is a software declaration,
not a claim that the capture binary is the inspected source.

The source is cannbot-dsl revision
4519e5372a8b9306f7d823688c0cb8141ee34fb1. Both the attention and separate
metadata Python files were read from that clean local checkout; their hashes
are embedded in prediction/mqsmla.py and the verification receipt.

The supported branch has BF16 Q/O [T,64,512], one KV head, 448 non-RoPE plus
64 RoPE features, contiguous unpadded Decode queries, scalar paged addressing,
no returned LSE and whole-query metadata ownership. SUPPORT_FD=False uses
contiguous quotient/remainder row partitioning across AICs. The source
SELECTIVE_FD_72 exception is explicitly rejected until its partial-output
reduction is modeled. Source domain errors remain visible.

## One cache representation across updates, reads and capacity

The model declares metadata.cache_representations:

| Region | Payload | Per-row scales | Physical bytes for D512 |
|---|---|---|---:|
| window | FP8 E4M3 | BF16, group32 | 544 |
| compressed_kv | FP4 E2M1 | BF16, group16 | 320 |

The shared catalog/cache_format.py resolver drives cache writes, analytic
attention reads and layer_cache_components. Consequently the unpadded
Memory.cache_bytes used by PD handoff also sees the new formats. CP sharding
changes rows, not the per-row scale count. Untouched cases retain QuantSpec
KV precision. Generic packed producers explicitly read floating-point input,
perform group reduction/scale/cast work and write packed payload plus scales.
They are analytic producers; the exact KvCompressEpilogV2 program is not
claimed. Legacy catalog-fact snapshots are not the capacity source of truth.

The compressed index-key projection consumes the normalized **pre-RoPE**
floating-point latent. Cache insertion separately consumes its rotated value,
then quantizes it. The two branches therefore fork at the compressor's norm,
not at the packed cache output. This boundary is independently supported by
the retained inferact-sra-tracking revision
0a5cb4e592a4a35a92f15b445cd3cc5340635f6e:
attention.py lines688–720 and1130–1146, compressor.py lines246–320.
The file SHA256s are respectively
ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e and
ed66374bb9e4c8c0201383d3fef811e061fd14f3cb0cb9db74bcb740f5a59009.
These are caller-contract evidence, not a capture-binary identity claim.
A custom packed-cache index owner without a modeled pre-RoPE producer is a
visible domain error; quantization loss and RoPE cannot be silently undone.

## Source work and bounded execution

Original and compressed lists remain separate, with their own bytes and valid
lengths, but share one online-softmax accumulator. The 128-row gather repeats
the last valid address for padded rows. QK therefore computes all 128 rows;
PV reduces only ceil(valid/16)*16. Q is read once per query, and each query
writes one final BF16 result.

Dequantized KV remains in L1 through QK and PV. Scores/PV use FIXPIPE directly
to the paired AIVs, and probabilities/dequantized KV use UB-to-L1 MTE3. Those
transfers contend on their real private pipes without invented HBM scratch.
One source-bound kernel owns the entire batch, so B8 uses eight AICs and
sixteen paired AIVs rather than eight serial kernel invocations.

The finite static reservation per AIC is L1 512 KiB, L0A/L0B 64 KiB each and
L0C 256 KiB; each paired AIV conservatively reserves 251,648 UB bytes including
Channel depths, NZ padding and scalar state. The current A5 profile declares
253,952 UB bytes. The source also mentions 216 KiB for its address arena; no
binary allocator equivalence or hardware capacity measurement is claimed.
Lower-capacity profiles fail explicitly; the program never enlarges them.

The executable phase window holds two complete tiles to preserve the source's
two P/alpha slots. The source reserves three KV slots and can prefetch more
aggressively. Query transitions conservatively wait for the previous output
before Q/sink reuse. These omitted overlaps, scalar page-lookup queue response,
vector-address prepass, FlashDecode partial reduction and compiler instruction
scheduling remain visible approximations. They are not compensated by a
multiplier or a fitted floor.

## Actual controls

The v41-mqsmla-source-20261030.producer.py verification producer constructs
complete eight-rank input graphs, then executes representative source kernels:

| Local B | Layer type | Predicted us |
|---:|---|---:|
| 1 | Original-window only | 6.511 |
| 1 | Window + 512 compressed rows | 20.017 |
| 8 | Original-window only | 6.511 |
| 8 | Window + 512 compressed rows | 20.017 |

B8 performs eight times the work on eight owners; it does not acquire eight
kernel floors. Input readiness, buffers and task cleanup were exercised on the
actual Engine. Dedicated bottleneck controls slow HBM, MTE1, AIV read/write,
vector and Cube service independently while preserving physical work.

The retained MqsmlaKernel___call__ events report N/A tensors. Their aggregate
means of roughly 9.58 / 28.25 / 32.84 us therefore cannot be used as heldout
source-ABI accuracy rows, and none feeds this program. The receipt deliberately
does not infer source dispatch or an entire body error from those averages.
No whole-body replay or new hardware experiment was run here.
