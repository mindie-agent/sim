# Arch22 paged MLA and declared KV reuse

`ascend_paged_mla_decode_arch22` executes BF16 q1 dense latent attention with
physical heads 8 or 16, latent width 512, separate RoPE width 64 and S2 tiles
of 512. Common SplitCore selects owners from valid per-sequence contexts;
retained C++ oracle cases verify that host partition. QK and PV use the
source N128/K288/K96 and N128/K256/K128 inner refills, three L1 slots and two
L0 consumers. The source-compatible implementation does not establish the
binary or runtime tiling used by a capture.

The default `retained_kv_rows: 0` charges every GM transfer to HBM. An explicit
`retained_kv_rows` between 0 and 512 models adjacent-tile KV retention only:

```json
{
  "runtime": {
    "kernels": {
      "lowerings": {
        "paged_mla_attention": {
          "implementation": "ascend_paged_mla_decode_arch22",
          "options": {"retained_kv_rows": 512}
        }
      }
    }
  }
}
```

The MLA caller passes the same `k_nope` tensor for K and V. QK reads both
latent halves, while PV reads four 128-column slices of those same addresses
using the same page table. Source order is QK(i), PV(i−1), so two tile
generations must survive per active owner. The reservation is
`2 × active_owners × retained_rows × 576 × 2` bytes, including RoPE
interference. At 20 owners and 512 retained rows this is 22.5 MiB. A shared
L2 pool must hold the reservation until the kernel completes.

Retained rows still pay their original QK HBM read, an L2 fill and a PV L2
read. The cached PV copy uses `mte2_aic_cached`, a service mode of the same
physical `mte2_aic` pipe. `l2_read` and `l2_write` must share one L2 resource.
The reservation and every service are visible to the common Engine. Partial
retention keeps the remaining PV rows on HBM. Probability, page indices,
score/PV workspaces, atomic updates and outputs remain on their original
external-memory paths. No cross-request prefix sharing or cross-forward
cache warmth is inferred.

The reproduction producer declares L2 capacity 168 MiB and the retained
platform scalar rates 110/86 bytes per cycle for read/write, using the
existing configured 1.65 GHz and 20 AICs. These are explicit analytical
platform mappings, not measured bandwidths or a change to the SKU defaults.
The source omits the GQA sibling's explicit L2-disable hint; that permits
reuse but does not prove a capture's cache hits or freedom from eviction.

`data/verification/kimi-mla-cache-domain-20261030.json` retains all 8,496 FIA
calls from 46 A3 captures, exact source identities, ABI groups, unavailable
counter counts and native context bounds. Six A5 captures are a different
domain. B6/B12 captured query capacity is B8/B16; they are excluded from
uniform valid-row response fitting.

With one declared 512-row retention condition and no latency fit, the 128K
core predictions for TP8 B1/B64 are 0.2325/13.1354 ms against
0.2382/14.3905 ms observed; TP16 gives 0.2270/12.8479 ms against
0.2380/14.3288 ms. B64 layer controls complete in 45.7–49.5 seconds below
43 MiB peak RSS, releasing all buffers. Short-context residuals remain:
B24 at 32K is about 30% high, while B2 at 4K is about 50% low. A separate
response fitted only to 4K/16K native operators validates 8K/32K within
7.3%, but underpredicts 128K by up to 35.2%; it is not admitted to runtime.
These are conditional mechanism controls, not an all-domain accuracy pass.

Reproduce a complete four-layer Body using the same physical declaration:

```sh
python3 data/verification/kimi-mla-cache-domain-20261030.producer.py \
  --tp 8 --batch 24 --context 126000 --scope body --output /tmp/mla-body.json
```

Use `--scope core` or `--scope layer` for narrower boundaries and
`--retained-kv-rows 0` for the cold comparison. Native duration, model label,
TP label and Body totals never set retention, rates, ownership or work.
