# Hardware transfer and device-rate overrides

This page defines the explicit machine transfer declarations and the device
rate aliases consumed by the Ascend input adapter. All declarations are
metadata: they never probe a device, load weights, or write artifacts.

## Machine transfer channels

`machine.transfer_channels` is merged into the registered SKU profile **per
channel**. Both forms are accepted:

```json
{"transfer_channels": {"d2d_inter_node": {"alpha_us": 7.0}}}
```

```json
{"transfer_channels": [
  {"channel": "d2d_intra_node", "alpha_us": 2.0},
  {"channel": "d2d_inter_node", "effective_uni_bw_gbps": 25.0}
]}
```

* A mapping is keyed by channel. A list record must declare `channel`.
* Duplicate channels in a list, channels absent from the registered profile,
  and fields that have no supported meaning are errors. They are not ignored.
* A partial record preserves every inherited channel field, including
  `alpha_us`, `protocol`, `transport_mode`, `channel_semantics`, and the
  bandwidth provenance fields.
* Explicit `channel_semantics` and neutral `semantics` are validated; neutral
  `semantics` wins when both are supplied. Supported values
  are `shared`, `per_port`, `per_pair`, `direct`, `fully_connected`, `mesh`,
  and `switched`; `switched` maps to `per_port` in the neutral link graph.
  Without an explicit declaration, generated links retain the original
  `per_port` behavior. Profile fabric annotations such as `direct` or `mesh`
  do not silently change the capacity-sharing model.
* The generic link framing fields `transaction_bytes`, `alignment_bytes`, and
  `protocol_overhead_bytes`, plus `latency_us`, are copied into the generated
  `Link`. `alpha_us` is the channel spelling of the same latency. Framing is
  preserved by routing and the Engine, not only by the parser.
* `machine.links` is the highest-priority declaration: when present it is used
  exactly as given and no links are generated from `transfer_channels`,
  convenience fields, or the profile. `links: []` explicitly means no links.
  The profile JSON itself is never modified.

### Units and effective bandwidth

`transfer_channels.*_bw_gbps` and the legacy `Link.capacity_gbps` field retain
their historical **GB/s** meaning. New machine convenience fields use standard
units:

* `inter_node_bw_gbps` and `intra_node_bw_gbps` are decimal **Gb/s** and are
  divided by 8.
* `inter_node_bw_bytes_per_s` and `intra_node_bw_bytes_per_s` are **B/s**.
  At the same layer `bytes_per_s` wins over `gbps`.

Thus `200 Gb/s`, `25 GB/s` from a transfer channel, and `25e9 B/s` all become
`25e9 B/s` (`capacity_bytes_per_ms = 25e6`). The convenience fields affect only
their corresponding `d2d_inter_node` or `d2d_intra_node` channel.

For one defined channel, effective unidirectional bandwidth is chosen at the
current layer as:

```
explicit effective_uni_bw_gbps
  > explicit calibrated_uni_bw_gbps
  > explicit derived_uni_bw_gbps
  > explicit uni_peak_bw_gbps
```

An explicit value is not masked by an older profile `effective_uni_bw_gbps`.
An explicit `effective_uni_bw_gbps: null` disables the channel even if
lower-tier values remain. Without that field, the first non-null explicitly
declared calibrated, derived or peak value supplies the effective rate. If all
explicit bandwidth values are null, the link is not generated. If a record
changes no bandwidth tier, the inherited effective value remains in force.

Overall link priority is:

```
explicit machine.links
  > explicit channel effective bandwidth
  > convenience bytes_per_s
  > same-layer Gb/s
  > profile effective bandwidth
```

`Hardware.assumptions` always exposes the normalized unit convention under
`transfer_bandwidth_units` and this ordering under
`transfer_bandwidth_precedence`.

### Generated topology

Without explicit `machine.links`, links are generated with the existing
topology rules: `d2d_intra_package` groups by `dies_per_package`,
`d2d_intra_node` by `ranks_per_node`, and `d2d_inter_node` covers ranks across
nodes only when `num_devices > ranks_per_node`. Changing bandwidth does not
rewrite `Topology.route`, add hops, or alter wire bytes.
Only these three D2D channels produce device graph links. Other registered
channels, including H2D and D2H, remain metadata; changing them does not change
host/offload service or the separate P/D connector. This support scope and the
default source-port capacity rule are exposed in `Hardware.assumptions`.

## Device rate entry points

Device declarations are normalized layer by layer: registered profile, common
fields directly on `machine`, `machine.device`, then `machine.devices[i]`
(or `machine.dies[i]`). Each later layer overrides the preceding one. The same
common-field path works for neutral hardware without an Ascend backend. Native
fields and neutral `rates` both feed the same
physical role map; within one layer an explicit `rates.<role>` wins over the
native alias.

| Field | Roles |
| --- | --- |
| `hbm_bw_bytes_per_s` | `hbm` |
| `vector_ops_per_s` | `vector` |
| `mte_bw_bytes_per_s` | `mte2`, `mte2_aic`, `mte2_aiv`, `mte3` |
| `mte1_bw_bytes_per_s` | `mte1` |
| `mte2_bw_bytes_per_s` | `mte2` |
| `mte2_aic_bw_bytes_per_s` | `mte2_aic` |
| `mte2_aiv_bw_bytes_per_s` | `mte2_aiv` |
| `mte3_bw_bytes_per_s` | `mte3` |
| `fixpipe_bw_bytes_per_s` | `fixpipe` |

`mte_bw_bytes_per_s` is a convenience declaration for all four MTE2/MTE3
roles. An explicit `rates.mte2_aic`, `rates.mte2_aiv`, `rates.mte3`, or
`rates.mte2` overrides the corresponding convenience role. Rates change
service time only. They never change `resource_units`, HBM capacity, owner
counts, or the shared HBM rate.

`cube_flops_per_s` sets only BF16 and FP16 peaks. Explicit
`compute_peaks.<dtype>` values win, and no INT4/FP8 ratio is inferred from a
BF16/FP16 declaration. `vector_ops_per_s` changes only the vector rate, and
`hbm_bw_bytes_per_s` changes only the HBM rate; neither rewrites owners or
other rates. These device-level fields do not expand into machine topology,
link, or placement declarations.

The specific pipe aliases override `mte_bw_bytes_per_s` at the same layer;
neutral `rates` override both. Common memory, compute, buffer, owner and
service-mode declarations use the same hierarchy. Rank, node and package
placement fields remain per-device records; common rate defaults never assign
every device to one rank. A normalized `Hardware.to_dict()` or `Case.to_dict()`
can be parsed again without scaling inherited modes twice.

### Physical service-mode inheritance

When a layer explicitly overrides an inherited physical resource rate, every
inherited service mode bound to that resource scales by `new_base / old_base`.
The mode and its base resource therefore retain their declared supply ratio.
For example, `mte1_l0a` and `mte1_l0b` follow their shared `mte1` rate, while A5
`ub_copy` and `reg_issue_slots` follow `vector`. Modes bound to other resources
remain unchanged. This avoids mode normalization cancelling an explicit base
rate change; it does not add independent capacity.

A mode `rate` declared in the current layer is explicit and is **not**
rescaled. Partial records preserve inherited fields, so a declaration with only
`rate` keeps the inherited `resource`. The same rule applies at every later layer and to
each per-device record independently. Within one layer, neutral
`rates.mte1` wins over the `mte1_bw_bytes_per_s` alias. No hidden provenance
or shadow state is retained: the normalized device contains the final rate,
mode records, and owners only.
