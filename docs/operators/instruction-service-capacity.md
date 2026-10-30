# Owner-local instruction and operation service

`device.compute_arch.instruction_services` declares hardware capacity for a
counted operation. Each service names `unit`, `owner_axis` (`vector` or `cube`
for the usual AIV/AIC partitions), and either `per_owner_rate` in units/second,
`issue_ns` per unit, or `cycles_per_unit` with an explicit `clock_hz`.
`provenance` retains the measurement/source and domain.
Normalization produces ordinary Engine rates, work units and owner counts;
explicit `device.rates` overrides retain precedence. This is one Engine, not a
second timing calculator.

An optional `resource` binds the service to an existing physical pipe through
`service_modes`. Its work keeps the declared units while Engine service shares
that pipe's capacity and owners. An explicit `service_modes.<name>.rate`
overrides this mode's aggregate supply. Derived rates/modes are recomputed on
hardware round trips, so changing their clock or linked physical supply does
not silently preserve an obsolete derived override.

`PrimitiveInventory.partition` keeps declared work and unknown work separate.
Source callers attach both to their program ledger. Unknown work remains an
unpriced count; it is not evidence of zero cost. Existing observed whole-call
service envelopes remain valid and are inclusive, so a newly priced phase is
not added again on top of the same native-call observation.

The capacity classes cover scalar access/ALU, DMA descriptor issue, SIMT warp
issue and atomic transactions, sort/merge operations, and pipe/core barriers.
The class names do not assert that a source API call is a single ISA operation.
In particular a source `Sort` does not prove its merge-instruction count, and
`SetAtomicAdd + DataCopyPad` is an atomic DMA operation, not one SIMT AtomicAdd
per expert. A stage must retain actual owner placement and dependency order.
A completed flag poll may consume issue service; its data-arrival dependency
must not be priced a second time as a fixed wait.

The retained architecture tables establish capabilities, ownership and cache
line sizes, but do not provide universal scalar/DMA/barrier issue intervals.
The convolution-state PMU numbers are busy time per source program in a mixed
kernel. They can parameterize that declared phase/domain; they cannot establish
a hardware-wide single-descriptor latency without its lowering/count contract.
No primitive rate is enabled by default.

## Optional Sort32 specification bound

The Ascend adapter accepts this explicit preset:

```json
{"backend":"ascend", "sku":"Ascend950DT_9581", "num_devices":1,
 "device":{"compute_arch":{"presets":["ascend_sort32_spec"]}}}
```

The pinned AscendC performance table declares **2 elements/cycle** for a
full `Sort32` repeat on half/float inputs (950 table 2; A2/A3 table 14).
The pipeline table assigns this API to **AIV / PIPE_V**. The preset therefore
declares `vbitsort32` work in `blocks32`, `16 cycles/block`, and `resource=vector`.
It does not create an independent sorting resource. With the 950DT_9581
profile's declared 1.65 GHz and 64 AIVs, per-owner capacity is
103,125,000 blocks/second; aggregate capacity is 6,600,000,000 blocks/second.
The original 32-element block counts remain unchanged.

The default clock is that SKU's `source.declaration.configured_clock_mhz`;
`compute_arch.clock_hz` explicitly overrides it before preset expansion.
Normalized hardware retains `clock_hz` and `cycles_per_unit` inside the
resolved instruction service, allowing subsequent frequency sweeps there.
`instruction_services.vbitsort32.per_owner_rate` or `issue_ns` supplies an
explicit capacity override; the generated `resource=vector` binding remains.
No observed kernel duration or platform-INI substitution chooses this clock.
Changing only `vector_ops_per_s` or `rates.vector` does not change this
clock-based Sort32 capacity: it changes the vector-work conversion on the
shared pipe. A frequency experiment must change the explicit clock declaration;
the adapter never infers a new clock from a vector-operation rate.

Provenance is `kind=isa_spec_bound` with
`spec_level=AscendC_API_theoretical_throughput`. This label describes the source
bound; it does not claim a native ISA issue interval, dependency latency or
measured kernel completion time. The two retained sources are
`vector_instruction_theoretical_perf_summary.md` and
`api_pipeline_type_summary.md` from `hicann/asc-devkit` revision
`56500ea0dc219380f50f4a17ce73ef5264f830fc`; their paths and SHA-256 digests are
preserved in the resolved service provenance.

Only separately established full `Sort32` repeats can use this supply.
`source_sort32_blocks` in the MoE-init planner normalizes an entire
`Sort<float,true>` macro, whose merge expansion is unknown; it remains
unpriced. The same table's MrgSort elements/cycle cannot establish a fixed
`merges4` rate without input lengths and exhaustion behavior. Scalar flags,
barriers and DMA issue also remain undeclared where no cycle bound exists.

An optional `scale_with` transfers a declared capacity under hardware sweeps:

```json
{"unit":"descriptors", "owner_axis":"vector", "issue_ns":1000,
 "scale_with":{"resource":"mte2_aiv", "reference_per_owner_rate":100000000000},
 "provenance":{"kind":"explicit_transfer_assumption"}}
```

This numerical example is a synthetic control. Supply scales by the current
MTE2_AIV **per-owner** rate divided by the reference. Adding idle owners alone
does not accelerate a pinned operation. Direct `rates` overrides still win.
The declaration is a transfer assumption: correlation in a PMU capture does not
automatically establish it. Services without this field remain independent;
conditional joint copy-busy parameters are not silently linked to HBM or AIV.

Validation uses real Engine execution with delayed input, same/different-owner
competition, 0.5/1/2 supply scans, conserved counts and hardware round trips.
