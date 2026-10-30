# MC2 source-operation service supply

The explicitly selected arch35 full-mesh phases can now consume independently
declared scalar UB access, scalar arithmetic and explicit pipe-barrier services.
The existing physical memory, Vector and network work remains unchanged. With
no such hardware declarations, known operation counts stay visible as unpriced
inventory and the original execution graph is unchanged.

This is a partial inventory of source calls, not a reconstruction of compiled
ISA. It does not convert native `scalar_time` into a per-instruction latency,
add a residual floor, or interpret data-arrival wait time as resource service.

## Source and supported operations

The retained `ops-transformer` revision is
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. Both implementations are reused by
the selected V3 protocol:

| Source file under `mc2/` | SHA-256 |
|---|---|
| `moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h` | `76db25293e4d3b34784012ace8fd824a977e4055fa974a9e3370c99975467c1e` |
| `moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h` | `133aef717fce163e06f942c82ffb162e5cc0fe8d8e9c56fcbaaccd13afd63429` |

These pins establish source semantics; they do not establish capture binary
identity or resolve the existing conditional payload-owner mapping.

| Existing phase | Added operations and position |
|---|---|
| Dispatch sender prefix | Expert ID UB read before the prefix scan; successful result UB read and four explicit `PIPE_V` barriers after it. The actual strided assignment index determines ownership and prefix length. |
| Dispatch framing | Three `FillTriple` UB stores per assignment, before the frame service. |
| Expert count | One status UB store per expert; one reduction-result read and four barriers when the ID array is nonempty; one initial barrier per count owner. |
| Count receipt and prefix | Gather-pattern stores, successful result reads, count/prefix reads and stores; one scalar addition per expert; explicit prefix barriers. Contiguous expert-tail partitions follow the source count owners. |
| Prefix ready | One successful UB result read per participating AIV. |
| Combine return framing | Three expand-index UB reads and one explicit frame barrier per row. |
| Combine successful flags | One initialization barrier per output token and one UB read plus SFF per successful 64-bit mask word. |
| Combine weighted reduction | One router-weight UB read before each token's dependent slot payload read. The finite per-owner token pipeline remains in place. |

Source line ranges are retained in `program_execution.primitive_services` with
per-owner, per-position counts, priced service and known unpriced service.
This supplements the earlier [assignment-prefix inventory](mc2-assignment-prefix.md).
The broad `unmodeled_services` list still describes remaining control work;
only the enumerated subset has gained a pricing interface.

Failed polling iterations, receive-batch-dependent payload checks, `SyncFunc`
and queue/event issue, and unlisted scalar address/control operations remain
outside this inventory. Their absence does not block analytic simulation.
Filtered assignment IDs are not reconstructed from total row counts. Fractional
expected row work retains the existing explicitly declared row-grid closure.

## Physical binding and limits

`compute_arch.instruction_services` declares a service's work unit, owner axis
and either per-owner rate or issue time. For example, a synthetic test can
declare `scalar_ub_access` in `calls` on the Vector owner axis. No MC2 hardware
rate is installed by this change. A measured service would also need provenance
and a supported hardware/implementation domain.

The planner adds a stage only when its role is declared. It uses the ordinary
`RecipeContext.owned` mapping, so two active AIVs sharing one physical scalar
pipe contend on that pipe; an idle pipe cannot supply them. Instruction work is
added once to the phase ledger and once to its program. The outer native kernel
group continues to own invocation cost and dependencies.

Barriers inside a pre-existing aggregate Vector step are conservatively retired
after that step. This preserves the source-side ordering boundary but is not an
instruction-level overlap model. Some owner schedules and expected row counts
are still conditional; declaring a rate does not make those dispatch choices
measured facts. Existing causal peer-arrival dependencies remain separate.

## Verification

The compact retained receipt and reproducible local producer are:

- `data/verification/mc2-instruction-services-20261030.json`
- `data/verification/mc2-instruction-services-20261030.producer.py`

The producer loads the prior committed planner (`83f54ee4`) solely as a test
oracle. All **84 old/new phase pairs** have exactly equal complete traces,
physical Work, serviced resources, task counts and completion times when no
instruction service is declared. The controls include idle senders, tails,
BF16/INT8/MX dispatch, combine, fractional expected rows, split copy ownership
and late input. They retain no full trace history in the report.

The focused suite passed **99 tests in 1.96 seconds**. It additionally executes
real kernels and an ordinary two-DP
native graph: independent source-call arithmetic; one successful flag mask;
per-slot weight-read dependencies; a 4-AIV/2-scalar-pipe contention case; partial
service availability; 0.5/1/2 supply scans; late rank input; unchanged wire and
HBM/copy/Vector work; and zero remaining tasks/programs. These are mechanism
controls, not native-call or Body accuracy claims. No observation or aggregate
PMU duration was used to assign instruction rates.
