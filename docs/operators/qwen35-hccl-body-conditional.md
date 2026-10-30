# Qwen3.5 TP4 complete bodies with conditional HCCL sources

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The three cases in
[`qwen35-hccl-conditional`](../../examples/cases/high-concurrency-20261030/qwen35-hccl-conditional)
retain the complete 40-layer W8A8 model, GDN source caller, grouped INT32
caller, forced-balanced routing and TP4/EP4 placement. Their only simulation
change is explicit HCCL selection: AllReduce AIV `mesh_twoshot` with 60 blocks,
AllGather AIV `mesh` with 64 blocks, both on TP. The retained source is a
conditional template, not verified capture dispatch.

**The current full graph executes 80 AllReduce calls and zero AllGather calls.**
Its MoE path still has 80 EP all-to-all calls. Installing an AllGather selector
does not insert a missing model/caller collective. The measured population has
80 AllReduce and 40 AllGather events per body. No EP all-to-all was renamed to
AllGather to make these counts agree. Identifying the source caller and exact
payload for those 40 measured calls remains necessary.

## Complete-body results

The observed population comes from the existing profiler HCCL ledger, using
only the same retained q4/B4, q4/B64 or q4/B128 geometry. It contains 84, 79 and
76 complete bodies. Using all 91/96/99 bodies from the unfiltered extraction
would mix other geometries. Context4096 is declared; active context lengths
are not established by cache capacity. Measured first-layer waits remain in
the body span. Host submission and inter-step waits are excluded from Sim.

| Batch | Observed mean ms | Baseline ms | Source AR condition ms | Source relative gap | Source wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 11.551367 | 5.528757 | 5.008220 | -56.64% | 5.03 |
| 64 | 21.956845 | 11.377590 | 11.688436 | -46.77% | 14.71 |
| 128 | 28.584774 | 15.257969 | 16.455612 | -42.43% | 21.85 |

These are conditional same-geometry complete-body comparisons. None passes
the 15% body target. The smallest case becomes less accurate; the results are
not presented as monotonic accuracy improvement or evidence of the captured
algorithm. No observed duration determined a rate, floor or body offset.

| Batch | Observed AR activity ms/body | Predicted AR interval sum ms/body | Observed AG activity ms/body | Executed AG calls |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2.113192 | 0.363525 | 0.422939 | 0 |
| 64 | 2.892219 | 1.495839 | 0.962846 | 0 |
| 128 | 3.779760 | 2.703628 | 1.206973 | 0 |

Predicted AR intervals span each complete native invocation from group entry
to exit. Local phases and wire stages are not summed again. Observed HCOM
summary duplicates and unlinked AICPU wrappers are not added to the device
event population. These activity sums remain distinct from body critical-path
latency and cannot be added linearly to the body residual.

## Verification and provenance

The six actual simulations completed with zero live tasks/programs. Three
non-HCCL graph signatures (all serialized operator fields) and all non-HCCL
semantic work counters match exactly between arms. Full-body source runs use
80.9–84.7 MiB maximum RSS on this macOS host. Each fresh worker had the same
60-second limit; no layer count or model geometry was reduced.

The first source attempts failed visibly because the old representative-rank
guard required exactly equal shard sizes. Source tail elements produce small
differences between TP ranks even for equal input payloads. Runtime
`20d7c5c5` preserves the actual send matrix and declares group-mean transport
accounting; local AIV work remains explicitly representative rank0. It is not
valid to multiply that rank's HBM by TP and call it exact cohort work.

Completed baselines were reused after this targeted transport change. The
final six-arm report
keeps each row's runtime package identity, input hash, declared/executed call
counts, resource ledger, RSS and wall/CPU time. The
previous report
retains the three source-domain failures and original baselines. Reproduction
uses the producer;
`--reuse-completed` reruns only unfinished arms. The producer's initial missing
`analysis_input` declaration was corrected before all retained baseline runs.

The next structural task is to locate the observed AllGather caller, preserving
its real data representation, producer/consumer dependencies and scope. Extra
polling, scalar control and first-operator flag initialization remain separate
source coverage gaps; this comparison supplies no justification to fit them
from the remaining whole-body difference.
