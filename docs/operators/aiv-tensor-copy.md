# Explicit finite AIV tensor copies

`aiv_tensor_copy` is a shared recipe for materializations already emitted by
`TensorLowering`. It does not add copies, decide whether a view needs a copy,
or replace the default GPU/generic memory path. Selection is explicit through
an ordinary recipe or family declaration:

```json
{
  "runtime": {
    "kernels": {
      "families": {
        "tensor_copy": {
          "kind": "aiv_tensor_copy",
          "tile_elements": 4096,
          "slots": 2,
          "active_units": 1
        }
      }
    }
  }
}
```

These numbers demonstrate the declaration syntax. They are not inferred
vendor tiling or recommended defaults. Binding an individual recipe name
instead of the entire family permits different known implementation domains.
`tile_elements`, `slots` and `active_units` are required. Active units cannot
exceed the hardware's physical AIV count; each active owner is a member of
that physical set, not a smaller renormalized pool borrowing idle capacity.

## Physical program and evidence

The retained CANN source
`data/observations/sources/arch3510-physical-services/cann/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h`
declares VecCore GM→UB on PIPE_MTE2 at lines 53–97 and UB→GM on PIPE_MTE3 at
99–123. It distinguishes 32-byte DataCopy lengths from byte-addressable
DataCopyPad payloads. That establishes the physical directions, not the
Slice/ConcatD/TensorMove launch geometry, UB allocation or buffer-slot count.

This recipe assigns logical tiles round-robin to the explicitly declared
physical AIV subset. Each tile reads through `mte2_aiv`, optionally performs
the already declared dtype-conversion vector work, then writes through
`mte3`. Each movement also consumes shared HBM once. Per-owner read, cast and
write streams retain their order; a slot becomes reusable only after its
write completes. Multiple slots can overlap the next read with an earlier
write. A single tile is therefore a read-plus-write path, rather than the
maximum of two unrelated aggregate bandwidth demands.

Each owner reserves `slots * tile_elements * input_element_bytes` of packed
UB storage for a same-dtype copy. A cast uses separate input and output tiles,
so its reservation is `slots * tile_elements * (input_bytes + output_bytes)`.
The reservation is held for the owner's program and freed at completion.
This is the declared packed-tile storage model; an implementation with extra
row padding or workspace needs a more specific declaration. Hardware capacity
is enforced by the existing Engine. Whole tensors and other owners' idle
UB cannot enlarge that reservation.

Tensor-lowering payload, dtype and address-map metadata remain authoritative.
For example, a BF16 `[16,24]` backing narrowed to `[16,12]` moves 384 bytes
in each direction, even though its rows retain stride 24. Concatenation
preserves all source segment layouts and moves the sum of their selected
payloads once. The generic recipe does not infer cache-line amplification,
descriptor latency, source reuse or bandwidth penalties from those strides.
That limitation is visible as
`selected_payload_without_inferred_stride_transaction_penalty` in execution
metadata. Existing HBM/MTE2/MTE3 byte declarations must match the selected
payload; preparation replaces the old generic MTE2 demand with AIV demand,
so it is not charged twice.

## Invocation and public output

Prepared operators expose the mapping under
`metadata.program_execution.tensor_copy_schedule` and declare
`metadata.program_execution.execution_class = "vector"`. This is an explicit
selected program, so a same-dtype copy with zero vector arithmetic still
receives the hardware's vector invocation declaration once. The existing
outer invocation wrapper owns startup and minimum completion; individual
tiles do not acquire additional floors. Original operator `invocation_class`
declarations take precedence. Unselected generic copies, communication and
other DMA paths do not acquire an AIV class from this addition.

The same selected recipe works through `operator_costs`, `graph_costs` and
ordinary `Execution`. No GDN caller, source copy count, hardware profile or
default family binding is changed here.

## Verification and limits

Seventeen focused controls use actual `TensorLowering` and Engine execution:

- A 64-AIV profile with one active BA-copy owner pays the selected 384-byte
  read and write sequentially, without using the other 63 owners.
- BF16→FP32 tails conserve distinct read/write payload and cast work, while
  finite UB reservations match declared slots.
- Two slots overlap copy directions; a four-tile, one-owner control takes
  five transfer periods instead of eight with one slot.
- Four AIVs mapped to two read pipes keep owners 0 and 1 on the same pipe;
  neither borrows the unused second pipe.
- Concatenation, all four service-rate controls, late input, slot release,
  one outer invocation and generic-path preservation remain explicit.

The associated invocation, mechanism and caller scope totals 72 passing
tests. Independent review additionally ran 60 random tail/slot/cast controls
against a separate finite-slot recurrence and confirmed resource conservation
and makespan. These are physical-program and interface controls, not evidence
that this engineering mapping reproduces a captured vendor tiler or closes a
model-forward accuracy gap.
