# Shared-core proposal: finite owner supply in the generic envelope

The reviewed sparse decoder is owner-limited, not chip-wide. An arch22
`SparseFlashAttention` query task occupies one AIC and its paired AIVs. The
generic `ResourceEnvelope` still prices the `latent_attention0` memory ledger
using the full device `hbm`/`mte2` rate, so a one-row decode has a 1.99 us
body-only bound while the source-owned task cannot borrow inactive AICs'
copy supply.

`prediction.indexed_attention` already applies this through explicit stage
placement. Extending the generic envelope is a shared-core change owned by
the runtime integration lane, not by this sparse-attention branch. A minimal
source-preserving interface would be:

1. The attention builder declares `active_owners` (the number of outer AIC
   tasks) and the per-owner `mte2_aic`/`mte2_aiv` partition instead of one
   aggregate `mte2` demand.
2. The envelope partitions copy and memory service over those owned units,
   using the existing `Stage.placement`/`ownership.unit_group` machinery.
3. The declaration is rejected when source geometry is outside the reviewed
   TND, one-query-per-AIC, block-size-one domain. Compression, dense windows
   and `SparseAttnSharedkv` retain their own interfaces.

No hit ratio, scalar residual or measured duration is required. Until the
core change exists, the indexed pipeline remains an explicit alternative and
the generic main case is unchanged. Selecting it merely because it is closer
to an observed span would violate the calibration boundary.
