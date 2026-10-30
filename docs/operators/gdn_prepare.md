# Native GDN Prepare

This plan describes the pinned arch35 implementation. The public
`gdn_prefill_arch35` lowering selects it explicitly; measurements are neither
required nor consulted. The source revision is
`a4a795857868729fbdbb0118c1e06c8655a7b31b`, from image
`sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4`.
Retained source identities are in
`data/observations/sources/gdn-prefill-image-c3d00a196905/manifest.json` and
`data/observations/sources/fla-a4a7958/`.

## Selected caller and flags

The integrated caller is noninterleaved QKVZ plus BA, PCP=1, BF16 Q/K/V,
64-token chunks, K=V=128 and FP32 recurrent state. It selects Q/K normalization,
FP32 g/beta, no repeated sigmoid, exp2, A output and internal beta_eff output.
The standalone planner also accepts its explicit alternate gate, precision,
output and V=256 branches; these are not the integrated caller's defaults.

The exact composite implementation casts beta to FP32 and uses state layout V,K.
Prepare's L0 wrapper allocates its internal FP32 beta_eff even when the outer
optional output is null. `output_beta=True` therefore represents actual kernel
work, not a public tensor that the caller necessarily consumes.

`fp32_mmad_mode` separates plain FP32 from inherited HF32. Plain FP32 is the
selected declaration. The retained binary has five static FP32 MMAD sites and
no local CTRL writes; its inherited runtime HF32 state was not captured.
Storage k0=8 is not an eight-wide ordinary FP32 compute instruction.

## Physical program

The source pack walk assigns three value heads per pack for GQA=3 and four
otherwise. Full GQA=2 packs reorder their owners. Only the owning key-head task
normalizes Q/K and executes KKT; every value head retains its own gate and W/U.
Stage 0 runs on both AIVs of every launched core, including cores without packs.

| Source stage | Executed work and dependencies |
| --- | --- |
| 0 | Build zero/identity leaves and copy them from UB to L1 |
| 1 | Tail fill, owner K normalization/L1 publication, Q normalization, cumsum and beta |
| 2 | BF16 KKT after that owner's K publication, independently of later Q/gate work |
| 3 | Gate preparation for all local heads, then KKT application, leaf pack and VCS |
| 4 | Four-stage blocked solve begins with I@I; Y is written through Fixpipe and reloaded |
| 5 | Two FP32 64-cubed products with distinct operand banks; full A reload or tail local publication |
| 6 | Beta/gate scaling of V and K, followed by direct UB-to-L1 copies |
| 7 | BF16 W/U products; source-dependent local retention or direct GM output |

There are four dense FP32 64-cubed products per value-head chunk. Their count
is independent of the mathematical triangular solve's FLOP shortcut. KKT and
W/U retain padded 64-row matrix tiles while global traffic retains valid rows.
MTE1 operand bytes, Fixpipe output bytes and local-only copies are explicit.

The two-pack issue window preserves per-task publication, source pipe order,
L0 operand reuse and cross-pack lifetime edges. K input preprocessing can
overlap the previous pack until its L1 reuse point. K export can overlap Q
reads. Negative-L publication can overlap VCS. V=128 W/U drain occurs in the
next pack after its local Stage 3, or in the terminal drain; V=256 U goes
directly to GM and W remains retained.

## Physical ownership and instruction accounting

Every matrix stage owns one AIC; vector stages own the corresponding AIV.
AIC/AIV reads use separately declared resources when present. Shared HBM is
never divided by core count. One core cannot borrow an idle core's compute or
copy supply. Each active physical owner reserves its own storage:

| Pool | Bytes per owner |
| --- | ---: |
| UB, each of two AIVs | 241664 |
| L1, AIC | 507904 |
| L0A / L0B, AIC | 65536 each |
| L0C, AIC | 262144 |

`gdn_registers.py` retains source calls, active lanes, UB payload and barriers
for CumsumScaleVF and L2NormK128VF. `InstructionInventory.capacity_work` projects
independent issue, register-write, UB-access, ALU, special, reduction and shuffle
capacity constraints. These are necessary capacity bounds. They do not assign
fractional-cycle instruction latency or assume a realizable dual-issue schedule.

For 64 BF16 rows, L2Norm has 1219 source calls, 1475 issue slots, 1025 data
register writes, 320 UB accesses and 192 stores. Cumsum has 88 calls, 143 issue
slots, 128 register writes, seven UB accesses and three stores. Masked scalar
operations retain their calls even when only one lane is active. VCS and other
vector functions still use elementary lane-work envelopes.

## Verification and unresolved mechanisms

`tests/operators/test_gdn_prepare.py` independently interprets register calls,
UB addresses, prefix sums and BF16 normalization. It also enumerates source
work, ownership, physical stage order, tail paths and the real frontend.
`tests/test_gdn_integration.py` compares isolated and iteration execution using
the same selected recipe and checks buffer release.

The source-matched 16K ABI replay is retained in
`data/verification/gdn-native-20261028-integrated-caller.json`: 1226.548 us
predicted versus 1541.022 us observed (-20.407%). CSV shapes do not expose
sequence-length values or every dispatch flag; those remain case/source
declarations. This is not latency accuracy acceptance.

Masked Sqrt/Div service, dependency latency, compiler bundles, scalar/flag costs,
cache behavior and remaining vector instruction classes are unresolved.
`compiler_ftz=True` selects the retained hardware path; the software FTZ=false
expansion is not implemented. No residual multiplier or latency floor fills
these omissions. See `docs/operator-mechanisms.md` for the current integrated
boundary and preserved evidence.
