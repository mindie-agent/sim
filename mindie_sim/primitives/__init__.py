"""Public CPU-only API for L0 primitive calibration and lowering.

[Role] Export the table schema, trusted loader and shared Engine lowering API.
[Invariants] Importing this package performs no calibration or device work.
[Timeouts & Locks] None.
[Side-Effect Policy] Imports type definitions and modeling functions.
"""
from .loader import (
    RateRole,
    default_primitives_dir,
    load_manifest,
    load_primitives,
    resolve_primitive_rate,
)
from .lowering import (
    BYTES_PER_S_TO_MS,
    MS_PER_SECOND,
    NS_TO_MS,
    US_TO_MS,
    PrimitiveLoweringResult,
    ShortFramePolicy,
    frame_wire_bytes,
    lower_primitive,
    submit_lowered_primitive,
)
from .schema import (
    MAX_CALIBRATED_CV_PCT,
    PRIMITIVE_IDS,
    PRIMITIVE_SCHEMA_VERSION,
    BoundKind,
    CellStatus,
    PrimitiveCell,
    PrimitiveProvenance,
    PrimitiveTable,
    SiliconFingerprint,
    UncertaintyBand,
    compute_envelope_sha256,
    parse_primitive_table,
)
from .sealer import (
    SKU_DEFAULT_FILES,
    build_primitive_table_payload,
    build_primitive_table_with_diagnostics,
    canonical_capture_bytes,
    sanitize_record_alias,
    summarize_samples,
    summarize_samples_with_reason,
    update_manifest,
)

__all__ = [
    "BYTES_PER_S_TO_MS",
    "MS_PER_SECOND",
    "NS_TO_MS",
    "US_TO_MS",
    "MAX_CALIBRATED_CV_PCT",
    "PRIMITIVE_IDS",
    "PRIMITIVE_SCHEMA_VERSION",
    "SKU_DEFAULT_FILES",
    "BoundKind",
    "CellStatus",
    "RateRole",
    "ShortFramePolicy",
    "UncertaintyBand",
    "SiliconFingerprint",
    "PrimitiveProvenance",
    "PrimitiveCell",
    "PrimitiveTable",
    "PrimitiveLoweringResult",
    "build_primitive_table_payload",
    "build_primitive_table_with_diagnostics",
    "canonical_capture_bytes",
    "compute_envelope_sha256",
    "parse_primitive_table",
    "default_primitives_dir",
    "load_manifest",
    "load_primitives",
    "resolve_primitive_rate",
    "frame_wire_bytes",
    "lower_primitive",
    "sanitize_record_alias",
    "submit_lowered_primitive",
    "summarize_samples",
    "summarize_samples_with_reason",
    "update_manifest",
]
