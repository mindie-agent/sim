"""Observed communication execution identity from profiler-captured pilots.

[Role]
  Validate requested-vs-observed communicator expansion identity and preserve
  the parser's negative evidence (counter reset/name-only joins, unrelated local
  reductions, generic device labels).  Profiler intervals are identity and
  ordering evidence only; they are never calibration samples.

[Invariants]
  * Requested mode is not observed execution.  A requested AIV call may fall
    back to AICPU, and the record retains both.
  * A typed AIV row requires a KERNEL_AIVEC task; AICPU requires a unique
    KERNEL_AICPU schedule overlap plus SDMA/control evidence.  blockDim is
    launch geometry, not busy-core residency.
  * Name/opId-only joins are not accepted across the observed counter reset;
    connection/group/payload and temporal containment are required.
  * Unrelated local ReduceAll kernels are excluded from communication joins.
  * ``calibration_eligible`` is always false for profiled identity records.

[Timeouts & Locks]
  Pure JSON validation; no device, profiler, network, lock or sleep.

[Side-Effect Policy]
  No file or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .a3_types import A3InputError


@dataclass(frozen=True)
class CommIdentityRecord:
    run_id: str
    requested_mode: int
    requested_execution: str
    observed_expansion: str
    observed_fallback: bool
    rank: int
    device_id: int
    payload_bytes: int
    dtype: str
    reduction: str
    algorithm_reported: str
    op_id: int
    op_name: str
    group_name: str
    raw_group_name: str
    aiv_block_dim: tuple[int, ...]
    typed_task_types: tuple[str, ...]
    connection_ids: tuple[int, ...]
    rejected_name_only_task_ids: tuple[int, ...]
    checksum_reduce_kernels_excluded: int
    evidence_scope: str = "profiled_identity_pilot"
    calibration_eligible: bool = False
    profiling_timing_excluded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested_mode": self.requested_mode,
            "requested_execution": self.requested_execution,
            "observed_expansion": self.observed_expansion,
            "observed_fallback": self.observed_fallback,
            "rank": self.rank, "device_id": self.device_id,
            "payload_bytes": self.payload_bytes, "dtype": self.dtype,
            "reduction": self.reduction,
            "algorithm_reported": self.algorithm_reported,
            "op_id": self.op_id, "op_name": self.op_name,
            "group_name": self.group_name,
            "raw_group_name": self.raw_group_name,
            "aiv_block_dim": list(self.aiv_block_dim),
            "typed_task_types": list(self.typed_task_types),
            "connection_ids": list(self.connection_ids),
            "rejected_name_only_task_ids": list(
                self.rejected_name_only_task_ids
            ),
            "checksum_reduce_kernels_excluded": (
                self.checksum_reduce_kernels_excluded
            ),
            "evidence_scope": self.evidence_scope,
            "calibration_eligible": self.calibration_eligible,
            "profiling_timing_excluded": self.profiling_timing_excluded,
        }

    def matching_features(self) -> dict[str, Any]:
        """Pre-use identity descriptors for communication matching."""
        return {
            "execution_backend": self.requested_execution,
            "observed_execution": self.observed_expansion,
            "collective": "all_reduce",
            "reduction": self.reduction,
            "dtype": self.dtype,
            "algorithm": "ring",
            "group_size": 2,
            "topology_class": "sio",
            "requested_aiv_cores": (
                self.aiv_block_dim[0] if self.aiv_block_dim else 0
            ),
            "observed_aiv_cores": (
                self.aiv_block_dim[0] if self.aiv_block_dim else 0
            ),
            "algorithm_regime": (
                f"payload_{self.payload_bytes}_"
                f"{self.observed_expansion}"
            ),
            "calibration_eligible": False,
        }


def _expected_observed(requested_mode: int, payload_bytes: int) -> str:
    if requested_mode == 3 and payload_bytes < 4 * 1024 * 1024:
        return "aiv"
    return "aicpu"


def _validate_row(
    row: Mapping[str, Any], *, run_id: str, rank: int,
    checksum_excluded: int,
) -> CommIdentityRecord:
    if row.get("correctness") is not True:
        raise A3InputError("comm_identity_correctness_missing")
    if row.get("timing_scope") != "profiled_identity_pilot_excluded_from_calibration":
        raise A3InputError("comm_identity_profiler_timing_scope_invalid")
    if any(key in row for key in ("duration_ms", "raw_samples_ms", "median_ms")):
        raise A3InputError("comm_identity_profiler_timing_not_calibration")
    requested_mode = int(row.get("requested_mode", 0))
    if requested_mode not in {2, 3}:
        raise A3InputError(f"comm_identity_requested_mode_invalid:{requested_mode}")
    payload = int(row.get("payload_bytes", 0))
    if payload <= 0:
        raise A3InputError("comm_identity_payload_invalid")
    observed = str(row.get("observed_expansion", ""))
    if observed not in {"aiv", "aicpu"}:
        raise A3InputError(f"comm_identity_observed_invalid:{observed}")
    expected = _expected_observed(requested_mode, payload)
    if observed != expected:
        raise A3InputError(
            f"comm_identity_fallback_mismatch:{requested_mode}:{payload}:{observed}"
        )
    fallback = bool(row.get("observed_fallback", False))
    if fallback != (requested_mode == 3 and observed == "aicpu"):
        raise A3InputError("comm_identity_fallback_flag_invalid")
    dtype = str(row.get("dtype", ""))
    if dtype not in {"BFP16", "BF16", "bf16"}:
        raise A3InputError(f"comm_identity_dtype_invalid:{dtype}")
    typed = tuple(
        str(item.get("device_task_type", ""))
        for item in (row.get("typed_kernel_evidence") or ())
    )
    if observed == "aiv":
        if "KERNEL_AIVEC" not in typed:
            raise A3InputError("comm_identity_aiv_task_missing")
        correlation = [
            str(item.get("correlation", ""))
            for item in (row.get("typed_kernel_evidence") or ())
            if str(item.get("device_task_type")) == "KERNEL_AIVEC"
        ]
        if any("direct_comm_global_task_id" not in item for item in correlation):
            raise A3InputError("comm_identity_aiv_correlation_invalid")
    else:
        if "KERNEL_AICPU" not in typed:
            raise A3InputError("comm_identity_aicpu_task_missing")
        if "KERNEL_AIVEC" in typed:
            raise A3InputError("comm_identity_aicpu_has_aiv_task")
        if any(
            "direct_comm_global_task_id" in str(item.get("correlation", ""))
            for item in (row.get("typed_kernel_evidence") or ())
        ):
            raise A3InputError("comm_identity_aicpu_name_only_direct_join")
        physical = {
            str(item.get("device_type"))
            for item in (row.get("physical_task_types") or ())
        }
        if "SDMA_SQE" not in physical:
            raise A3InputError("comm_identity_aicpu_transport_missing")
        joins = tuple(row.get("kfc_connection_join") or ())
        if not joins:
            raise A3InputError("comm_identity_aicpu_connection_join_missing")
        for join in joins:
            if int(join.get("elements", 0)) * 2 != payload:
                raise A3InputError("comm_identity_payload_join_mismatch")
            if int(join.get("rank_size", 0)) != 2:
                raise A3InputError("comm_identity_group_size_join_mismatch")
    op_name = str(row.get("op_name", ""))
    if "allreduceAivKernel" in op_name and observed != "aiv":
        raise A3InputError("comm_identity_name_only_execution_claim")
    if any(
        str(item.get("correlation", "")).startswith("name_only")
        for item in (row.get("typed_kernel_evidence") or ())
    ):
        raise A3InputError("comm_identity_name_only_correlation")
    if not str(row.get("raw_group_name", "")):
        raise A3InputError("comm_identity_raw_group_missing")
    if checksum_excluded != 9:
        raise A3InputError("comm_identity_checksum_reduction_not_excluded")
    block_dim = tuple(int(v) for v in (row.get("aiv_block_dim") or ()))
    connections = tuple(
        int(item.get("connection_id", -1))
        for item in (row.get("typed_kernel_evidence") or ())
        if item.get("connection_id") is not None
    )
    return CommIdentityRecord(
        run_id=run_id, requested_mode=requested_mode,
        requested_execution="aiv" if requested_mode == 3 else "aicpu",
        observed_expansion=observed, observed_fallback=fallback,
        rank=rank, device_id=int(row.get("device_id", -1)),
        payload_bytes=payload, dtype="bf16",
        reduction=str(row.get("reduction", "sum") or "sum"),
        algorithm_reported=str(row.get("algorithm_reported", "")),
        op_id=int(row.get("op_id", -1)), op_name=op_name,
        group_name=str(row.get("group_name", "")),
        raw_group_name=str(row.get("raw_group_name", "")),
        aiv_block_dim=block_dim, typed_task_types=typed,
        connection_ids=connections,
        rejected_name_only_task_ids=tuple(
            int(v) for v in (row.get("rejected_name_only_task_ids") or ())
        ),
        checksum_reduce_kernels_excluded=checksum_excluded,
    )


def parse_profile_identity(payload: Mapping[str, Any]) -> tuple[CommIdentityRecord, ...]:
    """Validate an independent profiler-correlation export.

    The input is the normalized identity export, not the raw profile database
    and not an unprofiled timing sample.
    """
    run_id = str(payload.get("run_id", ""))
    results = list(payload.get("results") or ())
    if not run_id or len(results) != 4:
        raise A3InputError("comm_identity_results_incomplete")
    records: list[CommIdentityRecord] = []
    for result in results:
        rank = int(result.get("rank", -1))
        checksum = int(result.get("checksum_reduce_kernels_excluded", 0))
        rows = list(result.get("rows") or ())
        if len(rows) != 9:
            raise A3InputError("comm_identity_row_count_invalid")
        for row in rows:
            records.append(_validate_row(
                row, run_id=run_id, rank=rank, checksum_excluded=checksum,
            ))
    if len(records) != 36:
        raise A3InputError("comm_identity_records_incomplete")
    if {record.requested_mode for record in records} != {2, 3}:
        raise A3InputError("comm_identity_requested_modes_incomplete")
    if not any(record.observed_fallback for record in records):
        raise A3InputError("comm_identity_observed_fallback_missing")
    return tuple(records)


def identity_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = parse_profile_identity(payload)
    return {
        "run_id": records[0].run_id,
        "scope": "observed profiler identity only; not calibration timing",
        "records": [record.to_dict() for record in records],
        "matching_features": [record.matching_features() for record in records],
        "calibration_eligible": False,
    }


def _fixture_records(payload: Mapping[str, Any]) -> tuple[CommIdentityRecord, ...]:
    records = []
    for item in payload.get("records") or ():
        if not isinstance(item, Mapping):
            raise A3InputError("comm_identity_fixture_record_invalid")
        if item.get("calibration_eligible") is not False:
            raise A3InputError("comm_identity_fixture_calibration_forbidden")
        if item.get("profiling_timing_excluded") is not True:
            raise A3InputError("comm_identity_fixture_timing_scope_invalid")
        if item.get("observed_expansion") not in {"aiv", "aicpu"}:
            raise A3InputError("comm_identity_fixture_observed_invalid")
        if int(item.get("payload_bytes", 0)) <= 0:
            raise A3InputError("comm_identity_fixture_payload_invalid")
        if int(item.get("requested_mode", 0)) not in {2, 3}:
            raise A3InputError("comm_identity_fixture_requested_mode_invalid")
        records.append(CommIdentityRecord(
            run_id=str(item.get("run_id", "")),
            requested_mode=int(item.get("requested_mode", 0)),
            requested_execution=str(item.get("requested_execution", "")),
            observed_expansion=str(item.get("observed_expansion", "")),
            observed_fallback=bool(item.get("observed_fallback", False)),
            rank=int(item.get("rank", -1)),
            device_id=int(item.get("device_id", -1)),
            payload_bytes=int(item.get("payload_bytes", 0)),
            dtype=str(item.get("dtype", "")),
            reduction=str(item.get("reduction", "")),
            algorithm_reported=str(item.get("algorithm_reported", "")),
            op_id=int(item.get("op_id", -1)),
            op_name=str(item.get("op_name", "")),
            group_name=str(item.get("group_name", "")),
            raw_group_name=str(item.get("raw_group_name", "")),
            aiv_block_dim=tuple(int(v) for v in item.get("aiv_block_dim", ())),
            typed_task_types=tuple(
                str(v) for v in item.get("typed_task_types", ())
            ),
            connection_ids=tuple(
                int(v) for v in item.get("connection_ids", ())
            ),
            rejected_name_only_task_ids=tuple(
                int(v) for v in item.get("rejected_name_only_task_ids", ())
            ),
            checksum_reduce_kernels_excluded=int(
                item.get("checksum_reduce_kernels_excluded", 0)
            ),
        ))
    if len(records) != 36:
        raise A3InputError("comm_identity_fixture_records_incomplete")
    if {record.requested_mode for record in records} != {2, 3}:
        raise A3InputError("comm_identity_fixture_requested_modes_incomplete")
    if not any(record.observed_fallback for record in records):
        raise A3InputError("comm_identity_fixture_observed_fallback_missing")
    return tuple(records)


def load_identity(payload: Mapping[str, Any]) -> tuple[CommIdentityRecord, ...]:
    """Load the raw profiler export or the product identity fixture.

    The two business documents are distinguished by their required collections:
    a raw export carries per-rank ``results``; a product fixture carries typed
    ``records``.  Neither is version-tagged.
    """
    if "results" in payload:
        return parse_profile_identity(payload)
    if "records" in payload:
        return _fixture_records(payload)
    raise A3InputError("comm_identity_document_unrecognized")


__all__ = [
    "CommIdentityRecord", "parse_profile_identity", "identity_fixture",
    "load_identity",
]
