"""Witnessed OCI/deployment -> DynamicMxQuant source selection.

[Role] Keep the native source revision separate from a quant layout or model
  label.  A source is certified only when the original deployment status/OCI
  identity is present; otherwise the producer remains explicitly unpinned.
[Invariants] r64 is certified only by the recovered r64 OCI image digest
  `8d422528...`; r31/r35 image ids are not silently mapped to r64.  No model
  label, cell id, group32 shape or trace key selects a revision.
[Timeouts & Locks] Constant metadata lookup; no I/O or locks.
[Side-Effect Policy] Pure provenance records.
"""
from __future__ import annotations

from dataclasses import dataclass

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R31_IMAGE_SHA = "9e869e81484"
R35_IMAGE_SHA = "5461a1a8253"


def image_digest(value: str) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def source_revision_from_oci(image: str = "", image_id: str = "") -> str:
    """Return r64 only for the witnessed r64 image digest, else unpinned."""
    for candidate in (image_id, image):
        digest = image_digest(candidate)
        if digest.startswith(R64_IMAGE_SHA):
            return "r64"
    return "unpinned"


def source_metadata(runtime_metadata) -> tuple[tuple[str, str], ...]:
    """Optional FeatureConfig metadata from a runtime/status record."""
    meta = dict(runtime_metadata or ())
    if meta.get("mx_quant_source_revision"):
        return ()
    image = str(meta.get("image") or meta.get("deployment_image") or meta.get("oci_image") or "")
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest") or meta.get("deployment_image_id") or "")
    if not image and not image_id:
        return ()
    return (("mx_quant_source_revision", source_revision_from_oci(image, image_id)),)
