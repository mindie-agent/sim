"""Declared native correspondences over existing semantic operator groups.

[Role] Retain fusion membership, scope and evidence at the composing builder.
[Invariants] Declarations use model/runtime geometry, never captured shapes.
  A group adds no execution node, work or measured duration.
[Timeouts & Locks] Finite layer metadata updates; no I/O or locks.
[Side-Effect Policy] Replaces records in the caller-owned operator list.
"""
from dataclasses import replace


_SOURCE_FILES = {
    "vllm_ascend/ops/gdn.py": (
        "724f43cc24ee4b4d4d01c36e7967d02d95646b5e4ab011940e16bb5374a20c9f", "local_worktree"),
    "vllm_ascend/models/deepseek_v4/model.py": (
        "257c415f820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab", "committed"),
    "vllm_ascend/models/deepseek_v4/compressor.py": (
        "37ca695e3b9564d95c60f628dbfee8268264885d774b91ac6caeeb97c2375ce9", "committed"),
    "vllm_ascend/models/kimi_k3.py": (
        "498839ee000fd1d0731915613de1b98b562474c9da5c5357c686b5ae2cc0922d", "local_worktree"),
    "vllm_ascend/ops/kimi_kda.py": (
        "2e9890d18312708875ca2ad0a6dda2116c6b45fd521e9041467c8499ad2da96d", "local_worktree"),
    "vllm_ascend/ops/fused_moe/moe_mlp.py": (
        "9a18993599563f7d65c47c06ad9ac5ac406590d779bde2def0d86620e6997656", "local_worktree"),
}


def declare_group(ops, *, key, role, family, members, entries, anchor, view,
                  implementation, source=None, block="attention", scope_role=None,
                  coverage=None):
    declaration = {
        "key": key, "role": role, "family": family,
        "op_ids": tuple(members), "entry_op_ids": tuple(entries),
        "tensor_view": view, "implementation": implementation, "source": source,
    }
    if coverage is not None:
        declaration["coverage"] = coverage
    selected = set(members)
    for index, op in enumerate(ops):
        if op.op_id not in selected:
            continue
        meta = dict(op.metadata)
        if block is not None:
            meta["structure_block"] = block
        if scope_role is not None:
            meta["structure_role"] = scope_role
        meta["structure_subgraph_memberships"] = (
            *meta.get("structure_subgraph_memberships", ()), key)
        if op.op_id in entries:
            meta["structure_subgraph_entries"] = (
                *meta.get("structure_subgraph_entries", ()), key)
        if op.op_id == anchor:
            meta["structure_subgraphs"] = (*meta.get("structure_subgraphs", ()), declaration)
        ops[index] = replace(op, metadata=tuple(meta.items()))


def source_reference(path, symbol):
    digest, state = _SOURCE_FILES[path]
    return {
        "repository": "vllm-ascend",
        "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
        "source_state": state, "sha256": digest,
        "path": path, "symbol": symbol,
        "scope": "independent_source_semantics; not_capture_binary_identity",
    }
