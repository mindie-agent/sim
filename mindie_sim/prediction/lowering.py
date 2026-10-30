"""Small graph construction helpers for semantic extensions.

[Role] Build typed operators with one common placement and quantization context.
[Invariants] Helpers preserve work, dependencies and units.
[Timeouts & Locks] Pure, bounded by the generated graph.
[Side-Effect Policy] Appends only to a caller-owned list.
"""
from functools import cached_property

from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op, cache_store_op
from mindie_sim.schema.operators import OpSpec
from mindie_sim.catalog.quant import projection_quant


class Builder:
    def __init__(self, prefix, plan, stage, views, model, mode):
        self.prefix, self.plan, self.model = prefix, plan, model
        self.common = dict(stage=stage, layer=plan.index,
                           requests=tuple(v.request_id for v in views), phase=mode)
        self.ops = []

    @cached_property
    def quant(self):
        # Vector-only callback builders have no model matrix representation.
        # Resolve per-layer projection precision only when a matrix needs it.
        return projection_quant(self.model.quant, self.plan)

    def add(self, op):
        self.ops.append(op)
        return op.op_id

    def matrix(self, name, deps, m, n, k, category="mla", **kwargs):
        return self.add(gemm_op(
            op_id=self.prefix + "." + name, deps=tuple(deps), m=m, n=n, k=k,
            quant=self.quant, category=category, recipe=name,
            family=kwargs.pop("family", "gemm"), **self.common, **kwargs,
        ))

    def vector(self, name, deps, elements, operations, **kwargs):
        return self.add(vector_op(
            op_id=self.prefix + "." + name, deps=tuple(deps), elements=elements,
            ops_per_element=operations, dtype=kwargs.pop("dtype", self.model.activation_dtype),
            recipe=name, family=kwargs.pop("family", "elementwise"), **self.common, **kwargs,
        ))

    def store(self, name, deps, rows, row_bytes, dtype):
        return self.add(cache_store_op(
            op_id=self.prefix + "." + name, deps=tuple(deps), rows=rows,
            row_bytes=row_bytes, dtype=dtype, kind=name, recipe=name, **self.common,
        ))

    def communicate(self, name, deps, volume, family, axis="tp", **metadata):
        return self.add(comm_op(
            op_id=self.prefix + "." + name, deps=tuple(deps), link_bytes=volume,
            family=family, recipe=name, shape=(volume,),
            metadata=tuple(sorted({"group_axis": axis, **metadata}.items())), **self.common,
        ))

    def work(self, name, deps, shape, *, flops=0, vector=0, read=0, write=0,
             family="attention", dtype=None, metadata=()):
        return self.add(OpSpec(
            op_id=self.prefix + "." + name, deps=tuple(deps), recipe=name, family=family,
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
            flops=flops, vector_ops=vector, hbm_bytes=read + write,
            mte2_bytes=read, mte3_bytes=write, dtype=dtype or self.model.compute_dtype,
            metadata=tuple(metadata), **self.common,
        ))
