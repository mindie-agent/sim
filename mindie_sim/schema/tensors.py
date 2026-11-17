"""Strided tensor views without storage allocation.

[Role] Describe address maps and decide whether a reshape can share storage.
[Invariants] Views preserve addresses; only materialization changes storage.
  Strides and offsets count elements, independently of the payload dtype.
[Timeouts & Locks] O(rank) metadata operations, no device contexts or locks.
[Side-Effect Policy] Immutable metadata only; never allocates tensor payloads.
"""
from dataclasses import dataclass, replace
from math import prod
from typing import Any

#: How a value obtains its execution-time binding.  ``static`` means the
#: declaration is already complete; the other kinds name the symbol, event or
#: external input that supplies the value.  A dynamic binding is never silently
#: replaced by a default.
TENSOR_BINDINGS = frozenset({"static", "symbolic", "event", "external"})


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int = 0
    storage_shape: tuple[int, ...] = ()

    @classmethod
    def contiguous(cls, shape):
        shape = tuple(shape)
        stride, strides = 1, []
        for size in reversed(shape):
            strides.append(stride)
            stride *= max(1, size)
        return cls(shape, tuple(reversed(strides)), storage_shape=shape)

    @property
    def elements(self):
        return prod(self.shape)

    @property
    def is_contiguous(self):
        if not self.elements:
            return True
        stride = 1
        for size, actual in zip(reversed(self.shape), reversed(self.strides)):
            if size != 1 and actual != stride:
                return False
            stride *= size
        return True

    def narrow(self, axis, start, length):
        axis %= len(self.shape)
        if start < 0 or length < 0 or start + length > self.shape[axis]:
            raise ValueError("tensor_slice_exceeds_dimension")
        shape = list(self.shape)
        shape[axis] = length
        return replace(self, shape=tuple(shape), offset=self.offset + start * self.strides[axis])

    def permute(self, axes):
        axes = tuple(axes)
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError("tensor_permutation_requires_each_axis")
        return replace(self, shape=tuple(self.shape[i] for i in axes),
                       strides=tuple(self.strides[i] for i in axes))

    def reshape_view(self, shape):
        """Return the same address sequence in a new shape, or None for a copy."""
        shape = tuple(shape)
        if any(size < 0 for size in shape) or prod(shape) != self.elements:
            raise ValueError("reshape_requires_same_element_count")
        if not self.elements:
            return replace(TensorLayout.contiguous(shape), offset=self.offset,
                           storage_shape=self.storage_shape)
        # A contiguous chunk can be split into arbitrary dimensions. Crossing
        # a gap between chunks would change addresses and requires a copy.
        chunks = []
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size == 1:
                continue
            if chunks and stride == chunks[-1][0] * chunks[-1][1]:
                chunks[-1] = chunks[-1][0] * size, chunks[-1][1]
            else:
                chunks.append((size, stride))
        index, strides = len(shape) - 1, [0] * len(shape)
        for count, stride in chunks or [(1, 1)]:
            consumed = 1
            while index >= 0 and (consumed < count or shape[index] == 1):
                strides[index] = consumed * stride
                consumed *= shape[index]
                index -= 1
            if consumed != count:
                return None
        if index >= 0:
            return None
        return replace(self, shape=shape, strides=tuple(strides))

    def to_dict(self):
        # Direct projection: ``dataclasses.asdict`` recurses through every
        # declared field (53k calls for one 92-layer model) only to copy tuple
        # values verbatim.  The layout is two tuples and two integers.
        return {"shape": self.shape, "strides": self.strides,
                "offset": self.offset, "storage_shape": self.storage_shape}


@dataclass(frozen=True)
class TensorValue:
    """One typed logical value with its own representation and binding.

    ``dtype`` describes the payload.  A quantisation scheme, physical layout
    family, storage identity, state version and execution binding are separate
    fields: a model-level activation dtype must not stand in for all of them.
    ``alias_of`` and ``storage`` are identities, not allocations; buffer
    lifetime belongs to the program's allocation effects.
    """

    producer: str
    dtype: str
    layout: TensorLayout
    output: str | int = 0
    #: Payload quantisation scheme (e.g. ``w8a8``/``mxfp4``).  Scale and
    #: zero-point are independent associated values, never folded in here.
    quant: str = ""
    #: Storage layout family (``strided``/``nz``/``fractal``), distinct from
    #: the element strides carried by ``layout``.
    layout_kind: str = "strided"
    #: Physical buffer identity when the value is materialised; empty for an
    #: abstract value that has no allocation at this layer.
    storage: str = ""
    #: Logical value identity this value aliases; empty means no alias.
    alias_of: str = ""
    #: State generation this value reads or writes, when it is state.
    state_version: int | None = None
    #: ``static``/``symbolic``/``event``/``external`` binding source.
    binding: str = "static"
    #: Symbol, event or external input that supplies a dynamic binding.
    binding_name: str = ""

    def __post_init__(self):
        if not isinstance(self.producer, str):
            raise TypeError("tensor_value_producer_requires_string")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("tensor_value_dtype_requires_nonempty_string")
        if not isinstance(self.layout, TensorLayout):
            raise TypeError("tensor_value_layout_requires_tensor_layout")
        if not isinstance(self.output, (str, int)) or isinstance(self.output, bool):
            raise TypeError("tensor_value_output_requires_string_or_int")
        for name in ("quant", "layout_kind", "storage", "alias_of", "binding",
                     "binding_name"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"tensor_value_{name}_requires_string")
        if not self.layout_kind:
            raise ValueError("tensor_value_layout_kind_requires_nonempty_string")
        if self.binding not in TENSOR_BINDINGS:
            raise ValueError(f"tensor_value_binding_unknown:{self.binding}")
        if self.binding != "static" and not self.binding_name:
            raise ValueError(f"tensor_value_binding_name_required:{self.binding}")
        if self.state_version is not None and (
                not isinstance(self.state_version, int)
                or isinstance(self.state_version, bool)
                or self.state_version < 0):
            raise ValueError("tensor_value_state_version_requires_nonnegative_int")

    def narrow(self, axis, start, length):
        return replace(self, layout=self.layout.narrow(axis, start, length))

    def permute(self, axes):
        return replace(self, layout=self.layout.permute(axes))

    def with_binding(self, binding: str, name: str = "",
                     *, state_version: int | None = None) -> "TensorValue":
        """Return the same representation bound to another execution source."""
        return replace(self, binding=binding, binding_name=name,
                       state_version=(self.state_version if state_version is None
                                      else state_version))

    @property
    def key(self) -> tuple:
        return (self.producer, self.dtype, self.layout, self.output, self.quant,
                self.layout_kind, self.storage, self.alias_of,
                self.state_version, self.binding, self.binding_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer, "dtype": self.dtype,
            "layout": self.layout.to_dict(), "output": self.output,
            "quant": self.quant, "layout_kind": self.layout_kind,
            "storage": self.storage, "alias_of": self.alias_of,
            "state_version": self.state_version, "binding": self.binding,
            "binding_name": self.binding_name,
        }


__all__ = ["TENSOR_BINDINGS", "TensorLayout", "TensorValue"]
