"""Strided tensor views without storage allocation.

[Role] Describe address maps and decide whether a reshape can share storage.
[Invariants] Views preserve addresses; only materialization changes storage.
  Strides and offsets count elements, independently of the payload dtype.
[Timeouts & Locks] O(rank) metadata operations, no device contexts or locks.
[Side-Effect Policy] Immutable metadata only; never allocates tensor payloads.
"""
from dataclasses import asdict, dataclass, replace
from math import prod


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
        return asdict(self)


@dataclass(frozen=True)
class TensorValue:
    producer: str
    dtype: str
    layout: TensorLayout
    output: str | int = 0

    def narrow(self, axis, start, length):
        return replace(self, layout=self.layout.narrow(axis, start, length))

    def permute(self, axes):
        return replace(self, layout=self.layout.permute(axes))
