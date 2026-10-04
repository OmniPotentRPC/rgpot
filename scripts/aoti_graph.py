"""Canonical tensor slices for AOTInductor graph lowering."""

from __future__ import annotations

import torch


def canonicalize_strided_slices(program):
    """Express single-axis strided writes with the functional slice operator.

    For a contiguous base, a view with the base strides and offset
    ``start * stride[axis]`` addresses the same elements as that axis slice.
    A functional slice write preserves every other element of the base.
    This representation keeps snapshots of the base independent of writes
    when the compiler lowers the graph.
    """
    program = program.run_decompositions({})
    for node in program.graph_module.graph.nodes:
        if node.target != torch.ops.aten.as_strided_scatter.default:
            continue
        if len(node.args) < 4:
            continue
        base, values, size, stride, *tail = node.args
        if not isinstance(base, torch.fx.Node):
            continue
        tensor = base.meta.get("val")
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
        offset = tail[0] if tail else node.kwargs.get("storage_offset")
        if offset is None:
            offset = tensor.storage_offset()
        dimensions = (*shape, *strides, *size, *stride, offset,
                      tensor.storage_offset())
        if not all(isinstance(value, int) for value in dimensions):
            continue
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            continue
        if len(size) != len(shape) or tuple(stride) != strides:
            continue
        axes = [axis for axis, (extent, original) in
                enumerate(zip(size, shape)) if extent != original]
        if len(axes) != 1:
            continue
        axis = axes[0]
        if strides[axis] <= 0 or offset % strides[axis] != 0:
            continue
        start = offset // strides[axis]
        end = start + size[axis]
        if not 0 <= start <= end <= shape[axis]:
            continue
        node.target = torch.ops.aten.slice_scatter.default
        node.args = (base, values, axis, start, end, 1)
        node.kwargs = {}
    program.graph_module.graph.lint()
    program.graph_module.recompile()
    program.validate()
    return program
