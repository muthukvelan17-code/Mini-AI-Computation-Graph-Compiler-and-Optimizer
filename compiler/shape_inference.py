"""Shape inference engine for the computation graph."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from compiler.node import Node, OpType
from compiler.graph import ComputationGraph, GraphError


class ShapeInferenceError(GraphError):
    """Raised when tensor shapes are incompatible for a given operation."""
    pass


def broadcast_shapes(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate the broadcasted output shape for two input shapes according to NumPy rules."""
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_rank = max(rank_a, rank_b)

    # Pad with leading 1s
    padded_a = (1,) * (max_rank - rank_a) + shape_a
    padded_b = (1,) * (max_rank - rank_b) + shape_b

    result: List[int] = []
    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            raise ShapeInferenceError(
                f"Tensors with shapes {shape_a} and {shape_b} cannot be broadcast together "
                f"due to mismatch at dimensions {dim_a} and {dim_b}."
            )
    return tuple(result)


def infer_node_shape(node: Node, input_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Infer the output shape of a single node given its input shapes."""
    op = node.op_type
    op_str = op.value if isinstance(op, OpType) else str(op)

    if op == OpType.INPUT:
        if node.shape is not None:
            return node.shape
        if node.data is not None:
            return node.data.shape
        raise ShapeInferenceError(f"Input node {node.id!r} has no defined shape or data.")

    if op == OpType.CONSTANT:
        if node.data is not None:
            return node.data.shape
        if node.shape is not None:
            return node.shape
        raise ShapeInferenceError(f"Constant node {node.id!r} has no data array or shape.")

    if op in (OpType.RELU, OpType.SIGMOID):
        if len(input_shapes) != 1:
            raise ShapeInferenceError(f"Unary op {op_str!r} for node {node.id!r} expects 1 input, got {len(input_shapes)}.")
        return input_shapes[0]

    if op == OpType.SCALAR_MUL:
        if len(input_shapes) != 1:
            raise ShapeInferenceError(f"Scalar multiplication node {node.id!r} expects 1 tensor input, got {len(input_shapes)}.")
        return input_shapes[0]

    if op in (OpType.ADD, OpType.SUB, OpType.MUL):
        if len(input_shapes) != 2:
            raise ShapeInferenceError(f"Binary element-wise op {op_str!r} on node {node.id!r} expects 2 inputs, got {len(input_shapes)}.")
        return broadcast_shapes(input_shapes[0], input_shapes[1])

    if op == OpType.MATMUL:
        if len(input_shapes) != 2:
            raise ShapeInferenceError(f"MatMul node {node.id!r} expects 2 inputs, got {len(input_shapes)}.")
        shape_a, shape_b = input_shapes[0], input_shapes[1]
        if len(shape_a) < 2 or len(shape_b) < 2:
            raise ShapeInferenceError(
                f"MatMul node {node.id!r} requires tensors of rank >= 2. Got shapes {shape_a} and {shape_b}."
            )
        if shape_a[-1] != shape_b[-2]:
            raise ShapeInferenceError(
                f"Cannot multiply matrix {node.inputs[0]!r} {shape_a} by {node.inputs[1]!r} {shape_b}: "
                f"inner dimensions mismatch ({shape_a[-1]} != {shape_b[-2]})."
            )
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        broadcast_batch = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return broadcast_batch + (shape_a[-2], shape_b[-1])

    if op == OpType.FUSED_MUL_ADD:
        if len(input_shapes) != 3:
            raise ShapeInferenceError(f"Fused Mul-Add node {node.id!r} expects 3 inputs (a, b, c), got {len(input_shapes)}.")
        mul_shape = broadcast_shapes(input_shapes[0], input_shapes[1])
        return broadcast_shapes(mul_shape, input_shapes[2])

    if op in (OpType.FUSED_MATMUL_ADD, OpType.FUSED_MATMUL_ADD_RELU):
        if len(input_shapes) != 3:
            raise ShapeInferenceError(f"Fused op {op_str!r} on node {node.id!r} expects 3 inputs (X, W, b), got {len(input_shapes)}.")
        shape_x, shape_w, shape_b = input_shapes[0], input_shapes[1], input_shapes[2]
        if len(shape_x) < 2 or len(shape_w) < 2:
            raise ShapeInferenceError(
                f"Fused Matmul node {node.id!r} requires X and W to have rank >= 2. Got {shape_x} and {shape_w}."
            )
        if shape_x[-1] != shape_w[-2]:
            raise ShapeInferenceError(
                f"Fused Matmul inner dimensions mismatch on node {node.id!r}: {shape_x[-1]} != {shape_w[-2]}."
            )
        batch_x = shape_x[:-2]
        batch_w = shape_w[:-2]
        broadcast_batch = broadcast_shapes(batch_x, batch_w) if (batch_x or batch_w) else ()
        mm_shape = broadcast_batch + (shape_x[-2], shape_w[-1])
        return broadcast_shapes(mm_shape, shape_b)

    if op == OpType.FUSED_ADD_RELU:
        if len(input_shapes) != 2:
            raise ShapeInferenceError(f"Fused Add-ReLU node {node.id!r} expects 2 inputs, got {len(input_shapes)}.")
        return broadcast_shapes(input_shapes[0], input_shapes[1])

    # Fallback for custom or unchanged shape
    if node.shape is not None:
        return node.shape
    raise ShapeInferenceError(f"Cannot infer shape for unknown operation type {op_str!r} on node {node.id!r}.")


def infer_shapes(graph: ComputationGraph) -> Dict[str, Tuple[int, ...]]:
    """Traverse graph in topological order and infer output shapes for all nodes.

    Updates each node's `shape` attribute and returns a dict mapping node_id -> shape.
    """
    order = graph.topological_sort()
    inferred: Dict[str, Tuple[int, ...]] = {}

    for nid in order:
        node = graph.get_node(nid)
        parent_shapes = [inferred[inp] for inp in node.inputs]
        try:
            node_shape = infer_node_shape(node, parent_shapes)
            node.shape = node_shape
            inferred[nid] = node_shape
        except ShapeInferenceError as e:
            raise ShapeInferenceError(f"Shape inference failed at node {nid!r} ({node.name}): {e}") from e

    return inferred
