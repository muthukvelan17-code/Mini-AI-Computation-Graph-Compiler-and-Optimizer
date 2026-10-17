"""Tests for shape inference engine and broadcasting validation."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.shape_inference import infer_shapes, broadcast_shapes, ShapeInferenceError


def test_broadcasting_rules():
    assert broadcast_shapes((3, 1), (1, 4)) == (3, 4)
    assert broadcast_shapes((2, 3, 4), (4,)) == (2, 3, 4)
    assert broadcast_shapes((5, 1, 2), (1, 3, 2)) == (5, 3, 2)

    with pytest.raises(ShapeInferenceError):
        broadcast_shapes((4, 3), (4, 2))


def test_matmul_shape_inference():
    g = ComputationGraph()
    x = g.add_input("x", shape=(8, 16))
    w = g.add_input("w", shape=(16, 32))
    out = g.matmul(x.id, w.id)
    g.mark_output(out.id)

    shapes = infer_shapes(g)
    assert shapes[out.id] == (8, 32)


def test_matmul_incompatible_dimensions():
    g = ComputationGraph()
    x = g.add_input("x", shape=(8, 16))
    w = g.add_input("w", shape=(15, 32))  # Inner mismatch 16 != 15
    out = g.matmul(x.id, w.id)
    g.mark_output(out.id)

    with pytest.raises(ShapeInferenceError, match="inner dimensions mismatch"):
        infer_shapes(g)


def test_fused_ops_shape_inference():
    g = ComputationGraph()
    x = g.add_input("x", shape=(10, 20))
    w = g.add_input("w", shape=(20, 5))
    b = g.add_constant(np.zeros((5,), dtype=np.float32), name="b")
    fused = g.add_op("fused_matmul_add_relu", [x.id, w.id, b.id])
    g.mark_output(fused.id)

    shapes = infer_shapes(g)
    assert shapes[fused.id] == (10, 5)
