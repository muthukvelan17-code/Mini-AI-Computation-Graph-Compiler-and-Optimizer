"""Tests for NumPy execution engine."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor, ExecutionError


def test_basic_arithmetic_execution():
    g = ComputationGraph()
    a = g.add_input("a", shape=(2, 2))
    b = g.add_input("b", shape=(2, 2))
    add_node = g.add(a.id, b.id)
    sub_node = g.sub(add_node.id, a.id)
    g.mark_output(sub_node.id)

    arr_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    executor = GraphExecutor(g)
    result = executor.execute({a.id: arr_a, b.id: arr_b})

    # (a + b) - a == b
    np.testing.assert_allclose(result.outputs[sub_node.id], arr_b, atol=1e-6)
    assert sub_node.id in result.output_shapes


def test_matmul_and_activation_execution():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 3))
    w = g.add_input("w", shape=(3, 2))
    mm = g.matmul(x.id, w.id)
    relu_out = g.relu(mm.id)
    sig_out = g.sigmoid(mm.id)
    g.mark_output(relu_out.id)
    g.mark_output(sig_out.id)

    arr_x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    arr_w = np.array([[1.0, -1.0], [0.5, 2.0], [-0.5, 0.5]], dtype=np.float32)

    executor = GraphExecutor(g)
    res = executor.execute({x.id: arr_x, w.id: arr_w})

    expected_mm = np.matmul(arr_x, arr_w)
    expected_relu = np.maximum(0.0, expected_mm)
    expected_sig = 1.0 / (1.0 + np.exp(-expected_mm))

    np.testing.assert_allclose(res.outputs[relu_out.id], expected_relu, atol=1e-5)
    np.testing.assert_allclose(res.outputs[sig_out.id], expected_sig, atol=1e-5)


def test_fused_operators_execution():
    g = ComputationGraph()
    x = g.add_input("x", shape=(4, 4))
    w = g.add_input("w", shape=(4, 4))
    b = g.add_input("b", shape=(4,))

    fused = g.add_op("fused_matmul_add_relu", [x.id, w.id, b.id])
    g.mark_output(fused.id)

    arr_x = np.random.randn(4, 4).astype(np.float32)
    arr_w = np.random.randn(4, 4).astype(np.float32)
    arr_b = np.random.randn(4).astype(np.float32)

    executor = GraphExecutor(g)
    res = executor.execute({x.id: arr_x, w.id: arr_w, b.id: arr_b})

    expected = np.maximum(0.0, np.matmul(arr_x, arr_w) + arr_b)
    np.testing.assert_allclose(res.outputs[fused.id], expected, atol=1e-5)


def test_input_immutability():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2))
    c = g.add_constant(np.ones((2, 2), dtype=np.float32))
    out = g.add(x.id, c.id)
    g.mark_output(out.id)

    original_input = np.zeros((2, 2), dtype=np.float32)
    input_copy = original_input.copy()

    executor = GraphExecutor(g)
    _ = executor.execute({x.id: original_input})

    # The caller's input array must not be altered
    np.testing.assert_array_equal(original_input, input_copy)


def test_missing_input_raises_error():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.relu(x.id)
    g.mark_output(out.id)

    executor = GraphExecutor(g)
    with pytest.raises(ExecutionError, match="Missing required input"):
        executor.execute({})
