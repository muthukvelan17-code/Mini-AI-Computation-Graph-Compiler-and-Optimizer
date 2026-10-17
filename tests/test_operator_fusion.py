"""Tests for Operator Fusion optimization pass."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from optimizations.operator_fusion import OperatorFusionPass


def test_fuse_mul_add():
    g = ComputationGraph()
    a = g.add_input("a", shape=(4, 4))
    b = g.add_input("b", shape=(4, 4))
    c = g.add_input("c", shape=(4, 4))

    # a * b + c
    mul_op = g.mul(a.id, b.id, name="mul")
    add_op = g.add(mul_op.id, c.id, name="add")
    g.mark_output(add_op.id)

    fusion = OperatorFusionPass()
    res = fusion.run(g)
    opt_g = res.optimized_graph

    assert len(res.nodes_fused) == 1
    fused_id = res.nodes_fused[0][0]
    assert opt_g.get_node(fused_id).op_type == "fused_mul_add"
    assert opt_g.outputs == [fused_id]

    # Verify numerical correctness
    inp = {
        a.id: np.random.randn(4, 4).astype(np.float32),
        b.id: np.random.randn(4, 4).astype(np.float32),
        c.id: np.random.randn(4, 4).astype(np.float32),
    }
    orig_val = GraphExecutor(g).execute(inp).outputs[add_op.id]
    opt_val = GraphExecutor(opt_g).execute(inp).outputs[fused_id]
    np.testing.assert_allclose(orig_val, opt_val, atol=1e-5)


def test_fuse_matmul_add_relu():
    g = ComputationGraph()
    x = g.add_input("x", shape=(8, 16))
    w = g.add_input("w", shape=(16, 32))
    b = g.add_input("b", shape=(32,))

    mm = g.matmul(x.id, w.id, name="mm")
    add_b = g.add(mm.id, b.id, name="add_b")
    relu_out = g.relu(add_b.id, name="relu_out")
    g.mark_output(relu_out.id)

    fusion = OperatorFusionPass()
    res = fusion.run(g)
    opt_g = res.optimized_graph

    # Should fuse into FusedLinearReLU
    fused_out = opt_g.get_node(opt_g.outputs[0])
    assert fused_out.op_type == "fused_matmul_add_relu"

    # Verify numerical correctness
    inp = {
        x.id: np.random.randn(8, 16).astype(np.float32),
        w.id: np.random.randn(16, 32).astype(np.float32),
        b.id: np.random.randn(32).astype(np.float32),
    }
    orig_val = GraphExecutor(g).execute(inp).outputs[relu_out.id]
    opt_val = GraphExecutor(opt_g).execute(inp).outputs[fused_out.id]
    np.testing.assert_allclose(orig_val, opt_val, atol=1e-5)
