"""Tests for Constant Folding optimization pass."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from optimizations.constant_folding import ConstantFoldingPass


def test_constant_folding_simple():
    g = ComputationGraph()
    c2 = g.add_constant(np.array(2.0, dtype=np.float32), name="c2")
    c3 = g.add_constant(np.array(3.0, dtype=np.float32), name="c3")
    x = g.add_input("x", shape=(3, 3))

    add_op = g.add(c2.id, c3.id, name="add_2_3")
    out = g.mul(add_op.id, x.id, name="out")
    g.mark_output(out.id)

    # Run pass
    cf = ConstantFoldingPass()
    result = cf.run(g)
    opt_g = result.optimized_graph

    # The add_op node should now be folded into a CONSTANT with value 5.0
    folded_node = opt_g.get_node(add_op.id)
    assert folded_node.is_constant
    assert folded_node.data == 5.0

    # Verify numerical correctness
    inp = np.ones((3, 3), dtype=np.float32)
    orig_res = GraphExecutor(g).execute({x.id: inp}).outputs[out.id]
    opt_res = GraphExecutor(opt_g).execute({x.id: inp}).outputs[out.id]
    np.testing.assert_allclose(orig_res, opt_res, atol=1e-6)
    np.testing.assert_allclose(opt_res, 5.0 * np.ones((3, 3), dtype=np.float32))


def test_multi_step_constant_folding():
    g = ComputationGraph()
    c1 = g.add_constant(np.array(10.0, dtype=np.float32))
    c2 = g.add_constant(np.array(5.0, dtype=np.float32))
    c3 = g.add_constant(np.array(2.0, dtype=np.float32))

    # (10 - 5) * 2 = 10
    sub_node = g.sub(c1.id, c2.id)
    mul_node = g.mul(sub_node.id, c3.id)
    g.mark_output(mul_node.id)

    cf = ConstantFoldingPass()
    res = cf.run(g)
    opt_g = res.optimized_graph

    assert opt_g.get_node(mul_node.id).is_constant
    assert opt_g.get_node(mul_node.id).data == 10.0
