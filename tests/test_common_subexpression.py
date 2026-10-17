"""Tests for Common Subexpression Elimination (CSE) optimization pass."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from optimizations.common_subexpression import CommonSubexpressionEliminationPass


def test_common_subexpression_elimination():
    g = ComputationGraph()
    a = g.add_input("a", shape=(3, 3))
    b = g.add_input("b", shape=(3, 3))

    # mul1 = a * b, mul2 = a * b
    mul1 = g.mul(a.id, b.id, name="mul1")
    mul2 = g.mul(a.id, b.id, name="mul2")
    res = g.add(mul1.id, mul2.id, name="res")
    g.mark_output(res.id)

    assert len(g.nodes) == 5

    cse = CommonSubexpressionEliminationPass()
    result = cse.run(g)
    opt_g = result.optimized_graph

    # Duplicate mul2 should be eliminated; res now takes [mul1, mul1]
    assert len(opt_g.nodes) == 4
    assert not opt_g.has_node(mul2.id)
    assert opt_g.get_node(res.id).inputs == [mul1.id, mul1.id]

    # Verify numerical correctness
    inp = {a.id: np.random.randn(3, 3).astype(np.float32), b.id: np.random.randn(3, 3).astype(np.float32)}
    orig_val = GraphExecutor(g).execute(inp).outputs[res.id]
    opt_val = GraphExecutor(opt_g).execute(inp).outputs[res.id]
    np.testing.assert_allclose(orig_val, opt_val, atol=1e-6)


def test_commutative_cse():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2))
    y = g.add_input("y", shape=(2, 2))

    # add1 = x + y, add2 = y + x (commutative equality)
    add1 = g.add(x.id, y.id, name="add1")
    add2 = g.add(y.id, x.id, name="add2")
    res = g.mul(add1.id, add2.id, name="res")
    g.mark_output(res.id)

    cse = CommonSubexpressionEliminationPass()
    result = cse.run(g)
    opt_g = result.optimized_graph

    assert len(opt_g.nodes) == 4
    assert not opt_g.has_node(add2.id)
