"""Tests for Algebraic Simplification optimization pass."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from optimizations.algebraic_simplification import AlgebraicSimplificationPass
from optimizations.dead_code_elimination import DeadCodeEliminationPass


def test_algebraic_simplification_add_zero_and_mul_one():
    g = ComputationGraph()
    x = g.add_input("x", shape=(3, 3))
    c1 = g.add_constant(np.array(1.0, dtype=np.float32), name="one")
    c0 = g.add_constant(np.array(0.0, dtype=np.float32), name="zero")

    # (x * 1) + 0 -> x
    mul_one = g.mul(x.id, c1.id, name="mul_one")
    add_zero = g.add(mul_one.id, c0.id, name="add_zero")
    g.mark_output(add_zero.id)

    alg = AlgebraicSimplificationPass()
    result = alg.run(g)
    opt_g = result.optimized_graph

    # Prune unused constants via DCE
    opt_g = DeadCodeEliminationPass().run(opt_g).optimized_graph

    # Both identity operations simplified: output should now directly be x!
    assert opt_g.outputs == [x.id]
    assert len(opt_g.nodes) == 1

    # Verify numerical correctness
    inp = {x.id: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)}
    orig_val = GraphExecutor(g).execute(inp).outputs[add_zero.id]
    opt_val = GraphExecutor(opt_g).execute(inp).outputs[x.id]
    np.testing.assert_allclose(orig_val, opt_val)


def test_scalar_mul_simplification():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2))
    sc = g.scalar_mul(x.id, 1.0, name="sc1")
    g.mark_output(sc.id)

    alg = AlgebraicSimplificationPass()
    res = alg.run(g)
    opt_g = res.optimized_graph

    assert opt_g.outputs == [x.id]
