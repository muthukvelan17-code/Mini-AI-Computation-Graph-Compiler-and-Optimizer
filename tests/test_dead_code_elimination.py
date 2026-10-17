"""Tests for Dead Code Elimination (DCE) optimization pass."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from optimizations.dead_code_elimination import DeadCodeEliminationPass


def test_dead_code_elimination():
    g = ComputationGraph()
    a = g.add_input("a", shape=(2, 2))
    b = g.add_input("b", shape=(2, 2))
    c = g.add_input("c", shape=(2, 2))

    # Live computation: a + b
    live_out = g.add(a.id, b.id, name="live")
    g.mark_output(live_out.id)

    # Dead path: (a * c) -> sigmoid
    dead_mul = g.mul(a.id, c.id, name="dead_mul")
    dead_sig = g.sigmoid(dead_mul.id, name="dead_sig")

    assert len(g.nodes) == 6

    dce = DeadCodeEliminationPass()
    result = dce.run(g)
    opt_g = result.optimized_graph

    # Dead nodes should be pruned
    assert not opt_g.has_node(dead_mul.id)
    assert not opt_g.has_node(dead_sig.id)
    assert not opt_g.has_node(c.id)  # Unused input node is also dead
    assert opt_g.has_node(a.id)
    assert opt_g.has_node(b.id)
    assert opt_g.has_node(live_out.id)
    assert len(opt_g.nodes) == 3

    # Verify numerical correctness
    inp = {a.id: np.ones((2, 2), dtype=np.float32), b.id: np.ones((2, 2), dtype=np.float32)}
    orig_val = GraphExecutor(g).execute({**inp, c.id: np.zeros((2, 2), dtype=np.float32)}).outputs[live_out.id]
    opt_val = GraphExecutor(opt_g).execute(inp).outputs[live_out.id]
    np.testing.assert_allclose(orig_val, opt_val)
