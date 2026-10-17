"""Tests for ComputationGraph construction, topological sort, cycle detection, and cloning."""

import pytest
import numpy as np

from compiler.graph import ComputationGraph, CycleDetectedError, GraphValidationError
from compiler.node import Node, OpType
from compiler.serialization import serialize_graph, deserialize_graph


def test_graph_construction():
    g = ComputationGraph(name="test_graph")
    x = g.add_input("x", shape=(4, 4))
    w = g.add_input("w", shape=(4, 2))
    b = g.add_constant(np.zeros((2,), dtype=np.float32), name="b")

    mm = g.matmul(x.id, w.id, name="mm")
    out = g.add(mm.id, b.id, name="out")
    g.mark_output(out.id)

    assert len(g.nodes) == 5
    assert g.outputs == [out.id]
    assert g.has_node(x.id)
    assert g.get_node(out.id).inputs == [mm.id, b.id]


def test_topological_sort_order():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2))
    c = g.add_constant(np.ones((2, 2), dtype=np.float32), name="c")
    m = g.mul(x.id, c.id, name="m")
    a = g.add(m.id, x.id, name="a")
    g.mark_output(a.id)

    order = g.topological_sort()
    assert order.index(x.id) < order.index(m.id)
    assert order.index(c.id) < order.index(m.id)
    assert order.index(m.id) < order.index(a.id)
    assert order.index(x.id) < order.index(a.id)


def test_cycle_detection():
    g = ComputationGraph()
    # Create cycle: node1 -> node2 -> node1
    n1 = Node(node_id="n1", op_type=OpType.RELU, inputs=["n2"])
    n2 = Node(node_id="n2", op_type=OpType.RELU, inputs=["n1"])
    g.add_node(n1)
    g.add_node(n2)
    g.mark_output("n1")

    with pytest.raises(CycleDetectedError):
        g.topological_sort()


def test_graph_validation():
    g = ComputationGraph()
    with pytest.raises(GraphValidationError, match="no nodes"):
        g.validate()

    g.add_input("x")
    with pytest.raises(GraphValidationError, match="no marked output"):
        g.validate()

    # Invalid input reference
    g2 = ComputationGraph()
    n = Node(node_id="n1", op_type=OpType.RELU, inputs=["nonexistent"])
    g2.add_node(n)
    g2.mark_output("n1")
    with pytest.raises(GraphValidationError, match="nonexistent input"):
        g2.validate()


def test_graph_clone():
    g = ComputationGraph(name="orig")
    x = g.add_input("x", shape=(2, 2))
    c = g.add_constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="c")
    out = g.add(x.id, c.id, name="out")
    g.mark_output(out.id)

    cloned = g.clone()
    assert len(cloned.nodes) == len(g.nodes)
    assert cloned.outputs == g.outputs
    # Mutating cloned graph must NOT affect original
    cloned.remove_node(c.id)
    assert not cloned.has_node(c.id)
    assert g.has_node(c.id)


def test_serialization_roundtrip():
    g = ComputationGraph(name="roundtrip")
    x = g.add_input("x", shape=(3, 3))
    c = g.add_constant(np.eye(3, dtype=np.float32), name="eye")
    out = g.matmul(x.id, c.id, name="res")
    g.mark_output(out.id)

    json_str = serialize_graph(g)
    deserialized = deserialize_graph(json_str)

    assert deserialized.name == "roundtrip"
    assert len(deserialized.nodes) == 3
    assert deserialized.outputs == [out.id]
    np.testing.assert_array_equal(deserialized.get_node(c.id).data, np.eye(3, dtype=np.float32))
