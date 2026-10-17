"""ComputationGraph DAG representation and manipulation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import networkx as nx

from compiler.node import Node, OpType


class GraphError(Exception):
    """Base exception for graph operations."""
    pass


class CycleDetectedError(GraphError):
    """Raised when a directed cycle is detected in the computation graph."""
    pass


class GraphValidationError(GraphError):
    """Raised when graph validation fails."""
    pass


class ComputationGraph:
    """Directed Acyclic Graph (DAG) representing tensor computations.

    Preserves node definitions, dependencies, outputs, and provides graph manipulation,
    topological sorting, cycle detection, validation, and cloning.
    """

    def __init__(self, name: str = "computation_graph"):
        self.name: str = name
        self.nodes: Dict[str, Node] = {}
        self.outputs: List[str] = []
        self._next_id: int = 0

    def _generate_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        while True:
            candidate = f"{prefix}_{self._next_id}"
            self._next_id += 1
            if candidate not in self.nodes:
                return candidate

    def add_node(self, node: Node) -> Node:
        """Add an existing Node to the graph."""
        if node.id in self.nodes:
            raise GraphError(f"Node with id {node.id!r} already exists in graph.")
        self.nodes[node.id] = node
        return node

    def add_input(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        data: Optional[Union[np.ndarray, List, float, int]] = None,
        node_id: Optional[str] = None,
    ) -> Node:
        """Add an Input tensor node to the graph."""
        nid = node_id or (name if name and name not in self.nodes else self._generate_id("input"))
        node = Node(
            node_id=nid,
            op_type=OpType.INPUT,
            inputs=[],
            data=data,
            shape=shape,
            dtype=dtype,
            name=name or nid,
        )
        return self.add_node(node)

    def add_constant(
        self,
        data: Union[np.ndarray, List, float, int],
        name: Optional[str] = None,
        dtype: str = "float32",
        node_id: Optional[str] = None,
    ) -> Node:
        """Add a Constant tensor node to the graph."""
        nid = node_id or (name if name and name not in self.nodes else self._generate_id("const"))
        node = Node(
            node_id=nid,
            op_type=OpType.CONSTANT,
            inputs=[],
            data=data,
            dtype=dtype,
            name=name or nid,
        )
        return self.add_node(node)

    def add_op(
        self,
        op_type: Union[OpType, str],
        inputs: List[str],
        name: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
    ) -> Node:
        """Add an operation node to the graph."""
        op_enum = OpType(op_type) if isinstance(op_type, str) and OpType.is_valid(op_type) else op_type
        prefix = op_enum.value if isinstance(op_enum, OpType) else str(op_enum)
        nid = node_id or (name if name and name not in self.nodes else self._generate_id(prefix))
        
        # Verify input references exist
        for inp in inputs:
            if inp not in self.nodes:
                raise GraphError(f"Input node {inp!r} not found in graph while adding {nid!r}.")

        node = Node(
            node_id=nid,
            op_type=op_enum,
            inputs=inputs,
            shape=shape,
            dtype=dtype,
            metadata=metadata or {},
            name=name or nid,
        )
        return self.add_node(node)

    # Convenience helper methods
    def add(self, a: str, b: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.ADD, [a, b], name=name)

    def sub(self, a: str, b: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.SUB, [a, b], name=name)

    def mul(self, a: str, b: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.MUL, [a, b], name=name)

    def matmul(self, a: str, b: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.MATMUL, [a, b], name=name)

    def relu(self, x: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.RELU, [x], name=name)

    def sigmoid(self, x: str, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.SIGMOID, [x], name=name)

    def scalar_mul(self, x: str, scalar: float, name: Optional[str] = None) -> Node:
        return self.add_op(OpType.SCALAR_MUL, [x], name=name, metadata={"scalar": float(scalar)})

    def mark_output(self, node_id: str) -> None:
        """Mark a node as a terminal output of the computation graph."""
        if node_id not in self.nodes:
            raise GraphError(f"Cannot mark nonexistent node {node_id!r} as output.")
        if node_id not in self.outputs:
            self.outputs.append(node_id)

    def set_outputs(self, node_ids: List[str]) -> None:
        """Set the complete list of terminal output nodes."""
        for nid in node_ids:
            if nid not in self.nodes:
                raise GraphError(f"Cannot mark nonexistent node {nid!r} as output.")
        self.outputs = list(node_ids)

    def get_node(self, node_id: str) -> Node:
        """Retrieve node by ID."""
        if node_id not in self.nodes:
            raise GraphError(f"Node {node_id!r} not found in computation graph.")
        return self.nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def remove_node(self, node_id: str) -> None:
        """Safely remove a node from the graph."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.outputs:
            self.outputs.remove(node_id)

    def get_parents(self, node_id: str) -> List[Node]:
        """Return parent nodes (direct inputs to this node)."""
        node = self.get_node(node_id)
        return [self.nodes[inp] for inp in node.inputs if inp in self.nodes]

    def get_children(self, node_id: str) -> List[Node]:
        """Return child nodes (direct consumers of this node's output)."""
        return [n for n in self.nodes.values() if node_id in n.inputs]

    def get_input_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.is_input]

    def get_constant_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.is_constant]

    def replace_node_references(self, old_id: str, new_id: str) -> None:
        """Rewire all consumers of old_id to consume new_id instead."""
        for node in self.nodes.values():
            node.inputs = [new_id if inp == old_id else inp for inp in node.inputs]
        # Also update output references if needed
        self.outputs = [new_id if out_id == old_id else out_id for out_id in self.outputs]

    def topological_sort(self) -> List[str]:
        """Return a list of node IDs in topological execution order.

        Uses Kahn's algorithm and detects cycles with precise reporting.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for inp in node.inputs:
                if inp in self.nodes:
                    adj[inp].append(nid)
                    in_degree[nid] += 1

        queue: List[str] = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) < len(self.nodes):
            unvisited = [nid for nid, deg in in_degree.items() if deg > 0]
            raise CycleDetectedError(f"Cycle detected in computation graph! Nodes in cycle or unreachable: {unvisited}")

        return order

    def validate(self) -> None:
        """Validate structural integrity of the computation graph.

        Raises GraphValidationError if any inconsistency is detected.
        """
        if not self.nodes:
            raise GraphValidationError("Graph contains no nodes.")

        # Check that all inputs exist
        for nid, node in self.nodes.items():
            for inp in node.inputs:
                if inp not in self.nodes:
                    raise GraphValidationError(
                        f"Node {nid!r} ({node.op_type}) references nonexistent input node {inp!r}."
                    )

        # Check outputs exist
        if not self.outputs:
            raise GraphValidationError("Graph has no marked output nodes.")
        for out_id in self.outputs:
            if out_id not in self.nodes:
                raise GraphValidationError(f"Marked output node {out_id!r} does not exist in graph.")

        # Check for cycles
        try:
            self.topological_sort()
        except CycleDetectedError as e:
            raise GraphValidationError(str(e)) from e

    def clone(self) -> ComputationGraph:
        """Return a completely independent deep copy of the computation graph."""
        cloned = ComputationGraph(name=f"{self.name}_copy")
        cloned._next_id = self._next_id
        for nid, node in self.nodes.items():
            cloned.nodes[nid] = node.clone()
        cloned.outputs = list(self.outputs)
        return cloned

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a NetworkX directed graph for analysis and layout."""
        G = nx.DiGraph()
        for nid, node in self.nodes.items():
            G.add_node(
                nid,
                op_type=node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type),
                shape=str(node.shape) if node.shape is not None else "Unknown",
                dtype=node.dtype,
                name=node.name,
                is_output=nid in self.outputs,
                is_input=node.is_input,
                is_constant=node.is_constant,
                is_fused=node.is_fused,
            )
            for inp in node.inputs:
                if inp in self.nodes:
                    G.add_edge(inp, nid)
        return G

    def __repr__(self) -> str:
        return f"ComputationGraph(name={self.name!r}, nodes={len(self.nodes)}, outputs={self.outputs})"
