"""Safe JSON serialization and deserialization for ComputationGraph."""

from __future__ import annotations
import json
from typing import Any, Dict
from compiler.graph import ComputationGraph, GraphError
from compiler.node import Node


class SerializationError(GraphError):
    """Raised when serialization or deserialization fails."""
    pass


def serialize_graph(graph: ComputationGraph) -> str:
    """Serialize a ComputationGraph to a clean JSON string without unsafe constructs."""
    try:
        nodes_data = [node.to_dict() for node in graph.nodes.values()]
        payload = {
            "format": "mini_ai_compiler_ir_v1",
            "name": graph.name,
            "outputs": graph.outputs,
            "nodes": nodes_data,
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        raise SerializationError(f"Failed to serialize graph {graph.name!r}: {e}") from e


def deserialize_graph(json_str: str) -> ComputationGraph:
    """Deserialize a ComputationGraph from a JSON string safely."""
    try:
        data: Dict[str, Any] = json.loads(json_str)
        if not isinstance(data, dict) or "nodes" not in data or "outputs" not in data:
            raise SerializationError("Invalid graph JSON structure: missing required 'nodes' or 'outputs' keys.")

        graph = ComputationGraph(name=data.get("name", "imported_graph"))
        
        for node_dict in data["nodes"]:
            node = Node.from_dict(node_dict)
            graph.add_node(node)

        graph.set_outputs(data["outputs"])
        graph.validate()
        return graph
    except Exception as e:
        raise SerializationError(f"Failed to deserialize graph from JSON: {e}") from e


def save_graph(graph: ComputationGraph, filepath: str) -> None:
    """Save graph to a JSON file."""
    content = serialize_graph(graph)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def load_graph(filepath: str) -> ComputationGraph:
    """Load graph from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return deserialize_graph(content)
