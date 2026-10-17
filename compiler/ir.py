"""Compiler Intermediate Representation (IR) generation and formatting."""

from __future__ import annotations
from typing import Any, Dict, List
from compiler.graph import ComputationGraph
from compiler.node import OpType


def generate_ir_instruction(node_id: str, graph: ComputationGraph) -> str:
    """Generate a single line of human-readable IR for a given node."""
    node = graph.get_node(node_id)
    op_str = node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type)
    shape_str = f"shape={list(node.shape)}" if node.shape is not None else "shape=<?>"
    dtype_str = f"dtype={node.dtype}"
    
    if node.is_input:
        desc = f"input({shape_str}, {dtype_str})"
    elif node.is_constant:
        extra = ""
        if node.data is not None:
            if node.data.size <= 4:
                extra = f", val={node.data.tolist()}"
            else:
                extra = f", numel={node.data.size}"
        desc = f"constant({shape_str}, {dtype_str}{extra})"
    elif node.op_type == OpType.SCALAR_MUL:
        scalar = node.metadata.get("scalar", 1.0)
        desc = f"scalar_mul(%{node.inputs[0]}, scalar={scalar}) -> ({shape_str}, {dtype_str})"
    else:
        args = ", ".join(f"%{inp}" for inp in node.inputs)
        desc = f"{op_str}({args}) -> ({shape_str}, {dtype_str})"

    return f"%{node_id} = {desc}"


def generate_ir(graph: ComputationGraph) -> str:
    """Generate intermediate representation (IR) text for the entire computation graph.

    Nodes are ordered according to topological execution order.
    """
    try:
        order = graph.topological_sort()
    except Exception:
        order = list(graph.nodes.keys())

    lines: List[str] = [
        f"; ========================================================",
        f"; Mini AI Compiler Intermediate Representation (IR)",
        f"; Graph Name: {graph.name}",
        f"; Node Count: {len(graph.nodes)} | Outputs: {graph.outputs}",
        f"; ========================================================",
        "",
    ]

    for nid in order:
        lines.append(generate_ir_instruction(nid, graph))

    lines.append("")
    if graph.outputs:
        out_args = ", ".join(f"%{out_id}" for out_id in graph.outputs)
        lines.append(f"return {out_args}")
    else:
        lines.append("; No outputs declared")

    return "\n".join(lines)


def print_ir(graph: ComputationGraph) -> None:
    """Print the formatted IR to stdout."""
    print(generate_ir(graph))


def get_ir_table(graph: ComputationGraph) -> List[Dict[str, Any]]:
    """Return a structured table representation of the IR for UI rendering."""
    try:
        order = graph.topological_sort()
    except Exception:
        order = list(graph.nodes.keys())

    table: List[Dict[str, Any]] = []
    for idx, nid in enumerate(order, start=1):
        node = graph.get_node(nid)
        op_str = node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type)
        inputs_str = ", ".join(f"%{inp}" for inp in node.inputs) if node.inputs else "-"
        shape_str = str(list(node.shape)) if node.shape is not None else "Unknown"
        
        table.append({
            "Step": idx,
            "Target Node": f"%{nid}",
            "Operation": op_str.upper(),
            "Arguments": inputs_str,
            "Inferred Shape": shape_str,
            "Dtype": node.dtype,
            "Is Output": "Yes (Output)" if nid in graph.outputs else "No",
            "Metadata": str(node.metadata) if node.metadata else "-",
        })

    return table
