"""Common Subexpression Elimination (CSE) Optimization Pass.

Identifies redundant duplicate computations and reuses the existing evaluated tensor.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import OpType
from optimizations.base import OptimizationPass, OptimizationResult


def _node_signature(node, graph: ComputationGraph) -> Tuple[Any, ...]:
    """Generate a canonical hashable signature for a node."""
    op = node.op_type
    op_str = op.value if isinstance(op, OpType) else str(op)

    # Input nodes are always distinct
    if node.is_input:
        return ("input", node.id)

    # Constant nodes signature
    if node.is_constant:
        if node.data is not None:
            # Hash array bytes for fast match
            data_bytes = node.data.tobytes()
            return ("constant", node.shape, node.dtype, data_bytes)
        return ("constant", node.id)

    # Commutative element-wise operations can sort their input IDs
    if op in (OpType.ADD, OpType.MUL, OpType.FUSED_ADD_RELU):
        inputs_key = tuple(sorted(node.inputs))
    else:
        inputs_key = tuple(node.inputs)

    # Include metadata if present (like scalar value)
    meta_items = tuple(sorted((k, str(v)) for k, v in node.metadata.items() if not k.startswith("folded_")))

    return (op_str, inputs_key, meta_items)


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Detects and removes identical redundant operations across the DAG."""

    name: str = "CommonSubexpressionElimination"
    description: str = "Replaces repeated identical operations with a single shared result."

    def run(self, graph: ComputationGraph) -> OptimizationResult:
        g = graph.clone()
        result = OptimizationResult(optimized_graph=g, pass_name=self.name)

        changed = True
        pass_iteration = 0
        max_iterations = 20

        while changed and pass_iteration < max_iterations:
            changed = False
            pass_iteration += 1
            seen_signatures: Dict[Tuple[Any, ...], str] = {}
            order = g.topological_sort()

            for nid in order:
                if nid not in g.nodes:
                    continue
                node = g.nodes[nid]

                # Don't eliminate input nodes
                if node.is_input:
                    continue

                sig = _node_signature(node, g)
                if sig in seen_signatures:
                    canonical_id = seen_signatures[sig]
                    if canonical_id != nid:
                        # Found duplicate computation!
                        op_str = node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type)
                        g.replace_node_references(nid, canonical_id)
                        g.remove_node(nid)
                        
                        result.nodes_removed.append(nid)
                        result.nodes_modified.append(canonical_id)
                        result.log.append(
                            f"Eliminated redundant subexpression %{nid} ({op_str}); "
                            f"reused result from %{canonical_id}."
                        )
                        changed = True
                        break  # restart pass
                else:
                    seen_signatures[sig] = nid

        if not result.nodes_removed:
            result.log.append("No common subexpressions detected; all operations are unique.")

        return result
