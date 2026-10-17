"""Dead Code Elimination (DCE) Optimization Pass.

Removes all nodes that do not transitively contribute to the marked graph outputs.
"""

from __future__ import annotations
from typing import Set
from compiler.graph import ComputationGraph
from compiler.node import OpType
from optimizations.base import OptimizationPass, OptimizationResult


class DeadCodeEliminationPass(OptimizationPass):
    """Prunes dead/unreachable operations from the computation graph."""

    name: str = "DeadCodeElimination"
    description: str = "Prunes unused operations and intermediate tensors not contributing to outputs."

    def run(self, graph: ComputationGraph) -> OptimizationResult:
        g = graph.clone()
        result = OptimizationResult(optimized_graph=g, pass_name=self.name)

        if not g.outputs:
            result.log.append("No outputs declared; DeadCodeElimination skipped.")
            return result

        # Reverse reachability traversal from outputs
        live_nodes: Set[str] = set()
        queue = [out_id for out_id in g.outputs if out_id in g.nodes]
        for out_id in queue:
            live_nodes.add(out_id)

        while queue:
            curr_id = queue.pop(0)
            node = g.nodes[curr_id]
            for parent_id in node.inputs:
                if parent_id in g.nodes and parent_id not in live_nodes:
                    live_nodes.add(parent_id)
                    queue.append(parent_id)

        # Identify dead nodes
        dead_nodes = [nid for nid in g.nodes if nid not in live_nodes]

        for dead_id in dead_nodes:
            dead_node = g.nodes[dead_id]
            op_str = dead_node.op_type.value if isinstance(dead_node.op_type, OpType) else str(dead_node.op_type)
            result.nodes_removed.append(dead_id)
            result.log.append(f"Pruned dead node %{dead_id} ({op_str}, inputs={dead_node.inputs})")
            g.remove_node(dead_id)

        if not dead_nodes:
            result.log.append("No dead code detected; all nodes contribute to graph outputs.")

        return result
