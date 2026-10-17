"""Constant Folding Optimization Pass.

Pre-computes operations whose input operands are all known compile-time constants.
"""

from __future__ import annotations
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import Node, OpType
from optimizations.base import OptimizationPass, OptimizationResult


class ConstantFoldingPass(OptimizationPass):
    """Folds operations with compile-time constant inputs into static constant tensors."""

    name: str = "ConstantFolding"
    description: str = "Pre-computes operations with constant operands at compile time."

    def run(self, graph: ComputationGraph) -> OptimizationResult:
        g = graph.clone()
        result = OptimizationResult(optimized_graph=g, pass_name=self.name)

        changed = True
        pass_iteration = 0
        max_iterations = 20

        while changed and pass_iteration < max_iterations:
            changed = False
            pass_iteration += 1
            order = g.topological_sort()

            for nid in order:
                if nid not in g.nodes:
                    continue
                node = g.nodes[nid]

                # Skip if already a constant, input, or has no inputs
                if node.is_constant or node.is_input or not node.inputs:
                    continue

                # Check if all inputs are constant nodes
                parent_nodes = [g.nodes[inp] for inp in node.inputs]
                all_parents_constant = all(p.is_constant and p.data is not None for p in parent_nodes)

                if all_parents_constant:
                    args = [p.data for p in parent_nodes]
                    op = node.op_type

                    try:
                        if op == OpType.ADD:
                            val = np.add(args[0], args[1])
                        elif op == OpType.SUB:
                            val = np.subtract(args[0], args[1])
                        elif op == OpType.MUL:
                            val = np.multiply(args[0], args[1])
                        elif op == OpType.MATMUL:
                            val = np.matmul(args[0], args[1])
                        elif op == OpType.RELU:
                            val = np.maximum(0.0, args[0])
                        elif op == OpType.SIGMOID:
                            val = np.where(args[0] >= 0, 1.0 / (1.0 + np.exp(-args[0])), np.exp(args[0]) / (1.0 + np.exp(args[0])))
                        elif op == OpType.SCALAR_MUL:
                            scalar = float(node.metadata.get("scalar", 1.0))
                            val = np.multiply(args[0], scalar)
                        elif op == OpType.FUSED_MUL_ADD:
                            val = np.add(np.multiply(args[0], args[1]), args[2])
                        elif op == OpType.FUSED_MATMUL_ADD:
                            val = np.add(np.matmul(args[0], args[1]), args[2])
                        elif op == OpType.FUSED_MATMUL_ADD_RELU:
                            val = np.maximum(0.0, np.add(np.matmul(args[0], args[1]), args[2]))
                        elif op == OpType.FUSED_ADD_RELU:
                            val = np.maximum(0.0, np.add(args[0], args[1]))
                        else:
                            continue

                        # Cast to target dtype
                        val = val.astype(np.dtype(node.dtype))
                        old_op = node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type)
                        
                        # Mutate node to CONSTANT
                        node.op_type = OpType.CONSTANT
                        node.data = val
                        node.shape = val.shape
                        old_inputs = list(node.inputs)
                        node.inputs = []
                        node.metadata["folded_from"] = old_op
                        node.metadata["folded_inputs"] = old_inputs

                        result.nodes_modified.append(nid)
                        result.log.append(
                            f"Folded operation %{nid} ({old_op} with inputs {old_inputs}) -> "
                            f"Constant tensor with shape {node.shape}."
                        )
                        changed = True
                        break  # restart topological pass to reflect updated constant status
                    except Exception as e:
                        result.log.append(f"Failed to fold %{nid} ({op}): {e}")

        return result
