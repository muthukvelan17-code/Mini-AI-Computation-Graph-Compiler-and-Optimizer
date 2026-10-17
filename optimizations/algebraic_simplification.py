"""Algebraic Simplification Optimization Pass.

Simplifies algebraic identity operations such as x + 0, x * 1, x - 0, and scalar multiplication by 1.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import Node, OpType
from compiler.shape_inference import infer_shapes
from optimizations.base import OptimizationPass, OptimizationResult


def _is_all_value(node: Node, val: float) -> bool:
    """Check if a constant node consists entirely of the specified scalar value."""
    if not node.is_constant or node.data is None:
        return False
    return bool(np.all(node.data == val))


class AlgebraicSimplificationPass(OptimizationPass):
    """Eliminates mathematical no-ops (x + 0, x * 1, etc.) while preserving tensor rank and shape."""

    name: str = "AlgebraicSimplification"
    description: str = "Simplifies expressions with identity values (e.g. x + 0 -> x, x * 1 -> x)."

    def run(self, graph: ComputationGraph) -> OptimizationResult:
        g = graph.clone()
        result = OptimizationResult(optimized_graph=g, pass_name=self.name)

        # Infer shapes first so we can verify shape preservation
        try:
            infer_shapes(g)
        except Exception:
            pass

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
                op = node.op_type

                replacement_id: Optional[str] = None
                reason: str = ""

                # --- Unary Scalar Multiplication ---
                if op == OpType.SCALAR_MUL:
                    scalar = float(node.metadata.get("scalar", 1.0))
                    parent_id = node.inputs[0]
                    if scalar == 1.0:
                        replacement_id = parent_id
                        reason = f"scalar_mul(%{parent_id}, 1.0) -> %{parent_id}"
                    elif scalar == 0.0:
                        # Replace with zero constant matching shape
                        p_node = g.nodes[parent_id]
                        if p_node.shape is not None:
                            zero_data = np.zeros(p_node.shape, dtype=np.dtype(node.dtype))
                            c_node = g.add_constant(zero_data, name=f"zero_{nid}", dtype=node.dtype)
                            replacement_id = c_node.id
                            reason = f"scalar_mul(%{parent_id}, 0.0) -> zero constant tensor"

                # --- Binary Operations: ADD, SUB, MUL ---
                elif op in (OpType.ADD, OpType.SUB, OpType.MUL) and len(node.inputs) == 2:
                    in0_id, in1_id = node.inputs[0], node.inputs[1]
                    in0, in1 = g.nodes[in0_id], g.nodes[in1_id]

                    # 1. ADD Simplification: x + 0 -> x or 0 + x -> x
                    if op == OpType.ADD:
                        if _is_all_value(in1, 0.0):
                            # Verify shape is preserved (broadcasting didn't expand shape beyond in0)
                            if node.shape is None or node.shape == in0.shape:
                                replacement_id = in0_id
                                reason = f"ADD(%{in0_id}, %{in1_id}(0)) -> %{in0_id}"
                        elif _is_all_value(in0, 0.0):
                            if node.shape is None or node.shape == in1.shape:
                                replacement_id = in1_id
                                reason = f"ADD(%{in0_id}(0), %{in1_id}) -> %{in1_id}"

                    # 2. SUB Simplification: x - 0 -> x, or x - x -> 0
                    elif op == OpType.SUB:
                        if in0_id == in1_id and node.shape is not None:
                            zero_data = np.zeros(node.shape, dtype=np.dtype(node.dtype))
                            c_node = g.add_constant(zero_data, name=f"zero_{nid}", dtype=node.dtype)
                            replacement_id = c_node.id
                            reason = f"SUB(%{in0_id}, %{in1_id}) -> zero constant tensor"
                        elif _is_all_value(in1, 0.0):
                            if node.shape is None or node.shape == in0.shape:
                                replacement_id = in0_id
                                reason = f"SUB(%{in0_id}, %{in1_id}(0)) -> %{in0_id}"

                    # 3. MUL Simplification: x * 1 -> x, 1 * x -> x, x * 0 -> 0
                    elif op == OpType.MUL:
                        if _is_all_value(in1, 1.0):
                            if node.shape is None or node.shape == in0.shape:
                                replacement_id = in0_id
                                reason = f"MUL(%{in0_id}, %{in1_id}(1)) -> %{in0_id}"
                        elif _is_all_value(in0, 1.0):
                            if node.shape is None or node.shape == in1.shape:
                                replacement_id = in1_id
                                reason = f"MUL(%{in0_id}(1), %{in1_id}) -> %{in1_id}"
                        elif _is_all_value(in1, 0.0) or _is_all_value(in0, 0.0):
                            if node.shape is not None:
                                zero_data = np.zeros(node.shape, dtype=np.dtype(node.dtype))
                                c_node = g.add_constant(zero_data, name=f"zero_{nid}", dtype=node.dtype)
                                replacement_id = c_node.id
                                reason = f"MUL by 0 -> zero constant tensor"

                if replacement_id is not None:
                    g.replace_node_references(nid, replacement_id)
                    g.remove_node(nid)
                    result.nodes_removed.append(nid)
                    result.nodes_modified.append(replacement_id)
                    result.log.append(f"Simplified %{nid} via identity: {reason}.")
                    changed = True
                    break

        if not result.nodes_removed:
            result.log.append("No algebraic simplifications applicable.")

        return result
