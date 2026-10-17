"""Operator Fusion Optimization Pass.

Fuses consecutive compatible operations into composite kernels to reduce kernel launch overhead
and intermediate memory traffic.
"""

from __future__ import annotations
from typing import List
from compiler.graph import ComputationGraph
from compiler.node import Node, OpType
from optimizations.base import OptimizationPass, OptimizationResult


class OperatorFusionPass(OptimizationPass):
    """Fuses eligible operation sequences (e.g. Mul+Add -> FusedMulAdd, MatMul+Add -> FusedLinear, etc.)."""

    name: str = "OperatorFusion"
    description: str = "Combines consecutive operations (e.g. a*b+c or (X@W)+b) into fused execution nodes."

    def _can_fuse_node(self, g: ComputationGraph, node_id: str) -> bool:
        """Check if node_id can be safely fused and deleted."""
        if node_id in g.outputs:
            return False
        total_uses = sum(n.inputs.count(node_id) for n in g.nodes.values())
        return total_uses == 1

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

                # --- 1. Pattern: MatMul-Add-ReLU -> FusedMatMulAddReLU ---
                if node.op_type == OpType.RELU and len(node.inputs) == 1:
                    child_id = node.inputs[0]
                    child = g.nodes.get(child_id)
                    if (
                        child is not None
                        and child.op_type == OpType.FUSED_MATMUL_ADD
                        and self._can_fuse_node(g, child_id)
                    ):
                        fused_id = f"fused_linear_relu_{nid}"
                        fused_node = Node(
                            node_id=fused_id,
                            op_type=OpType.FUSED_MATMUL_ADD_RELU,
                            inputs=list(child.inputs),  # [X, W, b]
                            shape=node.shape,
                            dtype=node.dtype,
                            metadata={"fused_from": [child_id, nid]},
                            name=f"FusedLinearReLU({child.name}+{node.name})",
                        )
                        g.add_node(fused_node)
                        g.replace_node_references(nid, fused_id)
                        g.remove_node(nid)
                        g.remove_node(child_id)

                        result.nodes_added.append(fused_id)
                        result.nodes_removed.extend([nid, child_id])
                        result.nodes_fused.append((fused_id, [child_id, nid]))
                        result.log.append(f"Fused FusedMatmulAdd %{child_id} + ReLU %{nid} -> %{fused_id}.")
                        changed = True
                        break

                # --- 2. Pattern: MatMul + Add -> FusedMatMulAdd (Linear layer) ---
                if node.op_type == OpType.ADD and len(node.inputs) == 2:
                    in0_id, in1_id = node.inputs[0], node.inputs[1]
                    in0, in1 = g.nodes[in0_id], g.nodes[in1_id]

                    mm_id, bias_id = None, None
                    if in0.op_type == OpType.MATMUL and self._can_fuse_node(g, in0_id) and in0_id != in1_id:
                        mm_id, bias_id = in0_id, in1_id
                    elif in1.op_type == OpType.MATMUL and self._can_fuse_node(g, in1_id) and in1_id != in0_id:
                        mm_id, bias_id = in1_id, in0_id

                    if mm_id is not None and bias_id is not None:
                        mm_node = g.nodes[mm_id]
                        fused_id = f"fused_linear_{nid}"
                        fused_node = Node(
                            node_id=fused_id,
                            op_type=OpType.FUSED_MATMUL_ADD,
                            inputs=[mm_node.inputs[0], mm_node.inputs[1], bias_id],
                            shape=node.shape,
                            dtype=node.dtype,
                            metadata={"fused_from": [mm_id, nid]},
                            name=f"FusedLinear({mm_node.name}+{node.name})",
                        )
                        g.add_node(fused_node)
                        g.replace_node_references(nid, fused_id)
                        g.remove_node(nid)
                        g.remove_node(mm_id)

                        result.nodes_added.append(fused_id)
                        result.nodes_removed.extend([nid, mm_id])
                        result.nodes_fused.append((fused_id, [mm_id, nid]))
                        result.log.append(f"Fused MatMul %{mm_id} + Add %{nid} -> %{fused_id}.")
                        changed = True
                        break

                # --- 3. Pattern: Mul + Add -> FusedMulAdd (a * b + c) ---
                if node.op_type == OpType.ADD and len(node.inputs) == 2:
                    in0_id, in1_id = node.inputs[0], node.inputs[1]
                    in0, in1 = g.nodes[in0_id], g.nodes[in1_id]

                    mul_id, c_id = None, None
                    if in0.op_type == OpType.MUL and self._can_fuse_node(g, in0_id) and in0_id != in1_id:
                        mul_id, c_id = in0_id, in1_id
                    elif in1.op_type == OpType.MUL and self._can_fuse_node(g, in1_id) and in1_id != in0_id:
                        mul_id, c_id = in1_id, in0_id

                    if mul_id is not None and c_id is not None:
                        mul_node = g.nodes[mul_id]
                        fused_id = f"fused_fma_{nid}"
                        fused_node = Node(
                            node_id=fused_id,
                            op_type=OpType.FUSED_MUL_ADD,
                            inputs=[mul_node.inputs[0], mul_node.inputs[1], c_id],
                            shape=node.shape,
                            dtype=node.dtype,
                            metadata={"fused_from": [mul_id, nid]},
                            name=f"FusedMulAdd({mul_node.name}+{node.name})",
                        )
                        g.add_node(fused_node)
                        g.replace_node_references(nid, fused_id)
                        g.remove_node(nid)
                        g.remove_node(mul_id)

                        result.nodes_added.append(fused_id)
                        result.nodes_removed.extend([nid, mul_id])
                        result.nodes_fused.append((fused_id, [mul_id, nid]))
                        result.log.append(f"Fused Mul %{mul_id} + Add %{nid} -> %{fused_id}.")
                        changed = True
                        break

                # --- 4. Pattern: Add + ReLU -> FusedAddReLU ---
                if node.op_type == OpType.RELU and len(node.inputs) == 1:
                    child_id = node.inputs[0]
                    child = g.nodes.get(child_id)
                    if (
                        child is not None
                        and child.op_type == OpType.ADD
                        and self._can_fuse_node(g, child_id)
                    ):
                        fused_id = f"fused_add_relu_{nid}"
                        fused_node = Node(
                            node_id=fused_id,
                            op_type=OpType.FUSED_ADD_RELU,
                            inputs=list(child.inputs),
                            shape=node.shape,
                            dtype=node.dtype,
                            metadata={"fused_from": [child_id, nid]},
                            name=f"FusedAddReLU({child.name}+{node.name})",
                        )
                        g.add_node(fused_node)
                        g.replace_node_references(nid, fused_id)
                        g.remove_node(nid)
                        g.remove_node(child_id)

                        result.nodes_added.append(fused_id)
                        result.nodes_removed.extend([nid, child_id])
                        result.nodes_fused.append((fused_id, [child_id, nid]))
                        result.log.append(f"Fused Add %{child_id} + ReLU %{nid} -> %{fused_id}.")
                        changed = True
                        break

        if not result.nodes_fused:
            result.log.append("No fusible operator patterns detected.")

        return result
