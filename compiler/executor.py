"""NumPy-based execution engine for ComputationGraph."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from compiler.node import Node, OpType
from compiler.graph import ComputationGraph, GraphError


class ExecutionError(GraphError):
    """Raised during graph execution failure."""
    pass


@dataclass
class ExecutionResult:
    """Stores the outputs and diagnostic metrics of a graph execution run."""
    outputs: Dict[str, np.ndarray]
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    node_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_ops: int = 0
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    def get_first_output(self) -> Optional[np.ndarray]:
        """Convenience method to retrieve the primary output tensor."""
        if not self.outputs:
            return None
        return next(iter(self.outputs.values()))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    # Split positive and negative values to prevent float overflow in exp(-x) or exp(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    result = np.empty_like(x, dtype=np.float32 if x.dtype == np.float32 else np.float64)
    
    # For x >= 0: 1 / (1 + exp(-x))
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    # For x < 0: exp(x) / (1 + exp(x))
    exp_neg = np.exp(x[neg_mask])
    result[neg_mask] = exp_neg / (1.0 + exp_neg)
    return result


class GraphExecutor:
    """Interpreter that executes DAG nodes in topological order using NumPy."""

    def __init__(self, graph: ComputationGraph):
        self.graph = graph

    def execute(
        self,
        inputs: Optional[Dict[str, Union[np.ndarray, List, float, int]]] = None,
        record_intermediates: bool = True,
        record_node_timings: bool = False,
    ) -> ExecutionResult:
        """Execute the computation graph with the provided input tensors.

        Args:
            inputs: Dictionary mapping input node ID (or name) to tensor array/data.
            record_intermediates: If True, intermediate node outputs are retained in the result.
            record_node_timings: If True, individual node execution latencies are measured.

        Returns:
            ExecutionResult containing output tensors, intermediate values, and timing metrics.
        """
        inputs = inputs or {}
        # Normalize input keys (match either node.id or node.name)
        resolved_inputs: Dict[str, np.ndarray] = {}
        for nid, node in self.graph.nodes.items():
            if node.is_input:
                if nid in inputs:
                    val = inputs[nid]
                elif node.name in inputs:
                    val = inputs[node.name]
                elif node.data is not None:
                    val = node.data
                else:
                    raise ExecutionError(f"Missing required input tensor for node {nid!r} ({node.name}).")

                # Ensure numpy array and avoid mutating caller's tensor
                if not isinstance(val, np.ndarray):
                    resolved_inputs[nid] = np.array(val, dtype=np.dtype(node.dtype)).copy()
                else:
                    resolved_inputs[nid] = val.astype(np.dtype(node.dtype), copy=True)

        order = self.graph.topological_sort()
        cache: Dict[str, np.ndarray] = {}
        node_timings: Dict[str, float] = {}
        op_count = 0

        total_start = time.perf_counter()

        for nid in order:
            node = self.graph.get_node(nid)
            op = node.op_type
            
            node_start = time.perf_counter() if record_node_timings else 0.0

            if op == OpType.INPUT:
                res = resolved_inputs[nid]
            elif op == OpType.CONSTANT:
                if node.data is None:
                    raise ExecutionError(f"Constant node {nid!r} has no data array.")
                res = node.data
            else:
                # Gather parent arguments from cache
                args = [cache[inp] for inp in node.inputs]
                op_count += 1
                
                if op == OpType.ADD:
                    res = np.add(args[0], args[1])
                elif op == OpType.SUB:
                    res = np.subtract(args[0], args[1])
                elif op == OpType.MUL:
                    res = np.multiply(args[0], args[1])
                elif op == OpType.MATMUL:
                    res = np.matmul(args[0], args[1])
                elif op == OpType.RELU:
                    res = np.maximum(0.0, args[0])
                elif op == OpType.SIGMOID:
                    res = _sigmoid(args[0])
                elif op == OpType.SCALAR_MUL:
                    scalar = float(node.metadata.get("scalar", 1.0))
                    res = np.multiply(args[0], scalar)
                elif op == OpType.FUSED_MUL_ADD:
                    # a * b + c (2 operations fused into 1 execution step)
                    res = np.add(np.multiply(args[0], args[1]), args[2])
                    op_count += 1
                elif op == OpType.FUSED_MATMUL_ADD:
                    # (X @ W) + b
                    res = np.add(np.matmul(args[0], args[1]), args[2])
                    op_count += 1
                elif op == OpType.FUSED_MATMUL_ADD_RELU:
                    # relu((X @ W) + b)
                    res = np.maximum(0.0, np.add(np.matmul(args[0], args[1]), args[2]))
                    op_count += 2
                elif op == OpType.FUSED_ADD_RELU:
                    # relu(a + b)
                    res = np.maximum(0.0, np.add(args[0], args[1]))
                    op_count += 1
                else:
                    raise ExecutionError(f"Unsupported operation type {op!r} on node {nid!r}.")

            if record_node_timings:
                node_timings[nid] = (time.perf_counter() - node_start) * 1000.0

            cache[nid] = res
            node.result = res

        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        # Collect outputs
        outputs: Dict[str, np.ndarray] = {
            out_id: cache[out_id] for out_id in self.graph.outputs if out_id in cache
        }
        output_shapes: Dict[str, Tuple[int, ...]] = {
            out_id: arr.shape for out_id, arr in outputs.items()
        }

        return ExecutionResult(
            outputs=outputs,
            intermediates=cache if record_intermediates else {},
            execution_time_ms=total_time_ms,
            node_timings_ms=node_timings,
            total_ops=op_count,
            output_shapes=output_shapes,
        )
