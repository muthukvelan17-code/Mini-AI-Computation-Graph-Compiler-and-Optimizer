"""Optional PyTorch comparison module for benchmarking against real tensor framework."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import OpType

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@dataclass
class PyTorchComparisonResult:
    available: bool
    execution_time_ms: float = 0.0
    outputs: Dict[str, np.ndarray] = None
    max_abs_diff_vs_numpy: float = 0.0
    mean_abs_diff_vs_numpy: float = 0.0
    is_numerically_close: bool = False
    message: str = ""


def is_pytorch_available() -> bool:
    """Return whether PyTorch is installed and available in the current environment."""
    return HAS_TORCH


def run_pytorch_benchmark(
    graph: ComputationGraph,
    inputs: Dict[str, np.ndarray],
    numpy_outputs: Optional[Dict[str, np.ndarray]] = None,
    iterations: int = 50,
    warmup: int = 5,
) -> PyTorchComparisonResult:
    """Execute the equivalent computation in PyTorch and compare numerical outputs and timing."""
    if not HAS_TORCH:
        return PyTorchComparisonResult(
            available=False,
            message="PyTorch is not installed in the environment. Install it via 'pip install torch' for real framework comparisons.",
        )

    # Convert inputs to torch tensors
    torch_inputs: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            torch_inputs[k] = torch.from_numpy(v.copy())
        else:
            torch_inputs[k] = torch.tensor(v, dtype=torch.float32)

    order = graph.topological_sort()

    def _execute_torch() -> Dict[str, torch.Tensor]:
        cache: Dict[str, torch.Tensor] = {}
        for nid in order:
            node = graph.get_node(nid)
            op = node.op_type

            if node.is_input:
                if nid in torch_inputs:
                    cache[nid] = torch_inputs[nid]
                elif node.name in torch_inputs:
                    cache[nid] = torch_inputs[node.name]
                elif node.data is not None:
                    cache[nid] = torch.from_numpy(node.data)
                else:
                    raise RuntimeError(f"Missing input tensor for {nid!r}")
            elif node.is_constant:
                cache[nid] = torch.from_numpy(node.data)
            else:
                args = [cache[inp] for inp in node.inputs]
                if op == OpType.ADD:
                    cache[nid] = torch.add(args[0], args[1])
                elif op == OpType.SUB:
                    cache[nid] = torch.sub(args[0], args[1])
                elif op == OpType.MUL:
                    cache[nid] = torch.mul(args[0], args[1])
                elif op == OpType.MATMUL:
                    cache[nid] = torch.matmul(args[0], args[1])
                elif op == OpType.RELU:
                    cache[nid] = torch.relu(args[0])
                elif op == OpType.SIGMOID:
                    cache[nid] = torch.sigmoid(args[0])
                elif op == OpType.SCALAR_MUL:
                    scalar = float(node.metadata.get("scalar", 1.0))
                    cache[nid] = args[0] * scalar
                elif op == OpType.FUSED_MUL_ADD:
                    cache[nid] = torch.addcmul(args[2], args[0], args[1])
                elif op == OpType.FUSED_MATMUL_ADD:
                    cache[nid] = torch.add(torch.matmul(args[0], args[1]), args[2])
                elif op == OpType.FUSED_MATMUL_ADD_RELU:
                    cache[nid] = torch.relu(torch.add(torch.matmul(args[0], args[1]), args[2]))
                elif op == OpType.FUSED_ADD_RELU:
                    cache[nid] = torch.relu(torch.add(args[0], args[1]))
                else:
                    raise RuntimeError(f"Unsupported operation in PyTorch benchmark: {op}")

        return {out_id: cache[out_id] for out_id in graph.outputs if out_id in cache}

    # Warmup runs
    for _ in range(warmup):
        _ = _execute_torch()

    # Timing runs
    durations: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pt_outs = _execute_torch()
        durations.append((time.perf_counter() - t0) * 1000.0)

    median_time_ms = float(np.median(durations))

    # Convert final outputs to numpy
    final_outputs = {k: v.detach().cpu().numpy() for k, v in pt_outs.items()}

    # Calculate differences against NumPy outputs if provided
    max_diff = 0.0
    mean_diff = 0.0
    is_close = True

    if numpy_outputs:
        diffs = []
        for out_id in graph.outputs:
            if out_id in numpy_outputs and out_id in final_outputs:
                np_val = numpy_outputs[out_id]
                pt_val = final_outputs[out_id]
                d = np.abs(np_val - pt_val)
                diffs.append(d)
                if not np.allclose(np_val, pt_val, atol=1e-5, rtol=1e-5):
                    is_close = False
        if diffs:
            all_diffs = np.concatenate([d.flatten() for d in diffs])
            max_diff = float(np.max(all_diffs))
            mean_diff = float(np.mean(all_diffs))

    return PyTorchComparisonResult(
        available=True,
        execution_time_ms=median_time_ms,
        outputs=final_outputs,
        max_abs_diff_vs_numpy=max_diff,
        mean_abs_diff_vs_numpy=mean_diff,
        is_numerically_close=is_close,
        message="PyTorch comparison executed successfully.",
    )
