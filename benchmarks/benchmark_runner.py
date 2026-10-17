"""Benchmark runner comparing original and optimized graph execution performance."""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np

from compiler.graph import ComputationGraph
from compiler.executor import GraphExecutor
from compiler.compiler_pipeline import PipelineResult
from benchmarks.metrics import BenchmarkMetrics


class BenchmarkRunner:
    """Runs rigorous, repeatable performance benchmarks and numerical correctness comparisons."""

    @staticmethod
    def run_benchmark(
        original_graph: ComputationGraph,
        optimized_graph: ComputationGraph,
        inputs: Dict[str, np.ndarray],
        iterations: int = 50,
        warmup: int = 5,
        pipeline_result: Optional[PipelineResult] = None,
    ) -> BenchmarkMetrics:
        """Execute multiple iterations of original vs optimized graphs and compute performance statistics.

        Args:
            original_graph: Baseline unoptimized graph.
            optimized_graph: Optimized graph produced by compiler.
            inputs: Dictionary of input arrays.
            iterations: Number of timed runs.
            warmup: Number of untimed warmup runs.
            pipeline_result: Optional PipelineResult for compilation overhead and pass counts.

        Returns:
            BenchmarkMetrics object containing statistics and numerical diffs.
        """
        orig_executor = GraphExecutor(original_graph)
        opt_executor = GraphExecutor(optimized_graph)

        # 1. Warmup runs
        for _ in range(warmup):
            _ = orig_executor.execute(inputs, record_intermediates=False)
            _ = opt_executor.execute(inputs, record_intermediates=False)

        # 2. Benchmark baseline (original graph)
        orig_times_ms: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = orig_executor.execute(inputs, record_intermediates=False)
            orig_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # 3. Benchmark optimized graph
        opt_times_ms: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = opt_executor.execute(inputs, record_intermediates=False)
            opt_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # 4. Statistical analysis using median and standard deviation
        orig_median = float(np.median(orig_times_ms))
        opt_median = float(np.median(opt_times_ms))
        orig_std = float(np.std(orig_times_ms))
        opt_std = float(np.std(opt_times_ms))

        abs_diff = orig_median - opt_median
        
        # Speedup (%) = ((Original Time - Optimized Time) / Original Time) * 100
        if orig_median > 0:
            speedup_pct = (abs_diff / orig_median) * 100.0
        else:
            speedup_pct = 0.0

        is_slowdown = speedup_pct < 0

        # Conclusiveness evaluation for micro-benchmarks
        # If execution time is extremely small (< 0.04 ms), noise is high
        if orig_median < 0.04 and opt_median < 0.04:
            is_conclusive = False
            conclusion_reason = (
                "Execution latency is under 40 microseconds. Python interpreter function call "
                "overhead dominates tensor computation time on tiny graphs."
            )
        elif abs(abs_diff) < 0.5 * min(orig_std, opt_std):
            is_conclusive = False
            conclusion_reason = "Timing difference is within the measurement variance (std dev)."
        else:
            is_conclusive = True
            conclusion_reason = "Statistically significant timing measurement."

        # 5. Numerical correctness verification
        orig_out_res = orig_executor.execute(inputs, record_intermediates=True)
        opt_out_res = opt_executor.execute(inputs, record_intermediates=True)

        diffs: List[np.ndarray] = []
        is_allclose = True

        for out_id in original_graph.outputs:
            if out_id in orig_out_res.outputs and out_id in opt_out_res.outputs:
                v_orig = orig_out_res.outputs[out_id]
                v_opt = opt_out_res.outputs[out_id]
                d = np.abs(v_orig - v_opt)
                diffs.append(d)
                if not np.allclose(v_orig, v_opt, atol=1e-5, rtol=1e-5):
                    is_allclose = False

        if diffs:
            flat_diffs = np.concatenate([d.flatten() for d in diffs])
            max_diff = float(np.max(flat_diffs))
            mean_diff = float(np.mean(flat_diffs))
        else:
            max_diff = 0.0
            mean_diff = 0.0

        # 6. Node counts & compilation metadata
        orig_node_count = len(original_graph.nodes)
        opt_node_count = len(optimized_graph.nodes)
        nodes_removed = orig_node_count - opt_node_count
        nodes_fused = pipeline_result.total_nodes_fused if pipeline_result else 0
        comp_time = pipeline_result.compilation_time_ms if pipeline_result else 0.0

        return BenchmarkMetrics(
            original_time_ms=orig_median,
            optimized_time_ms=opt_median,
            original_std_ms=orig_std,
            optimized_std_ms=opt_std,
            abs_time_diff_ms=abs_diff,
            speedup_percent=speedup_pct,
            is_slowdown=is_slowdown,
            is_conclusive=is_conclusive,
            conclusion_reason=conclusion_reason,
            original_node_count=orig_node_count,
            optimized_node_count=opt_node_count,
            nodes_removed=max(0, nodes_removed),
            nodes_fused=nodes_fused,
            compilation_time_ms=comp_time,
            max_abs_numerical_diff=max_diff,
            mean_abs_numerical_diff=mean_diff,
            is_numerically_identical=is_allclose,
            iterations=iterations,
            warmup_runs=warmup,
            original_run_times=orig_times_ms,
            optimized_run_times=opt_times_ms,
        )
