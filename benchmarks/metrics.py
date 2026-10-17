"""Benchmark metrics and diagnostic statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd


@dataclass
class BenchmarkMetrics:
    """Diagnostic comparison metrics between original and optimized graphs."""

    original_time_ms: float
    optimized_time_ms: float
    original_std_ms: float
    optimized_std_ms: float
    abs_time_diff_ms: float
    speedup_percent: float
    is_slowdown: bool
    is_conclusive: bool
    conclusion_reason: str

    original_node_count: int
    optimized_node_count: int
    nodes_removed: int
    nodes_fused: int

    compilation_time_ms: float
    max_abs_numerical_diff: float
    mean_abs_numerical_diff: float
    is_numerically_identical: bool

    iterations: int
    warmup_runs: int
    original_run_times: List[float] = field(default_factory=list)
    optimized_run_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a flat dictionary for display or JSON export."""
        return {
            "Original Time (ms)": round(self.original_time_ms, 4),
            "Optimized Time (ms)": round(self.optimized_time_ms, 4),
            "Absolute Time Diff (ms)": round(self.abs_time_diff_ms, 4),
            "Speedup (%)": round(self.speedup_percent, 2),
            "Status": "Slowdown" if self.is_slowdown else "Speedup",
            "Conclusive": self.is_conclusive,
            "Conclusion Note": self.conclusion_reason,
            "Original Nodes": self.original_node_count,
            "Optimized Nodes": self.optimized_node_count,
            "Nodes Removed": self.nodes_removed,
            "Ops Fused": self.nodes_fused,
            "Compilation Time (ms)": round(self.compilation_time_ms, 4),
            "Max Numerical Diff": f"{self.max_abs_numerical_diff:.2e}",
            "Mean Numerical Diff": f"{self.mean_abs_numerical_diff:.2e}",
            "Numerically Verified": self.is_numerically_identical,
            "Iterations": self.iterations,
        }

    def to_summary_df(self) -> pd.DataFrame:
        """Convert key metrics to a display-ready Pandas DataFrame."""
        data = [
            {"Metric": "Original Execution Time (Median)", "Value": f"{self.original_time_ms:.4f} ms"},
            {"Metric": "Optimized Execution Time (Median)", "Value": f"{self.optimized_time_ms:.4f} ms"},
            {"Metric": "Absolute Difference", "Value": f"{self.abs_time_diff_ms:+.4f} ms"},
            {"Metric": "Speedup / Slowdown", "Value": f"{self.speedup_percent:+.2f}% ({'Slowdown' if self.is_slowdown else 'Speedup'})"},
            {"Metric": "Measurement Validity", "Value": "Conclusive" if self.is_conclusive else f"Inconclusive ({self.conclusion_reason})"},
            {"Metric": "Original Node Count", "Value": str(self.original_node_count)},
            {"Metric": "Optimized Node Count", "Value": str(self.optimized_node_count)},
            {"Metric": "Nodes Removed", "Value": str(self.nodes_removed)},
            {"Metric": "Operations Fused", "Value": str(self.nodes_fused)},
            {"Metric": "Compilation & Pass Overhead", "Value": f"{self.compilation_time_ms:.4f} ms"},
            {"Metric": "Max Absolute Numerical Error", "Value": f"{self.max_abs_numerical_diff:.2e}"},
            {"Metric": "Numerical Equivalence (atol=1e-5)", "Value": "PASSED (Exact)" if self.is_numerically_identical else "FAILED (Discrepancy)"},
        ]
        return pd.DataFrame(data)
