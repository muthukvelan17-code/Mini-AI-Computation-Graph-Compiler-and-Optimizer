"""Benchmark comparison charts using Plotly."""

from __future__ import annotations
import plotly.graph_objects as go
from benchmarks.metrics import BenchmarkMetrics


def plot_execution_time_comparison(metrics: BenchmarkMetrics) -> go.Figure:
    """Bar chart comparing baseline vs optimized execution times with error bars."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline (Original)",
            x=["Original Graph"],
            y=[metrics.original_time_ms],
            error_y=dict(type="data", array=[metrics.original_std_ms], visible=True),
            marker_color="#ef4444",
            text=[f"{metrics.original_time_ms:.4f} ms"],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Optimized",
            x=["Optimized Graph"],
            y=[metrics.optimized_time_ms],
            error_y=dict(type="data", array=[metrics.optimized_std_ms], visible=True),
            marker_color="#10b981",
            text=[f"{metrics.optimized_time_ms:.4f} ms"],
            textposition="auto",
        )
    )

    speedup_text = (
        f"Speedup: {metrics.speedup_percent:+.2f}%"
        if not metrics.is_slowdown
        else f"Slowdown: {metrics.speedup_percent:+.2f}%"
    )

    fig.update_layout(
        title=f"Execution Time Comparison ({speedup_text})",
        yaxis_title="Median Execution Latency (ms)",
        template="plotly_white",
        height=380,
        barmode="group",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_latency_distribution(metrics: BenchmarkMetrics) -> go.Figure:
    """Boxplot displaying latency distributions across benchmark iterations."""
    fig = go.Figure()

    if metrics.original_run_times:
        fig.add_trace(
            go.Box(
                y=metrics.original_run_times,
                name="Original Graph",
                marker_color="#ef4444",
                boxpoints="outliers",
            )
        )

    if metrics.optimized_run_times:
        fig.add_trace(
            go.Box(
                y=metrics.optimized_run_times,
                name="Optimized Graph",
                marker_color="#10b981",
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=f"Latency Distribution ({metrics.iterations} Iterations)",
        yaxis_title="Latency (ms)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_node_reduction_chart(metrics: BenchmarkMetrics) -> go.Figure:
    """Bar chart illustrating node count changes before and after optimization."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Original Nodes", "Optimized Nodes", "Nodes Removed", "Ops Fused"],
            y=[
                metrics.original_node_count,
                metrics.optimized_node_count,
                metrics.nodes_removed,
                metrics.nodes_fused,
            ],
            marker_color=["#3b82f6", "#10b981", "#f59e0b", "#ec4899"],
            text=[
                str(metrics.original_node_count),
                str(metrics.optimized_node_count),
                str(metrics.nodes_removed),
                str(metrics.nodes_fused),
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Computation Graph Node Count Transformation",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
