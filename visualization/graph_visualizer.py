"""Graph visualization using NetworkX, Plotly, and Matplotlib."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import OpType


def _get_node_color(node) -> str:
    """Determine visual color for a node based on its operation type."""
    if node.is_input:
        return "#3b82f6"  # Blue
    if node.is_constant:
        return "#f59e0b"  # Amber / Orange
    if node.is_fused:
        return "#ec4899"  # Pink / Magenta
    if node.op_type == OpType.MATMUL:
        return "#6366f1"  # Indigo
    if node.op_type in (OpType.RELU, OpType.SIGMOID):
        return "#8b5cf6"  # Violet / Purple
    return "#10b981"      # Emerald Green for standard arithmetic


def _compute_hierarchical_layout(graph: ComputationGraph) -> Dict[str, Tuple[float, float]]:
    """Compute hierarchical coordinates (x, y) for nodes in the DAG."""
    try:
        order = graph.topological_sort()
    except Exception:
        order = list(graph.nodes.keys())

    # Calculate depth / layer for each node (longest path from any root)
    depth: Dict[str, int] = {}
    for nid in order:
        node = graph.get_node(nid)
        if not node.inputs:
            depth[nid] = 0
        else:
            parent_depths = [depth.get(inp, 0) for inp in node.inputs if inp in depth]
            depth[nid] = (max(parent_depths) + 1) if parent_depths else 0

    # Group nodes by depth layer
    layers: Dict[int, List[str]] = {}
    for nid, d in depth.items():
        layers.setdefault(d, []).append(nid)

    pos: Dict[str, Tuple[float, float]] = {}
    max_d = max(layers.keys()) if layers else 0

    for d, nids in layers.items():
        n_count = len(nids)
        # Y coordinate flows top-to-bottom
        y = 1.0 - (d / max(1, max_d)) if max_d > 0 else 0.5
        for idx, nid in enumerate(nids):
            # X coordinate centered across the layer
            x = (idx + 1) / (n_count + 1)
            pos[nid] = (x, y)

    return pos


def create_plotly_graph_figure(graph: ComputationGraph, title: Optional[str] = None) -> go.Figure:
    """Generate an interactive Plotly graph showing the computation graph DAG."""
    if not graph.nodes:
        fig = go.Figure()
        fig.add_annotation(text="Empty Graph", showarrow=False, font=dict(size=16))
        return fig

    pos = _compute_hierarchical_layout(graph)
    
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    annotations = []

    for nid, node in graph.nodes.items():
        x0, y0 = pos.get(nid, (0.5, 0.5))
        for inp in node.inputs:
            if inp in pos:
                x_p, y_p = pos[inp]
                edge_x.extend([x_p, x0, None])
                edge_y.extend([y_p, y0, None])
                # Add an arrow annotation along the edge
                annotations.append(
                    dict(
                        ax=x_p,
                        ay=y_p,
                        axref="x",
                        ayref="y",
                        x=(x_p + x0) / 2,
                        y=(y_p + y0) / 2,
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor="#94a3b8",
                    )
                )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#cbd5e1"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_borders = []
    node_border_widths = []
    hover_texts = []

    for nid in graph.nodes:
        node = graph.get_node(nid)
        x, y = pos.get(nid, (0.5, 0.5))
        node_x.append(x)
        node_y.append(y)

        op_str = node.op_type.value if isinstance(node.op_type, OpType) else str(node.op_type)
        is_output = nid in graph.outputs

        # Border styling: outputs have gold border
        border_color = "#eab308" if is_output else "#ffffff"
        border_width = 3.5 if is_output else 1.5

        node_borders.append(border_color)
        node_border_widths.append(border_width)
        node_colors.append(_get_node_color(node))

        # Node text label
        label = f"%{nid}<br><b>{op_str.upper()}</b>"
        node_text.append(label)

        # Hover tooltip
        shape_str = str(list(node.shape)) if node.shape is not None else "Unknown"
        inputs_str = ", ".join(f"%{i}" for i in node.inputs) if node.inputs else "None"
        tooltip = (
            f"<b>Node:</b> %{nid}<br>"
            f"<b>Name:</b> {node.name}<br>"
            f"<b>Operation:</b> {op_str.upper()}<br>"
            f"<b>Shape:</b> {shape_str}<br>"
            f"<b>Dtype:</b> {node.dtype}<br>"
            f"<b>Inputs:</b> {inputs_str}<br>"
            f"<b>Role:</b> {'Output' if is_output else ('Input' if node.is_input else ('Constant' if node.is_constant else 'Intermediate'))}"
        )
        if node.metadata:
            meta_str = "<br>".join(f"  {k}: {v}" for k, v in node.metadata.items())
            tooltip += f"<br><b>Metadata:</b><br>{meta_str}"
        hover_texts.append(tooltip)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_texts,
        text=[f"<b>%{nid}</b>" for nid in graph.nodes],
        textposition="top center",
        textfont=dict(size=11, color="#1e293b"),
        marker=dict(
            size=36,
            color=node_colors,
            line=dict(color=node_borders, width=node_border_widths),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title or f"Computation Graph: {graph.name} ({len(graph.nodes)} nodes)",
                font=dict(size=16),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            annotations=annotations,
            height=450,
        ),
    )
    return fig


def render_matplotlib_graph(graph: ComputationGraph) -> plt.Figure:
    """Generate a clean Matplotlib figure of the DAG for export or fallback."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    G = graph.to_networkx()

    pos = _compute_hierarchical_layout(graph)
    colors = [_get_node_color(graph.get_node(nid)) for nid in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1200, edgecolors="#334155")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#94a3b8", arrows=True, arrowsize=18, width=1.5)
    
    labels = {nid: f"{nid}\n({graph.get_node(nid).op_type})" for nid in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_family="sans-serif")

    ax.set_title(f"DAG: {graph.name} ({len(graph.nodes)} nodes)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig
