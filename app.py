"""Mini AI Computation Graph Compiler and Optimizer - Streamlit Dashboard.

An interactive web interface demonstrating tensor computation graph representation,
topological scheduling, compiler optimization passes, execution, and benchmarking.
"""

from __future__ import annotations
import io
import json
import time
import subprocess
import sys
import numpy as np
import pandas as pd
import streamlit as st

from compiler.node import Node, OpType
from compiler.graph import ComputationGraph
from compiler.shape_inference import infer_shapes, ShapeInferenceError
from compiler.executor import GraphExecutor, ExecutionError
from compiler.compiler_pipeline import CompilerPipeline, PipelineResult
from compiler.ir import generate_ir, get_ir_table
from compiler.serialization import serialize_graph, deserialize_graph
from compiler.pytorch_comparison import is_pytorch_available, run_pytorch_benchmark
from optimizations import (
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    AlgebraicSimplificationPass,
    OperatorFusionPass,
)
from benchmarks.benchmark_runner import BenchmarkRunner
from benchmarks.metrics import BenchmarkMetrics
from examples.example_graphs import EXAMPLES, get_example
from visualization.graph_visualizer import create_plotly_graph_figure
from visualization.charts import (
    plot_execution_time_comparison,
    plot_latency_distribution,
    plot_node_reduction_chart,
)

# Page configuration
st.set_page_config(
    page_title="Mini AI Compiler & Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "current_graph" not in st.session_state:
        g, inputs, desc = get_example("6. AI Computation (ReLU((X @ W) + b))")
        infer_shapes(g)
        st.session_state["current_graph"] = g
        st.session_state["graph_inputs"] = inputs
        st.session_state["graph_description"] = desc
        st.session_state["selected_example_name"] = "6. AI Computation (ReLU((X @ W) + b))"

    if "pipeline_result" not in st.session_state:
        # Pre-compile baseline graph
        pipeline = CompilerPipeline.default_pipeline()
        st.session_state["pipeline_result"] = pipeline.compile(st.session_state["current_graph"])

    if "benchmark_metrics" not in st.session_state:
        st.session_state["benchmark_metrics"] = None

    if "last_exec_results" not in st.session_state:
        st.session_state["last_exec_results"] = None


init_session_state()

# Sidebar: Global controls and graph status
with st.sidebar:
    st.markdown("## ⚡ Mini AI Compiler")
    st.caption("DAG Optimization & Execution Engine")
    st.markdown("---")

    st.subheader("⚡ Quick Load Preset")
    example_options = list(EXAMPLES.keys())
    current_idx = (
        example_options.index(st.session_state["selected_example_name"])
        if st.session_state["selected_example_name"] in example_options
        else 5
    )
    selected_preset = st.selectbox("Select Preset Graph:", example_options, index=current_idx)

    if st.button("Load Preset Graph", use_container_width=True, type="primary"):
        g, inputs, desc = get_example(selected_preset)
        infer_shapes(g)
        st.session_state["current_graph"] = g
        st.session_state["graph_inputs"] = inputs
        st.session_state["graph_description"] = desc
        st.session_state["selected_example_name"] = selected_preset
        
        # Auto-compile new graph
        pipeline = CompilerPipeline.default_pipeline()
        st.session_state["pipeline_result"] = pipeline.compile(g)
        st.session_state["last_exec_results"] = None
        st.session_state["benchmark_metrics"] = None
        st.toast(f"Loaded: {selected_preset}", icon="✅")
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Graph Summary")
    curr_g = st.session_state["current_graph"]
    st.markdown(f"**Name:** `{curr_g.name}`")
    st.markdown(f"**Nodes:** `{len(curr_g.nodes)}`")
    st.markdown(f"**Outputs:** `{curr_g.outputs}`")

    st.markdown("---")
    st.subheader("🔥 PyTorch Backend")
    if is_pytorch_available():
        st.success("PyTorch Available for Comparison")
    else:
        st.info("PyTorch Optional: Not Installed\nRun: `pip install torch`")

# Header Section
st.markdown('<div class="main-header">⚡ Mini AI Computation Graph Compiler & Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">A modular AI compiler demonstrating DAG Intermediate Representation (IR), '
    'Shape Inference, Dead Code Elimination, Constant Folding, CSE, Algebraic Simplification, '
    'and Operator Fusion with NumPy execution.</div>',
    unsafe_allow_html=True,
)

curr_g = st.session_state["current_graph"]
st.info(
    f"📍 **Currently Loaded Graph:** `{curr_g.name}` ({len(curr_g.nodes)} nodes, {len(curr_g.outputs)} outputs) — "
    f"*{st.session_state.get('graph_description', '')}*\n\n"
    f"👉 **Click on Tab 2 (Graph Builder), Tab 3 (Optimization Passes), or view the live DAG preview below!**"
)

# 7 Main Tabs as specified in requirements
tabs = st.tabs([
    "1. Overview",
    "2. Graph Builder",
    "3. Optimization Passes",
    "4. Execution & Parity",
    "5. Benchmarking",
    "6. Compiler Internals",
    "7. Tests & Documentation",
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.subheader("📌 Project Overview & Architecture")
    st.markdown(
        """
        Deep learning frameworks (like PyTorch 2.0 / TorchDynamo, TensorFlow XLA, and TVM) execute computations 
        not as isolated operator calls, but as a **Computation Graph (DAG)**. Compiling this graph enables 
        global optimizations that reduce memory bandwidth, remove redundant arithmetic, and fuse consecutive operations.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supported Node Types", "9 Core + 4 Fused")
    with col2:
        st.metric("Optimization Passes", "5 Core Passes")
    with col3:
        st.metric("Execution Engine", "NumPy Interpreter")
    with col4:
        st.metric("Automated Tests", "31 Unit Tests")

    st.markdown("### 🔄 Compiler Pipeline Stages")
    st.info(
        "**Graph Construction** ➔ **Static Shape Inference** ➔ **Common Subexpression Elimination** ➔ "
        "**Constant Folding** ➔ **Algebraic Simplification** ➔ **Dead Code Elimination** ➔ "
        "**Operator Fusion** ➔ **NumPy Interpretation / Execution**"
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### ✨ Implemented Optimization Passes")
        st.markdown(
            """
            1. **Constant Folding:** Evaluates operations with constant operands at compile time.
            2. **Dead Code Elimination (DCE):** Prunes computations that do not contribute to terminal graph outputs.
            3. **Common Subexpression Elimination (CSE):** Detects repeated identical subexpressions and reuses evaluated tensors.
            4. **Algebraic Simplification:** Applies algebraic identities ($x + 0 \\to x$, $x \\times 1 \\to x$, $x \\times 0 \\to 0$).
            5. **Operator Fusion:** Combines eligible consecutive kernels (e.g. `a * b + c` $\\to$ `FusedMulAdd`, `(X @ W) + b` $\\to$ `FusedLinear`, `ReLU((X @ W) + b)` $\\to$ `FusedLinearReLU`).
            """
        )
    with col_right:
        st.markdown("#### 🎯 Engineering Principles")
        st.markdown(
            """
            - **Strict Numerical Parity:** Every transformation guarantees mathematical equivalence within float32 tolerance.
            - **Safe Transformations:** Original graphs remain immutable; optimizations operate on explicit cloned copies.
            - **Zero Magic Benchmarks:** Timing reports actual median durations and explicitly labels microsecond noise as inconclusive.
            - **Pure Python & NumPy Core:** Runs with zero compulsory deep learning dependencies.
            """
        )

    st.markdown("---")
    st.markdown(f"### 🌐 Live Graph Visualization: **`{curr_g.name}`** ({len(curr_g.nodes)} nodes)")
    st.caption(f"Visualizing current graph structure. Switch to **Tab 2 (Graph Builder)** to edit, or **Tab 3 (Optimization Passes)** to optimize.")
    fig_overview = create_plotly_graph_figure(curr_g, title=f"DAG: {curr_g.name}")
    st.plotly_chart(fig_overview, use_container_width=True)

# ==============================================================================
# TAB 2: GRAPH BUILDER
# ==============================================================================
with tabs[1]:
    st.subheader("🛠️ Computation Graph Builder")
    st.caption("Inspect the active graph, configure inputs and operations, or import/export via JSON.")

    col_graph, col_controls = st.columns([1.6, 1.0])

    with col_controls:
        st.markdown("#### 📂 Load or Modify Graph")
        st.write(f"**Current Preset:** {st.session_state.get('selected_example_name', 'Custom')}")
        st.write(f"*{st.session_state.get('graph_description', '')}*")

        with st.expander("➕ Add Custom Operation Node"):
            new_op = st.selectbox("Operation Type", [op.value for op in OpType if not op.value.startswith("fused_")])
            current_node_ids = list(st.session_state["current_graph"].nodes.keys())
            
            if new_op in ("relu", "sigmoid", "scalar_mul"):
                parent_1 = st.selectbox("Select Input Node", current_node_ids, key="custom_p1")
                custom_scalar = st.number_input("Scalar (for scalar_mul)", value=1.0) if new_op == "scalar_mul" else None
                if st.button("Add Unary Node"):
                    g = st.session_state["current_graph"]
                    if new_op == "scalar_mul":
                        g.scalar_mul(parent_1, custom_scalar)
                    elif new_op == "relu":
                        g.relu(parent_1)
                    else:
                        g.sigmoid(parent_1)
                    infer_shapes(g)
                    st.rerun()
            elif new_op in ("add", "sub", "mul", "matmul"):
                parent_1 = st.selectbox("Select Left Input (A)", current_node_ids, key="custom_p1_bin")
                parent_2 = st.selectbox("Select Right Input (B)", current_node_ids, key="custom_p2_bin")
                if st.button("Add Binary Node"):
                    g = st.session_state["current_graph"]
                    g.add_op(new_op, [parent_1, parent_2])
                    infer_shapes(g)
                    st.rerun()

        with st.expander("🏷️ Manage Outputs"):
            all_nids = list(st.session_state["current_graph"].nodes.keys())
            curr_outs = st.session_state["current_graph"].outputs
            selected_outs = st.multiselect("Terminal Output Nodes", all_nids, default=curr_outs)
            if st.button("Update Outputs"):
                if selected_outs:
                    st.session_state["current_graph"].set_outputs(selected_outs)
                    st.success("Outputs updated!")
                    st.rerun()
                else:
                    st.error("Graph must have at least one marked output.")

        with st.expander("💾 JSON Graph Export / Import"):
            json_export = serialize_graph(st.session_state["current_graph"])
            st.download_button(
                "Download Graph JSON",
                data=json_export,
                file_name=f"{st.session_state['current_graph'].name}.json",
                mime="application/json",
                use_container_width=True,
            )
            uploaded_file = st.file_uploader("Upload Graph JSON", type=["json"])
            if uploaded_file is not None:
                try:
                    loaded_g = deserialize_graph(uploaded_file.read().decode("utf-8"))
                    infer_shapes(loaded_g)
                    st.session_state["current_graph"] = loaded_g
                    st.session_state["selected_example_name"] = "Custom Imported"
                    st.session_state["pipeline_result"] = CompilerPipeline.default_pipeline().compile(loaded_g)
                    st.success("Imported graph successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import error: {e}")

    with col_graph:
        st.markdown("#### 🌐 Active Computation Graph (DAG)")
        fig = create_plotly_graph_figure(st.session_state["current_graph"], title=f"Baseline Graph: {st.session_state['current_graph'].name}")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3: OPTIMIZATION PASSES
# ==============================================================================
with tabs[2]:
    st.subheader("⚙️ Compiler Optimization Passes")
    st.caption("Select compiler passes, execute transformations, and inspect the optimization ledger.")

    col_opt_ctrl, col_opt_view = st.columns([1.0, 1.8])

    with col_opt_ctrl:
        st.markdown("#### 🛠️ Configure Passes")
        pass_cf = st.checkbox("Constant Folding", value=True)
        pass_dce = st.checkbox("Dead Code Elimination", value=True)
        pass_cse = st.checkbox("Common Subexpression Elimination", value=True)
        pass_alg = st.checkbox("Algebraic Simplification", value=True)
        pass_fuse = st.checkbox("Operator Fusion", value=True)

        selected_pass_objs = []
        if pass_cse:
            selected_pass_objs.append(CommonSubexpressionEliminationPass())
        if pass_cf:
            selected_pass_objs.append(ConstantFoldingPass())
        if pass_alg:
            selected_pass_objs.append(AlgebraicSimplificationPass())
        if pass_dce:
            selected_pass_objs.append(DeadCodeEliminationPass())
        if pass_fuse:
            selected_pass_objs.append(OperatorFusionPass())
            if pass_dce:
                selected_pass_objs.append(DeadCodeEliminationPass())

        if st.button("🚀 Run Compiler Passes", type="primary", use_container_width=True):
            pipeline = CompilerPipeline(selected_pass_objs)
            pipe_res = pipeline.compile(st.session_state["current_graph"])
            st.session_state["pipeline_result"] = pipe_res
            st.session_state["benchmark_metrics"] = None
            st.session_state["last_exec_results"] = None
            st.success("Compilation passes applied successfully!")

        pipe_res: PipelineResult = st.session_state.get("pipeline_result")
        if pipe_res:
            st.markdown("---")
            st.markdown("#### 📊 Transformation Summary")
            orig_n = len(pipe_res.original_graph.nodes)
            opt_n = len(pipe_res.optimized_graph.nodes)
            diff_n = orig_n - opt_n

            c1, c2 = st.columns(2)
            c1.metric("Original Nodes", orig_n)
            c2.metric("Optimized Nodes", opt_n, delta=f"{-diff_n} nodes" if diff_n != 0 else "0")

            c3, c4 = st.columns(2)
            c3.metric("Nodes Removed", pipe_res.total_nodes_removed)
            c4.metric("Fused Kernels", pipe_res.total_nodes_fused)

            st.metric("Compilation Time", f"{pipe_res.compilation_time_ms:.3f} ms")

    with col_opt_view:
        pipe_res = st.session_state.get("pipeline_result")
        if pipe_res:
            st.markdown("#### 🌐 Optimized Graph Visualizer")
            fig_opt = create_plotly_graph_figure(
                pipe_res.optimized_graph,
                title=f"Optimized Graph: {pipe_res.optimized_graph.name} ({len(pipe_res.optimized_graph.nodes)} nodes)",
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            st.markdown("#### 📜 Optimization Transformation Log")
            if pipe_res.full_log:
                log_box = "\n".join(pipe_res.full_log)
                st.code(log_box, language="text")
            else:
                st.info("No transformations triggered by active passes.")

# ==============================================================================
# TAB 4: EXECUTION & PARITY
# ==============================================================================
with tabs[3]:
    st.subheader("▶️ Graph Execution & Numerical Correctness")
    st.caption("Interpret the baseline and optimized graphs with NumPy and verify exact floating-point parity.")

    col_exec_inputs, col_exec_outputs = st.columns([1.0, 1.8])
    pipe_res = st.session_state.get("pipeline_result")
    curr_g = st.session_state["current_graph"]
    opt_g = pipe_res.optimized_graph if pipe_res else curr_g

    with col_exec_inputs:
        st.markdown("#### 📥 Input Tensors")
        input_nodes = curr_g.get_input_nodes()

        if not input_nodes:
            st.info("This graph contains only constant tensors.")
        else:
            st.write(f"Configured Inputs ({len(input_nodes)} tensors):")
            inputs_dict = st.session_state.get("graph_inputs", {})
            for inp_node in input_nodes:
                val = inputs_dict.get(inp_node.id)
                shape_str = str(val.shape) if val is not None else str(inp_node.shape)
                st.markdown(f"- **%{inp_node.id}** ({inp_node.name}): `shape={shape_str}`, `dtype={inp_node.dtype}`")

        input_scale = st.slider("Input Matrix Scale / Dimension Multiplier", 1, 10, 1, help="Scales input dimensions to demonstrate larger tensor workloads.")

        if st.button("Execute Both Graphs", type="primary", use_container_width=True):
            try:
                # Prepare inputs (scale if requested)
                base_inputs = st.session_state.get("graph_inputs", {})
                active_inputs = {}
                rng = np.random.default_rng(123)

                for inp_node in curr_g.get_input_nodes():
                    orig_val = base_inputs.get(inp_node.id)
                    if orig_val is not None:
                        if input_scale > 1 and orig_val.ndim >= 2:
                            # Tile or scale shape
                            new_shape = list(orig_val.shape)
                            new_shape[0] *= input_scale
                            if len(new_shape) > 1:
                                new_shape[1] *= input_scale
                            active_inputs[inp_node.id] = rng.uniform(0.5, 2.0, size=tuple(new_shape)).astype(np.float32)
                        else:
                            active_inputs[inp_node.id] = orig_val.copy()
                    else:
                        shape = inp_node.shape or (4, 4)
                        active_inputs[inp_node.id] = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)

                orig_exec = GraphExecutor(curr_g).execute(active_inputs, record_intermediates=True)
                opt_exec = GraphExecutor(opt_g).execute(active_inputs, record_intermediates=True)

                st.session_state["last_exec_results"] = (orig_exec, opt_exec, active_inputs)
                st.success("Execution complete!")
            except Exception as e:
                st.error(f"Execution Error: {e}")

    with col_exec_outputs:
        exec_data = st.session_state.get("last_exec_results")
        if exec_data:
            orig_res, opt_res, active_inputs = exec_data
            st.markdown("#### 🎯 Execution Diagnostic Parity Check")

            # Numerical comparisons
            all_diffs = []
            is_parity_ok = True

            for idx, out_id in enumerate(curr_g.outputs):
                opt_out_id = opt_g.outputs[idx] if idx < len(opt_g.outputs) else None
                if out_id in orig_res.outputs and opt_out_id in opt_res.outputs:
                    orig_arr = orig_res.outputs[out_id]
                    opt_arr = opt_res.outputs[opt_out_id]

                    diff = np.abs(orig_arr - opt_arr)
                    all_diffs.append(diff)
                    if not np.allclose(orig_arr, opt_arr, atol=1e-5, rtol=1e-5):
                        is_parity_ok = False

            if all_diffs:
                flat_diffs = np.concatenate([d.flatten() for d in all_diffs])
                max_err = float(np.max(flat_diffs))
                mean_err = float(np.mean(flat_diffs))
            else:
                max_err, mean_err = 0.0, 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Numerical Status", "EXACT MATCH ✅" if is_parity_ok else "DISCREPANCY ❌")
            m2.metric("Max Absolute Error", f"{max_err:.2e}")
            m3.metric("Mean Absolute Error", f"{mean_err:.2e}")

            st.markdown("##### 📄 Output Tensors Sample")
            for idx, out_id in enumerate(curr_g.outputs):
                opt_out_id = opt_g.outputs[idx] if idx < len(opt_g.outputs) else None
                orig_arr = orig_res.outputs[out_id]
                opt_arr = opt_res.outputs[opt_out_id]

                with st.expander(f"Output %{out_id} (Shape: {orig_arr.shape})", expanded=True):
                    c_orig, c_opt = st.columns(2)
                    with c_orig:
                        st.caption("Baseline Graph Output")
                        st.write(orig_arr[:4] if orig_arr.ndim > 0 else orig_arr)
                    with c_opt:
                        st.caption("Optimized Graph Output")
                        st.write(opt_arr[:4] if opt_arr.ndim > 0 else opt_arr)

            # Optional PyTorch Comparison Section
            st.markdown("---")
            st.markdown("#### 🔬 Real Framework Comparison (PyTorch)")
            if is_pytorch_available():
                if st.button("Run PyTorch Comparison Benchmark"):
                    with st.spinner("Executing in PyTorch..."):
                        pt_res = run_pytorch_benchmark(curr_g, active_inputs, orig_res.outputs)
                        if pt_res.available:
                            c_pt1, c_pt2, c_pt3 = st.columns(3)
                            c_pt1.metric("PyTorch Median Latency", f"{pt_res.execution_time_ms:.4f} ms")
                            c_pt2.metric("Max Error vs NumPy", f"{pt_res.max_abs_diff_vs_numpy:.2e}")
                            c_pt3.metric("Parity Check", "PASSED" if pt_res.is_numerically_close else "FAILED")
                            st.caption(
                                "Note: This is an educational mini-compiler. Industrial compilers (like PyTorch Inductor / "
                                "TVM / XLA) generate lowered C++/CUDA machine code."
                            )
            else:
                st.info(
                    "Optional PyTorch backend is not currently installed. To enable direct PyTorch comparisons, install it via:\n"
                    "`pip install torch`"
                )
        else:
            st.info("Click 'Execute Both Graphs' to run interpretation and verify output equivalence.")

# ==============================================================================
# TAB 5: BENCHMARKING
# ==============================================================================
with tabs[4]:
    st.subheader("⏱️ Rigorous Performance Benchmarking")
    st.caption("Benchmark baseline vs optimized graphs over multiple iterations with warmups and statistical analysis.")

    pipe_res = st.session_state.get("pipeline_result")
    curr_g = st.session_state["current_graph"]
    opt_g = pipe_res.optimized_graph if pipe_res else curr_g

    b_col_ctrl, b_col_view = st.columns([1.0, 2.0])

    with b_col_ctrl:
        st.markdown("#### ⚙️ Benchmark Parameters")
        iterations = st.slider("Iterations", 10, 200, 50, step=10)
        warmup = st.slider("Warmup Runs", 2, 20, 5)
        tensor_scale = st.selectbox("Tensor Dimension Scale", [1, 2, 4, 8], index=1, help="Enlarges matrices to reduce Python interpreter overhead and increase compute intensity.")

        if st.button("🏁 Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Benchmarking in progress..."):
                base_inputs = st.session_state.get("graph_inputs", {})
                active_inputs = {}
                rng = np.random.default_rng(42)

                for inp_node in curr_g.get_input_nodes():
                    orig_val = base_inputs.get(inp_node.id)
                    if orig_val is not None:
                        new_shape = list(orig_val.shape)
                        if len(new_shape) >= 2:
                            new_shape[0] *= tensor_scale
                            new_shape[1] *= tensor_scale
                        active_inputs[inp_node.id] = rng.uniform(0.5, 2.0, size=tuple(new_shape)).astype(np.float32)
                    else:
                        shape = inp_node.shape or (16 * tensor_scale, 16 * tensor_scale)
                        active_inputs[inp_node.id] = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)

                metrics = BenchmarkRunner.run_benchmark(
                    curr_g,
                    opt_g,
                    active_inputs,
                    iterations=iterations,
                    warmup=warmup,
                    pipeline_result=pipe_res,
                )
                st.session_state["benchmark_metrics"] = metrics
                st.success("Benchmark completed!")

        metrics: BenchmarkMetrics = st.session_state.get("benchmark_metrics")
        if metrics:
            st.markdown("---")
            st.markdown("#### 📤 Export Results")
            metrics_dict = metrics.to_dict()
            json_str = json.dumps(metrics_dict, indent=2)
            st.download_button(
                "Download Benchmark JSON",
                data=json_str,
                file_name=f"benchmark_{curr_g.name}.json",
                mime="application/json",
                use_container_width=True,
            )
            csv_str = pd.DataFrame([metrics_dict]).to_csv(index=False)
            st.download_button(
                "Download Benchmark CSV",
                data=csv_str,
                file_name=f"benchmark_{curr_g.name}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with b_col_view:
        metrics: BenchmarkMetrics = st.session_state.get("benchmark_metrics")
        if metrics:
            st.markdown("#### 📈 Benchmark Results")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Original Time", f"{metrics.original_time_ms:.4f} ms")
            m_col2.metric("Optimized Time", f"{metrics.optimized_time_ms:.4f} ms")
            
            speedup_label = "Speedup" if not metrics.is_slowdown else "Slowdown"
            m_col3.metric(
                speedup_label,
                f"{metrics.speedup_percent:+.2f}%",
                delta=f"{metrics.abs_time_diff_ms:+.4f} ms",
                delta_color="normal" if not metrics.is_slowdown else "inverse",
            )

            if not metrics.is_conclusive:
                st.warning(f"⚠️ **Inconclusive Result**: {metrics.conclusion_reason}")
            else:
                st.success(f"✅ **Conclusive Result**: {metrics.conclusion_reason}")

            # Visual charts
            c_fig1, c_fig2 = st.columns(2)
            with c_fig1:
                st.plotly_chart(plot_execution_time_comparison(metrics), use_container_width=True)
            with c_fig2:
                st.plotly_chart(plot_latency_distribution(metrics), use_container_width=True)

            st.plotly_chart(plot_node_reduction_chart(metrics), use_container_width=True)

            st.markdown("#### 📋 Detailed Metrics Breakdown")
            st.table(metrics.to_summary_df())
        else:
            st.info("Click 'Run Benchmark' to execute repeated timed runs.")

# ==============================================================================
# TAB 6: COMPILER INTERNALS
# ==============================================================================
with tabs[5]:
    st.subheader("🔍 Compiler Intermediate Representation (IR) & Stage Inspector")
    st.caption("Deep-dive into the intermediate representation, topological execution schedules, and phase diffs.")

    curr_g = st.session_state["current_graph"]
    pipe_res: PipelineResult = st.session_state.get("pipeline_result")

    internals_tab1, internals_tab2, internals_tab3 = st.tabs([
        "Intermediate Representation (IR)",
        "Topological Execution Schedule",
        "Stage-by-Stage Compiler Transformation",
    ])

    with internals_tab1:
        st.markdown("#### 📜 Low-Level IR Code Dump")
        col_ir1, col_ir2 = st.columns(2)
        with col_ir1:
            st.markdown("**Original Graph IR:**")
            st.code(generate_ir(curr_g), language="llvm")
        with col_ir2:
            if pipe_res:
                st.markdown("**Optimized Graph IR:**")
                st.code(generate_ir(pipe_res.optimized_graph), language="llvm")

        st.markdown("#### 📊 Structured IR Instruction Table")
        st.dataframe(pd.DataFrame(get_ir_table(curr_g)), use_container_width=True)

    with internals_tab2:
        st.markdown("#### ⏳ Execution Order (Kahn's Topological Sort)")
        st.write("Topological sorting resolves data dependencies to construct a strictly valid linear execution schedule:")
        order = curr_g.topological_sort()
        steps_data = []
        for idx, nid in enumerate(order, start=1):
            n = curr_g.get_node(nid)
            steps_data.append({
                "Execution Order": idx,
                "Node ID": f"%{nid}",
                "Operation": n.op_type.value if isinstance(n.op_type, OpType) else str(n.op_type),
                "Depends On": ", ".join(f"%{i}" for i in n.inputs) if n.inputs else "None (Root)",
                "Shape": str(list(n.shape)) if n.shape is not None else "Unknown",
            })
        st.dataframe(pd.DataFrame(steps_data), use_container_width=True)

    with internals_tab3:
        if pipe_res and pipe_res.stage_graphs:
            st.markdown("#### 🎞️ Compiler Pass Evolution")
            stage_names = [f"Stage {i}: {name}" for i, (name, _) in enumerate(pipe_res.stage_graphs)]
            selected_stage_idx = st.slider("Select Pipeline Stage:", 0, len(stage_names) - 1, 0, format="%d")
            
            stage_name, stage_graph = pipe_res.stage_graphs[selected_stage_idx]
            st.markdown(f"**Viewing: {stage_names[selected_stage_idx]}** ({len(stage_graph.nodes)} nodes)")
            
            st.plotly_chart(
                create_plotly_graph_figure(stage_graph, title=f"DAG at {stage_name}"),
                use_container_width=True,
            )
        else:
            st.info("Run compiler passes to view stage-by-stage progression.")

# ==============================================================================
# TAB 7: TESTS & DOCUMENTATION
# ==============================================================================
with tabs[6]:
    st.subheader("🧪 Automated Test Suite & Architecture Docs")
    st.caption("Verify compiler correctness directly from the dashboard and inspect documentation.")

    col_test_runner, col_docs = st.columns([1.2, 1.0])

    with col_test_runner:
        st.markdown("#### 🏃 Live Test Runner")
        if st.button("Execute PyTest Suite", type="primary", use_container_width=True):
            with st.spinner("Running automated pytest suite..."):
                t0 = time.perf_counter()
                cmd = [sys.executable, "-m", "pytest", "-v"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                duration = time.perf_counter() - t0

                if proc.returncode == 0:
                    st.success(f"All tests passed in {duration:.2f}s! 🎉")
                else:
                    st.error(f"Tests failed with returncode {proc.returncode}")

                st.code(proc.stdout, language="text")

        st.markdown("#### 💻 Windows PowerShell Run Instructions")
        st.code(
            """# 1. Activate Virtual Environment
.\\.venv\\Scripts\\Activate.ps1

# 2. Run Test Suite
python -m pytest -v

# 3. Launch Streamlit Dashboard
streamlit run app.py
""",
            language="powershell",
        )

    with col_docs:
        st.markdown("#### 📚 Architecture & Modules")
        st.markdown(
            """
            - **`compiler/node.py`**: Node primitives, op types, and metadata.
            - **`compiler/graph.py`**: DAG representation, cycle detection, topological sort.
            - **`compiler/shape_inference.py`**: Static shape inference with broadcasting.
            - **`compiler/executor.py`**: Vectorized NumPy interpreter with intermediate caching.
            - **`compiler/compiler_pipeline.py`**: Multi-pass compiler orchestrator.
            - **`optimizations/`**: Dedicated pass implementations (CF, DCE, CSE, Simplification, Fusion).
            - **`benchmarks/`**: Statistical benchmarking, noise analysis, CSV/JSON export.
            - **`visualization/`**: Interactive Plotly DAG and performance chart generation.
            """
        )

st.markdown("---")
st.caption("Mini AI Computation Graph Compiler and Optimizer | Pair-Programmed by Senior AI Systems Engineer")
