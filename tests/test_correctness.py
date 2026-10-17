"""End-to-end correctness and pipeline test across all built-in examples."""

import pytest
import numpy as np

from compiler.compiler_pipeline import CompilerPipeline
from compiler.executor import GraphExecutor
from benchmarks.benchmark_runner import BenchmarkRunner
from examples.example_graphs import EXAMPLES


@pytest.mark.parametrize("example_name", list(EXAMPLES.keys()))
def test_all_examples_pipeline_correctness(example_name):
    """Test that every built-in example compiles and executes with exact numerical parity."""
    builder = EXAMPLES[example_name]
    graph, inputs, desc = builder()

    pipeline = CompilerPipeline.default_pipeline()
    pipeline_result = pipeline.compile(graph)
    opt_graph = pipeline_result.optimized_graph

    # Execute original and optimized
    orig_res = GraphExecutor(graph).execute(inputs)
    opt_res = GraphExecutor(opt_graph).execute(inputs)

    # Check each output node
    for out_id in graph.outputs:
        # In case the output was replaced or renamed by the pass, check corresponding output
        assert len(opt_graph.outputs) == len(graph.outputs)
        orig_arr = orig_res.outputs[out_id]
        
        # Match output by index or key
        idx = graph.outputs.index(out_id)
        opt_out_id = opt_graph.outputs[idx]
        opt_arr = opt_res.outputs[opt_out_id]

        assert orig_arr.shape == opt_arr.shape
        np.testing.assert_allclose(
            orig_arr,
            opt_arr,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"Parity mismatch in {example_name} between {out_id} and {opt_out_id}",
        )

    # Test benchmark runner completes without errors
    metrics = BenchmarkRunner.run_benchmark(
        graph, opt_graph, inputs, iterations=10, warmup=2, pipeline_result=pipeline_result
    )
    assert metrics.is_numerically_identical
    assert metrics.original_node_count >= metrics.optimized_node_count
