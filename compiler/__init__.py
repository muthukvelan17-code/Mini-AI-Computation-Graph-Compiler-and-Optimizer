"""Mini AI Computation Graph Compiler and Optimizer.

Core compiler package containing Node definitions, ComputationGraph DAG,
shape inference, intermediate representation (IR), serialization, and NumPy executor.
"""

from compiler.node import Node, OpType
from compiler.graph import ComputationGraph
from compiler.shape_inference import infer_shapes, ShapeInferenceError
from compiler.executor import GraphExecutor, ExecutionResult
from compiler.ir import generate_ir, print_ir
from compiler.serialization import serialize_graph, deserialize_graph
from compiler.compiler_pipeline import CompilerPipeline

__all__ = [
    "Node",
    "OpType",
    "ComputationGraph",
    "infer_shapes",
    "ShapeInferenceError",
    "GraphExecutor",
    "ExecutionResult",
    "generate_ir",
    "print_ir",
    "serialize_graph",
    "deserialize_graph",
    "CompilerPipeline",
]
