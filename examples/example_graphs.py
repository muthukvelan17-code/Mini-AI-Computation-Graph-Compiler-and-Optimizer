"""Built-in demonstration example computation graphs."""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np

from compiler.graph import ComputationGraph
from compiler.node import OpType


def example_constant_folding() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 1: (2 + 3) * x

    Demonstrates Constant Folding:
    2 + 3 is computed at compile time to 5, transforming (2 + 3) * x into 5 * x.
    """
    g = ComputationGraph(name="Constant_Folding_Example")
    # Constants 2 and 3
    c2 = g.add_constant(np.array(2.0, dtype=np.float32), name="const_2")
    c3 = g.add_constant(np.array(3.0, dtype=np.float32), name="const_3")
    # Input x
    x = g.add_input(name="x", shape=(4, 4), dtype="float32")
    # c2 + c3
    add_node = g.add(c2.id, c3.id, name="two_plus_three")
    # (c2 + c3) * x
    mul_node = g.mul(add_node.id, x.id, name="result")
    g.mark_output(mul_node.id)

    # Deterministic test input
    rng = np.random.default_rng(42)
    inputs = {x.id: rng.uniform(1.0, 5.0, size=(4, 4)).astype(np.float32)}
    desc = "Computes (2 + 3) * x. Compile-time constant addition (2+3) is folded into static constant 5."
    return g, inputs, desc


def example_dead_code_elimination() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 2: Dead Code Elimination

    Demonstrates pruning of unused computations:
    y = a * b (used)
    dead_branch = a + c (never used for output)
    """
    g = ComputationGraph(name="Dead_Code_Elimination_Example")
    a = g.add_input(name="a", shape=(3, 3), dtype="float32")
    b = g.add_input(name="b", shape=(3, 3), dtype="float32")
    c = g.add_input(name="c", shape=(3, 3), dtype="float32")

    # Live computation: a * b
    y = g.mul(a.id, b.id, name="live_mul")
    g.mark_output(y.id)

    # Dead computation: a + c and sigmoid(dead_add) (unused)
    dead_add = g.add(a.id, c.id, name="unused_add")
    g.sigmoid(dead_add.id, name="unused_sigmoid")

    rng = np.random.default_rng(42)
    inputs = {
        a.id: rng.uniform(0.5, 2.0, size=(3, 3)).astype(np.float32),
        b.id: rng.uniform(1.0, 3.0, size=(3, 3)).astype(np.float32),
        c.id: rng.uniform(0.1, 1.0, size=(3, 3)).astype(np.float32),
    }
    desc = "Graph contains a live path (a * b) and an unused path (sigmoid(a + c)). DCE prunes the dead nodes."
    return g, inputs, desc


def example_common_subexpression() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 3: Common Subexpression Elimination

    Computes: (a * b) + (a * b)
    The second identical multiplication is detected, pruned, and wired to the first multiplication's output.
    """
    g = ComputationGraph(name="Common_Subexpression_Example")
    a = g.add_input(name="a", shape=(4, 4), dtype="float32")
    b = g.add_input(name="b", shape=(4, 4), dtype="float32")

    mul1 = g.mul(a.id, b.id, name="mul_1")
    mul2 = g.mul(a.id, b.id, name="mul_2")
    res = g.add(mul1.id, mul2.id, name="sum_result")
    g.mark_output(res.id)

    rng = np.random.default_rng(42)
    inputs = {
        a.id: rng.uniform(1.0, 4.0, size=(4, 4)).astype(np.float32),
        b.id: rng.uniform(0.5, 2.0, size=(4, 4)).astype(np.float32),
    }
    desc = "Computes (a * b) + (a * b). CSE detects identical subexpressions and reuses mul_1 for both operands."
    return g, inputs, desc


def example_algebraic_simplification() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 4: Algebraic Simplification

    Computes: (x * 1) + 0
    Multiplication by 1 is simplified to x, then addition of 0 is simplified to x.
    """
    g = ComputationGraph(name="Algebraic_Simplification_Example")
    x = g.add_input(name="x", shape=(3, 3), dtype="float32")
    c1 = g.add_constant(np.array(1.0, dtype=np.float32), name="const_one")
    c0 = g.add_constant(np.array(0.0, dtype=np.float32), name="const_zero")

    mul1 = g.mul(x.id, c1.id, name="mul_by_one")
    add0 = g.add(mul1.id, c0.id, name="add_zero")
    g.mark_output(add0.id)

    rng = np.random.default_rng(42)
    inputs = {x.id: rng.normal(0.0, 1.0, size=(3, 3)).astype(np.float32)}
    desc = "Computes (x * 1) + 0. Multiplicative and additive identities are eliminated, leaving direct reference to x."
    return g, inputs, desc


def example_operator_fusion() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 5: Operator Fusion

    Computes: (a * b) + c
    Fuses element-wise multiplication and addition into a single FusedMulAdd kernel.
    """
    g = ComputationGraph(name="Operator_Fusion_Example")
    a = g.add_input(name="a", shape=(16, 16), dtype="float32")
    b = g.add_input(name="b", shape=(16, 16), dtype="float32")
    c = g.add_input(name="c", shape=(16, 16), dtype="float32")

    mul_node = g.mul(a.id, b.id, name="mul_ab")
    add_node = g.add(mul_node.id, c.id, name="fma_target")
    g.mark_output(add_node.id)

    rng = np.random.default_rng(42)
    inputs = {
        a.id: rng.uniform(-1.0, 1.0, size=(16, 16)).astype(np.float32),
        b.id: rng.uniform(-1.0, 1.0, size=(16, 16)).astype(np.float32),
        c.id: rng.uniform(-1.0, 1.0, size=(16, 16)).astype(np.float32),
    }
    desc = "Computes (a * b) + c. Operator Fusion combines Mul + Add into FusedMulAdd."
    return g, inputs, desc


def example_ai_style_computation() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 6: AI-Style Neural Network Dense Layer

    Computes: ReLU((X @ W) + b)
    X: (8, 16), W: (16, 32), b: (32,)
    Undergoes Operator Fusion: MatMul + Add + ReLU -> FusedLinearReLU.
    """
    g = ComputationGraph(name="AI_Dense_Layer_Example")
    X = g.add_input(name="X", shape=(8, 16), dtype="float32")
    W = g.add_input(name="W", shape=(16, 32), dtype="float32")
    b = g.add_constant(np.zeros((32,), dtype=np.float32), name="bias")

    mm = g.matmul(X.id, W.id, name="dense_gemm")
    add_bias = g.add(mm.id, b.id, name="dense_bias_add")
    relu_out = g.relu(add_bias.id, name="dense_relu_activation")
    g.mark_output(relu_out.id)

    rng = np.random.default_rng(42)
    inputs = {
        X.id: rng.normal(0.0, 1.0, size=(8, 16)).astype(np.float32),
        W.id: (rng.normal(0.0, 1.0, size=(16, 32)) * 0.1).astype(np.float32),
    }
    desc = "AI Dense Layer: ReLU((X @ W) + b). Fuses Matrix Multiply + Bias Add + ReLU activation into a single kernel."
    return g, inputs, desc


def example_two_layer_mlp() -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Example 7: Two-Layer Multi-Layer Perceptron (MLP)

    Computes:
    H1 = Sigmoid((X @ W1) + b1)
    H2 = ReLU((H1 @ W2) + b2)
    Plus dead regularization subexpression.
    """
    g = ComputationGraph(name="Two_Layer_MLP_Example")
    X = g.add_input(name="X", shape=(16, 32), dtype="float32")
    W1 = g.add_input(name="W1", shape=(32, 64), dtype="float32")
    b1 = g.add_constant(np.zeros((64,), dtype=np.float32), name="b1")

    # Layer 1
    mm1 = g.matmul(X.id, W1.id, name="layer1_gemm")
    add1 = g.add(mm1.id, b1.id, name="layer1_add")
    sig1 = g.sigmoid(add1.id, name="layer1_sigmoid")

    # Layer 2
    W2 = g.add_input(name="W2", shape=(64, 10), dtype="float32")
    b2 = g.add_constant(np.zeros((10,), dtype=np.float32), name="b2")
    mm2 = g.matmul(sig1.id, W2.id, name="layer2_gemm")
    add2 = g.add(mm2.id, b2.id, name="layer2_add")
    out = g.relu(add2.id, name="mlp_output")
    g.mark_output(out.id)

    # Dead computation: unused L2 penalty
    dead_w_sq = g.mul(W1.id, W1.id, name="dead_w1_squared")

    rng = np.random.default_rng(42)
    inputs = {
        X.id: rng.normal(0.0, 1.0, size=(16, 32)).astype(np.float32),
        W1.id: (rng.normal(0.0, 1.0, size=(32, 64)) * 0.1).astype(np.float32),
        W2.id: (rng.normal(0.0, 1.0, size=(64, 10)) * 0.1).astype(np.float32),
    }
    desc = "2-Layer MLP: Sigmoid(X@W1+b1) -> ReLU(H1@W2+b2) with dead weight squaring. Tests DCE and multiple fusions."
    return g, inputs, desc


EXAMPLES: Dict[str, Callable[[], Tuple[ComputationGraph, Dict[str, np.ndarray], str]]] = {
    "1. Constant Folding ((2 + 3) * x)": example_constant_folding,
    "2. Dead Code Elimination (unused branch)": example_dead_code_elimination,
    "3. Common Subexpression Elimination ((a*b) + (a*b))": example_common_subexpression,
    "4. Algebraic Simplification ((x * 1) + 0)": example_algebraic_simplification,
    "5. Operator Fusion ((a * b) + c)": example_operator_fusion,
    "6. AI Computation (ReLU((X @ W) + b))": example_ai_style_computation,
    "7. Two-Layer MLP (Deep AI Graph)": example_two_layer_mlp,
}


def get_example(name: str) -> Tuple[ComputationGraph, Dict[str, np.ndarray], str]:
    """Retrieve an example computation graph by name."""
    if name not in EXAMPLES:
        raise KeyError(f"Example {name!r} not found. Available: {list(EXAMPLES.keys())}")
    return EXAMPLES[name]()
