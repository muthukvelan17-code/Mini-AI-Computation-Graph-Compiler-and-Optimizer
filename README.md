# Mini AI Computation Graph Compiler and Optimizer

An educational, functional mini compiler and graph optimizer that represents mathematical and AI tensor operations as a Directed Acyclic Graph (DAG), applies compiler optimization passes, and executes the optimized graph using a vectorized NumPy interpreter.

---

## 🚀 Key Features

- **Computation Graph Engine (DAG):**
  - Directed Acyclic Graph representation of tensor computations with cycle detection (Kahn's algorithm).
  - Node types: `Input`, `Constant`, `Add`, `Sub`, `Mul`, `MatMul`, `ReLU`, `Sigmoid`, `ScalarMul`, and `Fused` operations.
  - Safe, non-mutating transformations with deep graph cloning.
  - Safe, pickle-free JSON graph export and import.

- **Intermediate Representation (IR) & Static Shape Inference:**
  - LLVM-style human-readable textual IR and structured instruction table.
  - Compile-time static shape inference adhering to NumPy multidimensional broadcasting and matrix multiplication dimensionality rules.
  - Clear, informative diagnostics for dimension mismatches.

- **5 Compiler Optimization Passes:**
  1. **Constant Folding:** Pre-computes operations with compile-time known constant operands.
  2. **Dead Code Elimination (DCE):** Traverses backward from graph terminal outputs to prune unreferenced and unreachable computation paths.
  3. **Common Subexpression Elimination (CSE):** Detects identical subexpressions (including commutative awareness for `add` and `mul`) and reuses evaluated tensors.
  4. **Algebraic Simplification:** Applies algebraic identities ($x + 0 \to x$, $x \times 1 \to x$, $x - 0 \to x$, $x \times 0 \to 0$, $x \times \text{scalar}(1) \to x$) while verifying shape preservation.
  5. **Operator Fusion:** Combines consecutive operations (`a * b + c` $\to$ `FusedMulAdd`, `(X @ W) + b` $\to$ `FusedLinear`, `ReLU((X @ W) + b)` $\to$ `FusedLinearReLU`, `ReLU(a + b)` $\to$ `FusedAddReLU`) to reduce kernel launch overhead and intermediate tensor allocations.

- **Execution Engine:**
  - Interprets graphs in topological execution order using NumPy.
  - Intermediate result caching and per-node execution telemetry.
  - Guaranteed input immutability.

- **Repeatable Benchmarking & Correctness Engine:**
  - Excludes compilation and graph construction from execution timing.
  - Reports median latencies, standard deviations, speedup/slowdown percentages, and node reduction counts.
  - Detects microsecond-scale execution noise on tiny graphs and reports results as inconclusive instead of fabricating speedups.
  - Verifies exact numerical parity ($\text{atol}=10^{-5}, \text{rtol}=10^{-5}$) against baseline execution.
  - Exports benchmark metrics as JSON or CSV.

- **Optional PyTorch Backend:**
  - If PyTorch is installed, runs equivalent computations in PyTorch to compare execution performance and numerical accuracy against the NumPy engine.
  - Graceful fallback with clear installation instructions if PyTorch is absent.

- **Interactive Streamlit Web Dashboard:**
  - 7 structured sections: Overview, Graph Builder, Optimization Passes, Execution & Parity, Benchmarking, Compiler Internals, and Tests & Documentation.
  - Interactive Plotly DAG visualization with hierarchical layout and color-coded node categories.

---

## 📁 Project Structure

```text
mini-ai-compiler/
├── app.py                         # Streamlit interactive web dashboard
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation & instructions
├── .gitignore                     # Git ignore rules
├── compiler/
│   ├── __init__.py                # Package exports
│   ├── node.py                    # Node & OpType representations
│   ├── graph.py                   # ComputationGraph DAG & cycle detection
│   ├── ir.py                      # Intermediate Representation generator
│   ├── shape_inference.py         # Static shape inference & broadcasting
│   ├── executor.py                # NumPy execution interpreter
│   ├── compiler_pipeline.py       # Multi-pass compiler pipeline orchestrator
│   ├── serialization.py           # Safe JSON graph export/import
│   └── pytorch_comparison.py      # Optional PyTorch comparison runner
├── optimizations/
│   ├── __init__.py                # Pass exports
│   ├── base.py                    # OptimizationPass abstract base class
│   ├── constant_folding.py        # Constant folding pass
│   ├── dead_code_elimination.py   # Dead code elimination pass
│   ├── common_subexpression.py    # Common subexpression elimination pass
│   ├── algebraic_simplification.py# Algebraic simplification pass
│   └── operator_fusion.py         # Operator fusion pass
├── benchmarks/
│   ├── __init__.py                # Benchmark exports
│   ├── metrics.py                 # BenchmarkMetrics dataclass & formatting
│   └── benchmark_runner.py        # Repeated timing runner & noise analysis
├── examples/
│   ├── __init__.py                # Examples exports
│   └── example_graphs.py          # 7 Built-in demonstration graphs
├── visualization/
│   ├── __init__.py                # Visualization exports
│   ├── graph_visualizer.py        # Plotly & Matplotlib DAG visualizers
│   └── charts.py                  # Benchmark comparison & latency distribution charts
└── tests/
    ├── __init__.py
    ├── test_graph.py              # DAG construction & cycle detection tests
    ├── test_shape_inference.py    # Shape inference & broadcast error tests
    ├── test_executor.py           # Execution engine & immutability tests
    ├── test_constant_folding.py   # Constant folding pass unit tests
    ├── test_dead_code_elimination.py # DCE unit tests
    ├── test_common_subexpression.py # CSE unit tests
    ├── test_algebraic_simplification.py # Algebraic simplification unit tests
    ├── test_operator_fusion.py    # Operator fusion unit tests
    └── test_correctness.py        # End-to-end parity test across all examples
```

---

## 💻 Installation & Setup (Windows PowerShell)

### Step 1: Create Virtual Environment

```powershell
python -m venv .venv
```

*(If `python` is not directly in PATH, use `py -m venv .venv`)*

### Step 2: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Troubleshooting PowerShell Script Execution Policy:**
> If you encounter an execution policy error such as `cannot be loaded because running scripts is disabled on this system`, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Alternatively, execute using the virtual environment's Python executable directly:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> ```

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

*(Optional PyTorch installation for real framework comparison)*:
```powershell
python -m pip install torch
```

---

## 🧪 Running the Automated Test Suite

Execute the full automated test suite containing 31 comprehensive unit and integration tests:

```powershell
python -m pytest -v
```

All tests validate:
- Cycle detection raising `CycleDetectedError`
- Static shape inference with invalid broadcast error checks
- Intermediate representation generation
- Constant folding, DCE, CSE, algebraic simplification, and operator fusion passes
- Exact numerical floating-point parity across all 7 built-in example graphs

---

## 🖥️ Launching the Streamlit Web Application

To launch the interactive dashboard:

```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🔍 Built-in Demonstration Examples

The compiler includes pre-configured working examples matching core compiler optimization principles:

1. **Example 1: Constant Folding**
   - Expression: `(2 + 3) * x`
   - Demonstrates: The addition of constants `2 + 3` is evaluated at compile time to `5`, transforming into `5 * x`.

2. **Example 2: Dead Code Elimination (DCE)**
   - Expression: Live path `y = a * b`, Dead path `dead = sigmoid(a + c)`
   - Demonstrates: Dead operations and unused input `c` are completely pruned from the computation graph.

3. **Example 3: Common Subexpression Elimination (CSE)**
   - Expression: `(a * b) + (a * b)`
   - Demonstrates: The second multiplication is detected as identical to the first and eliminated, with its output rewired to reuse the first computation.

4. **Example 4: Algebraic Simplification**
   - Expression: `(x * 1) + 0`
   - Demonstrates: Multiplicative and additive identities are simplified, eliminating intermediate computations and directly referencing `x`.

5. **Example 5: Operator Fusion**
   - Expression: `(a * b) + c`
   - Demonstrates: Combines element-wise multiplication and addition into a single `FusedMulAdd` execution node.

6. **Example 6: AI Neural Network Layer**
   - Expression: `ReLU((X @ W) + b)`
   - Demonstrates: Matrix multiplication, bias addition, and ReLU activation are fused into a single `FusedLinearReLU` execution kernel.

7. **Example 7: Two-Layer Multi-Layer Perceptron (Deep Graph)**
   - Expression: `Sigmoid(X @ W1 + b1)` followed by `ReLU(H1 @ W2 + b2)` with dead weight regularization branches.
   - Demonstrates: Multi-pass compilation combining DCE and multiple operator fusions.

---

## 📊 Benchmarking & Performance Methodology

- **Execution Timing:** Timed using Python's high-resolution `time.perf_counter()`.
- **Compilation vs. Execution Isolation:** Compilation time (graph analysis, transformations, shape inference) is recorded separately and excluded from execution benchmarks.
- **Representative Statistics:** Runs configurable warmup runs followed by $N$ benchmark iterations, reporting median execution times and standard deviations.
- **Microsecond Noise Handling:** For micro-graphs where execution duration is under 40 microseconds, Python interpreter function call overhead exceeds NumPy BLAS computation. In this regime, timing fluctuations produce noisy measurements; the benchmark runner explicitly identifies these cases as **Inconclusive** rather than claiming an artificial speedup.
- **Formula Used:**
  $$\text{Speedup (\%)} = \left(\frac{\text{Baseline Time} - \text{Optimized Time}}{\text{Baseline Time}}\right) \times 100$$
  Negative values are explicitly reported as **Slowdowns**.

---

## ⚠️ Limitations & Future Work

1. **Host-Language Interpreter Overhead:**
   - This project is an educational compiler written in pure Python using NumPy for execution. Industrial compilers (e.g. PyTorch Inductor, XLA, TVM, Triton) compile down to low-level C++, LLVM IR, or CUDA/Triton kernels to eliminate host interpreter overhead.
2. **Dynamic Tensor Rank:**
   - Static shape inference currently requires tensor ranks to be specified at compile time. Dynamic rank inference is left for future work.
3. **C Integration / Lowering:**
   - Future extensions can add a C code generator backend (e.g. emitting C99 code with OpenMP or BLAS) and compiling shared libraries (`.so`/`.dll`) via `ctypes`.

---

## 🌐 Publishing to GitHub and Deploying Online

### Step 1: Initialize Git and Push to GitHub

```powershell
git init
git add .
git commit -m "Initial commit: Mini AI Computation Graph Compiler and Optimizer"
git branch -M main
git remote add origin https://github.com/<your-username>/mini-ai-compiler.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with GitHub.
3. Click **New app**.
4. Select your repository: `<your-username>/mini-ai-compiler`.
5. Set Main file path to: `app.py`.
6. Click **Deploy!**
