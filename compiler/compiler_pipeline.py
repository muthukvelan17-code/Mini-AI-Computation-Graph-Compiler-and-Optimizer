"""Compiler Pipeline coordinating optimization passes and graph transformations."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from compiler.graph import ComputationGraph
from compiler.shape_inference import infer_shapes
from optimizations.base import OptimizationPass, OptimizationResult
from optimizations.constant_folding import ConstantFoldingPass
from optimizations.dead_code_elimination import DeadCodeEliminationPass
from optimizations.common_subexpression import CommonSubexpressionEliminationPass
from optimizations.algebraic_simplification import AlgebraicSimplificationPass
from optimizations.operator_fusion import OperatorFusionPass


@dataclass
class PipelineResult:
    """Detailed summary of the compilation pipeline execution."""
    original_graph: ComputationGraph
    optimized_graph: ComputationGraph
    stage_graphs: List[Tuple[str, ComputationGraph]] = field(default_factory=list)
    stage_results: List[OptimizationResult] = field(default_factory=list)
    full_log: List[str] = field(default_factory=list)
    compilation_time_ms: float = 0.0

    @property
    def total_nodes_removed(self) -> int:
        return sum(len(res.nodes_removed) for res in self.stage_results)

    @property
    def total_nodes_added(self) -> int:
        return sum(len(res.nodes_added) for res in self.stage_results)

    @property
    def total_nodes_fused(self) -> int:
        return sum(len(res.nodes_fused) for res in self.stage_results)


class CompilerPipeline:
    """Manages ordered compiler optimization passes and validation."""

    def __init__(self, passes: Optional[List[OptimizationPass]] = None):
        if passes is not None:
            self.passes = list(passes)
        else:
            self.passes = [
                CommonSubexpressionEliminationPass(),
                ConstantFoldingPass(),
                AlgebraicSimplificationPass(),
                DeadCodeEliminationPass(),
                OperatorFusionPass(),
                DeadCodeEliminationPass(),
            ]

    @classmethod
    def default_pipeline(cls) -> CompilerPipeline:
        return cls()

    @classmethod
    def from_pass_names(cls, pass_names: List[str]) -> CompilerPipeline:
        """Create a pipeline from human-readable pass names."""
        mapping = {
            "constant_folding": ConstantFoldingPass,
            "dead_code_elimination": DeadCodeEliminationPass,
            "common_subexpression": CommonSubexpressionEliminationPass,
            "algebraic_simplification": AlgebraicSimplificationPass,
            "operator_fusion": OperatorFusionPass,
        }
        selected_passes: List[OptimizationPass] = []
        for name in pass_names:
            key = name.lower().replace(" ", "_")
            if key in mapping:
                selected_passes.append(mapping[key]())
        return cls(selected_passes)

    def compile(self, graph: ComputationGraph, infer_shapes_first: bool = True) -> PipelineResult:
        """Run the compiler pipeline on the input graph.

        The original graph is never mutated.
        """
        start_time = time.perf_counter()
        current_graph = graph.clone()

        if infer_shapes_first:
            try:
                infer_shapes(current_graph)
            except Exception:
                pass

        stage_graphs: List[Tuple[str, ComputationGraph]] = [("Original", graph.clone())]
        stage_results: List[OptimizationResult] = []
        full_log: List[str] = []

        for p in self.passes:
            res = p.run(current_graph)
            stage_results.append(res)
            current_graph = res.optimized_graph
            
            # Re-infer shapes after graph structure transformation
            try:
                infer_shapes(current_graph)
            except Exception:
                pass

            stage_graphs.append((p.name, current_graph.clone()))

            if res.log:
                for line in res.log:
                    full_log.append(f"[{p.name}] {line}")
            else:
                full_log.append(f"[{p.name}] No changes made.")

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineResult(
            original_graph=graph.clone(),
            optimized_graph=current_graph,
            stage_graphs=stage_graphs,
            stage_results=stage_results,
            full_log=full_log,
            compilation_time_ms=total_time_ms,
        )
