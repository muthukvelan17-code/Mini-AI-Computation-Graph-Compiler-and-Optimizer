"""Compiler optimization passes package."""

from optimizations.base import OptimizationPass, OptimizationResult
from optimizations.constant_folding import ConstantFoldingPass
from optimizations.dead_code_elimination import DeadCodeEliminationPass
from optimizations.common_subexpression import CommonSubexpressionEliminationPass
from optimizations.algebraic_simplification import AlgebraicSimplificationPass
from optimizations.operator_fusion import OperatorFusionPass

__all__ = [
    "OptimizationPass",
    "OptimizationResult",
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "CommonSubexpressionEliminationPass",
    "AlgebraicSimplificationPass",
    "OperatorFusionPass",
]
