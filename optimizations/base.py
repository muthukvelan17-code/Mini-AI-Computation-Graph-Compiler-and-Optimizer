"""Base class and result data structure for compiler optimization passes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
from compiler.graph import ComputationGraph


@dataclass
class OptimizationResult:
    """Stores the resulting graph and transformation ledger of an optimization pass."""
    optimized_graph: ComputationGraph
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_modified: List[str] = field(default_factory=list)
    nodes_fused: List[Tuple[str, List[str]]] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    pass_name: str = "OptimizationPass"

    @property
    def has_changed(self) -> bool:
        """Returns True if any graph mutations were performed by this pass."""
        return (
            len(self.nodes_added) > 0
            or len(self.nodes_removed) > 0
            or len(self.nodes_modified) > 0
            or len(self.nodes_fused) > 0
        )


class OptimizationPass(ABC):
    """Abstract base class for all compiler graph transformation passes."""

    name: str = "BaseOptimizationPass"
    description: str = "Abstract optimization pass"

    @abstractmethod
    def run(self, graph: ComputationGraph) -> OptimizationResult:
        """Execute the optimization pass on a copy of the graph.

        Args:
            graph: The input ComputationGraph (must NOT be mutated directly).

        Returns:
            OptimizationResult containing the optimized graph and change ledger.
        """
        pass
