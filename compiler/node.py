"""Node representation for the Computation Graph."""

from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import numpy as np


class OpType(str, Enum):
    """Supported operation types in the computation graph."""
    INPUT = "input"
    CONSTANT = "constant"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MATMUL = "matmul"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SCALAR_MUL = "scalar_mul"
    # Fused operation types produced by optimization passes
    FUSED_MUL_ADD = "fused_mul_add"              # a * b + c
    FUSED_MATMUL_ADD = "fused_matmul_add"        # (X @ W) + b
    FUSED_MATMUL_ADD_RELU = "fused_matmul_add_relu"  # relu((X @ W) + b)
    FUSED_ADD_RELU = "fused_add_relu"            # relu(a + b)

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return any(value == item.value for item in cls)

    @classmethod
    def is_fused(cls, op: Union[OpType, str]) -> bool:
        op_str = op.value if isinstance(op, OpType) else str(op)
        return op_str.startswith("fused_")


class Node:
    """Represents a single computational unit (node) in the DAG.

    Attributes:
        id: Unique identifier for the node (e.g., "node_0", "input_X").
        op_type: Operation type (OpType or string).
        inputs: List of parent node IDs that feed into this node.
        data: Constant tensor data or default input values (NumPy array).
        shape: Inferred or declared output shape as a tuple of ints.
        dtype: Data type string (e.g. 'float32', 'int32').
        metadata: Extra attributes (e.g. scalar factor, fused subgraph info, doc).
        name: Human-friendly display label.
        result: Cached execution result array.
    """

    def __init__(
        self,
        node_id: str,
        op_type: Union[OpType, str],
        inputs: Optional[List[str]] = None,
        data: Optional[Union[np.ndarray, List, float, int]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        self.id = str(node_id)
        if isinstance(op_type, OpType):
            self.op_type = op_type
        else:
            try:
                self.op_type = OpType(op_type)
            except ValueError:
                self.op_type = op_type  # Allow custom/extension op types

        self.inputs: List[str] = list(inputs) if inputs is not None else []
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self.name: str = name if name is not None else self.id
        self.dtype: str = str(dtype)
        self.result: Optional[np.ndarray] = None

        # Process tensor data
        if data is not None:
            if not isinstance(data, np.ndarray):
                self.data = np.array(data, dtype=np.dtype(dtype))
            else:
                self.data = data.astype(np.dtype(dtype)) if data.dtype != np.dtype(dtype) else data
            self.shape = self.data.shape
        else:
            self.data = None
            self.shape = tuple(shape) if shape is not None else None

    @property
    def is_constant(self) -> bool:
        return self.op_type == OpType.CONSTANT or (self.op_type == "constant")

    @property
    def is_input(self) -> bool:
        return self.op_type == OpType.INPUT or (self.op_type == "input")

    @property
    def is_fused(self) -> bool:
        return OpType.is_fused(self.op_type)

    def clone(self) -> Node:
        """Create a deep copy of this node."""
        cloned_data = self.data.copy() if self.data is not None else None
        cloned_node = Node(
            node_id=self.id,
            op_type=self.op_type,
            inputs=list(self.inputs),
            data=cloned_data,
            shape=self.shape,
            dtype=self.dtype,
            metadata={k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.metadata.items()},
            name=self.name,
        )
        if self.result is not None:
            cloned_node.result = self.result.copy()
        return cloned_node

    def to_dict(self) -> dict[str, Any]:
        """Convert node to a dictionary representation suitable for serialization."""
        d: dict[str, Any] = {
            "id": self.id,
            "op_type": self.op_type.value if isinstance(self.op_type, OpType) else str(self.op_type),
            "inputs": self.inputs,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "metadata": self.metadata,
            "name": self.name,
        }
        if self.data is not None:
            d["data"] = self.data.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        """Construct a Node from a dictionary representation."""
        data = np.array(d["data"], dtype=d.get("dtype", "float32")) if "data" in d and d["data"] is not None else None
        shape = tuple(d["shape"]) if d.get("shape") is not None else None
        return cls(
            node_id=d["id"],
            op_type=d["op_type"],
            inputs=d.get("inputs", []),
            data=data,
            shape=shape,
            dtype=d.get("dtype", "float32"),
            metadata=d.get("metadata", {}),
            name=d.get("name", d["id"]),
        )

    def __repr__(self) -> str:
        shape_str = f", shape={self.shape}" if self.shape is not None else ""
        inputs_str = f", inputs={self.inputs}" if self.inputs else ""
        return f"Node({self.id!r}, {self.op_type.value if isinstance(self.op_type, OpType) else self.op_type}{inputs_str}{shape_str})"
