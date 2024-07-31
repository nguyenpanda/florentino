from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Union, Optional


class Layer(ABC):

    def __init__(self, nodes: int, pre_nodes: int, dtype=np.float64):
        # @formatter:off
        self.nodes: int                     = nodes
        self.pre_nodes: int                 = pre_nodes
        self.dtype: Union[np.floating, str] = dtype
        self.x: Optional[np.ndarray]        = None
        self.z: Optional[np.ndarray]        = None
        self.a: Optional[np.ndarray]        = None
        self.b: np.ndarray                  = (np.random.rand(nodes, 1) - 0.5).astype(dtype)
        self.w: np.ndarray                  = (np.random.rand(nodes, pre_nodes) - 0.5).astype(dtype)
        # @formatter:off

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, delta: np.ndarray, w_after: np.ndarray, alpha: float) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def validate(self, x: np.ndarray) -> np.ndarray:
        pass

    def size(self) -> int:
        return self.w.size + self.b.size

    def __str__(self):
        return f"{self.__class__.__name__}(nodes={self.nodes})"

    def __len__(self):
        return self.size()
