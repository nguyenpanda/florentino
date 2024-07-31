import numpy as np
from typing_extensions import *

from .activation import Activation
from .layer import Layer


class Dense(Layer):

    def __init__(self, nodes: int, pre_node: int, activation: str | Callable, dtype=np.float64):
        self.f = Activation.map_func(activation)
        self.df = Activation.map_dfunc(activation)
        super().__init__(nodes, pre_node, dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = self.w @ self.x + self.b
        self.a = self.f(self.z)
        return self.a

    def backward(self, delta: np.ndarray, w_after: np.ndarray, alpha: float) -> (np.ndarray, np.ndarray):
        scaler = alpha / delta.shape[1]

        if w_after is None:
            _tmp = delta * self.df(self.z)
        else:
            _tmp = (w_after.T @ delta) * self.df(self.z)

        self.b -= scaler * np.sum(_tmp, axis=1, keepdims=True)
        self.w -= scaler * (_tmp @ self.x.T)
        return _tmp, self.w

    def validate(self, x: np.ndarray) -> np.ndarray:
        return self.f(self.w @ x + self.b)

    def __str__(self):
        return f"{self.__class__.__name__}(nodes={self.nodes}, activate={self.f.__name__})"
