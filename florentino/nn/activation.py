import numpy as np
from typing_extensions import Callable, Union


class Activation:

    @staticmethod
    def none(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def dnone(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=x.dtype)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def drelu(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def map_name(func_name: Union[str, Callable]) -> str:
        if isinstance(func_name, str):
            return func_name
        elif callable(func_name):
            func_map = {
                Activation.none: 'none',
                Activation.relu: 'relu',
                Activation.sigmoid: 'sigmoid'
            }
            if func_name in func_map:
                return func_map[func_name]
            else:
                raise ValueError(f'Unknown activation function: {func_name}')
        else:
            raise ValueError('func_name must be a str or Callable')

    @staticmethod
    def map_func(func_name: Union[str, Callable]) -> Callable:
        if callable(func_name):
            return func_name

        func_map = {
            'relu': Activation.relu,
            'sigmoid': Activation.sigmoid,
            'none': Activation.none
        }

        if isinstance(func_name, str):
            func_name = func_name.lower()
            if func_name in func_map:
                return func_map[func_name]
            else:
                raise ValueError(f'Unknown activation function name: {func_name}')
        else:
            raise ValueError('func_name must be a str or Callable')

    @staticmethod
    def map_dfunc(func_name: Union[str, Callable]) -> Callable:
        if callable(func_name):
            func_name = func_name.__name__

        dfunc_map = {
            'relu': Activation.drelu,
            'sigmoid': Activation.dsigmoid,
            'none': Activation.dnone
        }

        if isinstance(func_name, str):
            func_name = func_name.lower()
            if func_name in dfunc_map:
                return dfunc_map[func_name]
            else:
                raise ValueError(f'Unknown derivative function name: {func_name}')
        else:
            raise ValueError('dfunc_name must be a str or Callable')
