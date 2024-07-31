import abc

import numpy as np


class Loss(abc.ABC):

    def __init__(self, f, df):
        self.f = f
        self.df = df

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.f(y_pred, y_true)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.df(y_pred, y_true)

    def __str__(self):
        return f'{self.__class__.__name__}'


class MeanSquareError(Loss):

    def __init__(self):
        def f(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
            return np.mean((y_pred - y_true) ** 2)

        def df(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            return y_pred - y_true

        super().__init__(f, df)


class CrossEntropy(Loss):

    def __init__(self, epsilon: float = 1e-15):
        def f(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return - np.sum(y_true * np.log(y_pred))

        def df(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            y_pred = np.clip(y_pred, epsilon, 1)
            return - y_true / y_pred

        super().__init__(f, df)


class BinaryCrossEntropy(Loss):

    def __init__(self, epsilon: float = 1e-15):
        def f(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        def df(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return (1 - y_true) / (1 - y_pred) - y_true / y_pred

        super().__init__(f, df)
