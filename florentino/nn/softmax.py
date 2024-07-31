import numpy as np

from .layer import Layer


class Softmax(Layer):

    def __init__(self, nodes: int, pre_node: int, dtype=np.float64):
        super().__init__(nodes, pre_node, dtype)

    @staticmethod
    def _create_a_dot_aT(a: np.ndarray) -> np.ndarray:
        # a.shape =  (n, 60000)
        # (60000, 1, n) mul (60000, n, 1) -> (60000, n, n)
        a_T = a.T
        tmp = np.matmul(a_T[:, :, np.newaxis], a_T[:, np.newaxis, :])
        return tmp  # (m, n, n)

    @staticmethod
    def _create_dia(a: np.ndarray) -> np.ndarray:
        a_T = a.T  # (60000, 10)
        eye = np.eye(a_T.shape[1])
        return a_T[:, :, np.newaxis] * eye

    @staticmethod
    def _f(z) -> np.ndarray:
        tmp = np.exp(z - np.max(z, axis=0))
        return tmp / np.sum(tmp, axis=0, keepdims=True)  # shape (nodes, number_of_image)

    @staticmethod
    def _df(a) -> np.ndarray:  # dz / da
        return Softmax._create_dia(a) - Softmax._create_a_dot_aT(a)  #

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # shape (pre_node, number_of_image)
        self.z = self.w @ self.x + self.b  # shape (nodes, number_of_image)
        self.a = self._f(self.z)  # shape (nodes, number_of_image)
        return self.a

    def backward(self, delta: np.ndarray, w_after: np.ndarray, alpha: float) -> (np.ndarray, np.ndarray):
        n, m = delta.shape
        scaler = alpha / m

        if w_after is None:
            # delta         = (n, m)
            # self._df()    = (m, n, n)
            # _tmp          = (n, m)
            delta = np.expand_dims(delta.T, axis=2)
            _tmp = np.matmul(self._df(self.a), delta).squeeze().T
        else:
            delta = delta.T.reshape(1, m, n, 1)
            _tmp = np.matmul(w_after.T @ delta, self._df(self.a)).squeeze().T

        self.b -= scaler * np.sum(_tmp, axis=1, keepdims=True)
        self.w -= scaler * (_tmp @ self.x.T)
        return _tmp, self.w

    def validate(self, x: np.ndarray) -> np.ndarray:
        return self._f(self.w @ x + self.b)
