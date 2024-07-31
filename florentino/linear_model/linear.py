import numpy as np


class LinearRegression:
    """
    A simple linear regression model.

    Attributes:
        x (np.ndarray): The feature matrix with an added column of ones.
        y (np.ndarray): The target vector.
        beta (np.ndarray): The coefficients of the linear regression model.
        num_data (int): The number of data points.
        num_feature (int): The number of features.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Initializes the LinearRegression model with training data.

        Parameters:
            x_train (np.ndarray): The training feature matrix.
            y_train (np.ndarray): The training target vector.

        Raises:
            ValueError: If the input dimensions are incorrect.
        """
        _is_valid = self._check(x_train, y_train)
        if _is_valid:
            raise _is_valid
        _is_valid = self._check(x_train, y_train)
        self._fit(x_train, y_train)
        self.num_data, self.num_feature = self.x.shape
        self._train()

    @staticmethod
    def _check(x: np.ndarray, y: np.ndarray):
        if x.ndim != 2:
            return ValueError("x must be a 2-dimensional array")
        if y.ndim != 2:
            return ValueError("y must be a 2-dimensional array")
        if y.shape[1] != 1:
            return ValueError("y must have exactly one column")
        if x.shape[0] != y.shape[0]:
            return ValueError("Number of rows in x must match number of rows in y")
        return None

    def _fit(self, x: np.ndarray, y: np.ndarray):
        """
        Prepares the feature matrix and target vector for training.

        Parameters:
            x (np.ndarray): The feature matrix.
            y (np.ndarray): The target vector.
        """
        _one_column = np.ones((x.shape[0], 1), dtype=x.dtype)
        self.x = np.hstack((_one_column, x))
        self.y = y.reshape(-1, 1)

    def _train(self):
        _x = self.x
        self.beta = (np.linalg.inv(_x.T @ _x)).T @ _x.T @ self.y

    def predict(self, x: np.ndarray):
        """
        Predicts the target values for a given feature matrix.

        Parameters:
            x (np.ndarray): The feature matrix for which to make predictions.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If the input dimensions are incorrect.
        """
        if x.ndim != 2:
            raise ValueError("x must be a 2-dimensional array")
        if x.shape[1] != self.num_feature - 1:
            raise ValueError(f"x must have {self.num_feature - 1} features")

        _one_column = np.ones((x.shape[0], 1), dtype=x.dtype)
        x = np.hstack((_one_column, x))
        return x @ self.beta

    def para(self) -> np.ndarray:
        return self.beta

    def loss(self):
        return np.sum((self.y - self.x @ self.beta) ** 2)
