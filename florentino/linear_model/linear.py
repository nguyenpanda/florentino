import numpy as np


class LinearRegression:
    """
    A simple linear regression model, extended for polynomial regression.

    Attributes:
        x (np.ndarray): The feature matrix with an added column of ones (and polynomial features).
        y (np.ndarray): The target vector.
        beta (np.ndarray): The coefficients of the regression model.
        num_data (int): The number of data points.
        num_feature (int): The number of features.
        degree (int): The degree of the polynomial regression.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, degree: int = 1):
        """
        Initializes the LinearRegression model with training data and degree of polynomial.

        Parameters:
            x_train (np.ndarray): The training feature matrix.
            y_train (np.ndarray): The training target vector.
            degree (int): The degree of the polynomial features (default is 1 for linear regression).

        Raises:
            ValueError: If the input dimensions are incorrect.
        """
        self.degree = degree
        _is_valid = self._check(x_train, y_train)
        if _is_valid:
            raise _is_valid
        x_train_poly = self._polynomial_features(x_train, self.degree)
        self._fit(x_train_poly, y_train)
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
            x (np.ndarray): The feature matrix (polynomial features included).
            y (np.ndarray): The target vector.
        """
        _one_column = np.ones((x.shape[0], 1), dtype=x.dtype)
        self.x = np.hstack((_one_column, x))
        self.y = y.reshape(-1, 1)

    def _train(self):
        """
        Trains the model by calculating the coefficients (beta) using the normal equation.
        """
        _x = self.x
        self.beta = np.linalg.inv(_x.T @ _x) @ _x.T @ self.y

    def predict(self, x: np.ndarray) -> np.ndarray:
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
        x_poly = self._polynomial_features(x, self.degree)
        if x_poly.shape[1] != self.num_feature - 1:
            raise ValueError(f"x must have {self.num_feature - 1} features (including polynomial terms)")

        _one_column = np.ones((x.shape[0], 1), dtype=x.dtype)
        x = np.hstack((_one_column, x_poly))
        return x @ self.beta

    def para(self) -> np.ndarray:
        """
        Returns the learned coefficients (beta).

        Returns:
            np.ndarray: The coefficients of the regression model.
        """
        return self.beta

    def loss(self):
        """
        Calculates the loss function (sum of squared errors).

        Returns:
            float: The sum of squared residuals.
        """
        return np.sum((self.y - self.x @ self.beta) ** 2)

    @staticmethod
    def _polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:  # TODO: Fix this in the future
        """
        Transforms the input matrix into a polynomial feature space of a given degree.

        Parameters:
            x (np.ndarray): The original feature matrix.
            degree (int): The degree of the polynomial transformation.

        Returns:
            np.ndarray: The feature matrix transformed into polynomial feature space.
        """
        # Start with the original features
        poly_features = x.copy()
        for deg in range(2, degree + 1):
            poly_features = np.hstack((poly_features, x ** deg))
        return poly_features
