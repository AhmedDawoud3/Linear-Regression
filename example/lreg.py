import pickle
from typing import Optional

import numpy as np


class LinearRegression:
    _m: float
    _b: float
    is_fitted: bool = False
    """
    A simple linear regression model.

    Attributes:
        is_fitted (bool): Whether the model has been fitted.
    """

    def __init__(self):
        """
        Initialize a new instance of LinearRegression.
        """
        self._m = np.random.rand()
        self._b = np.random.rand()

    def _validate(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Validate the input arrays.

        Args:
            x (np.ndarray): The input features.
            y (Optional[np.ndarray]): The target values. If None, only x is validated.

        Raises:
            AssertionError: If the input arrays do not meet the requirements.
        """
        assert x.shape[0] > 1, "X must have more than one row"
        assert len(x.shape) == 1 or x.shape[1] == 1, "X must be a single column vector"
        assert len(x) > 1, "X must have more than one row"
        if hasattr(x, "nunique"):
            assert x.nunique() > 1, "X must have more than one unique value"
        else:
            assert len(set(x)) > 1, "X must have more than one unique value"

        if y is None:
            return

        assert len(x) == len(y), "X and y must be the same length"
        assert len(y.shape) == 1 or y.shape[1] == 1, "y must be a single column vector"

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data.

        Args:
            x (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Raises:
            AssertionError: If the input arrays do not meet the requirements.
        """
        self._validate(x, y)

        n = len(x)

        sumX = x.sum()
        sumY = y.sum()
        sumXY = (x * y).sum()
        sumX2 = (x * x).sum()

        self._m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX**2)
        self._b = (sumY - self._m * sumX) / n

        self.is_fitted = True

    def predict(self, x: np.ndarray | float | int | list):
        """
        Predict the target values for the given input features.

        Args:
            x (np.ndarray | float | int | list): The input features.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            AssertionError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise AssertionError("Model must be fitted before calling this method")

        if isinstance(x, (float, int)):
            return self._m * x + self._b

        return np.array([self._m * xi + self._b for xi in x])

    def intercept_(self) -> float:
        """
        Get the y-intercept of the regression line.

        Returns:
            float: The y-intercept.

        Raises:
            AssertionError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise AssertionError("Model must be fitted before calling this method")
        return self._b

    def slope_(self) -> float:
        """
        Get the slope of the regression line.

        Returns:
            float: The slope.

        Raises:
            AssertionError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise AssertionError("Model must be fitted before calling this method")
        return self._m

    def __repr__(self):
        if self.is_fitted:
            return f"LinearRegression(m={self._m}, b={self._b})"
        return "LinearRegression not fitted"

    def __str__(self):
        return self.__repr__()

    def as_dict(self):
        return {"m": self._m, "b": self._b, "is_fitted": self.is_fitted}

    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """
        Load a model from a file.

        Args:
            path (str): The path to load the model from.

        Returns:
            LinearRegression: The loaded model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
