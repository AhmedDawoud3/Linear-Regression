import os

import numpy as np
import pytest

from lreg import LinearRegression

# Constants for file paths used in tests
MODEL_SAVE_PATH = "test_model.pkl"


# Happy path tests with various realistic test values
@pytest.mark.parametrize(
    "x, y, expected_m, expected_b",
    [
        # ID: Test-Simple-Linear-1
        (np.array([1, 2, 3]), np.array([2, 3, 4]), 1, 1),
        # ID: Test-Simple-Linear-2
        (np.array([0, 1, 2]), np.array([1, 3, 5]), 2, 1),
        # ID: Test-Simple-Linear-3
        (np.array([-1, 0, 1]), np.array([-1, 1, 3]), 2, 1),
    ],
)
def test_fit_happy_path(x, y, expected_m, expected_b):
    # Arrange
    model = LinearRegression()

    # Act
    model.fit(x, y)

    # Assert
    assert model.is_fitted
    assert np.isclose(model.slope_(), expected_m)
    assert np.isclose(model.intercept_(), expected_b)


# Edge cases
@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([1]), np.array([1])),  # ID: Test-Edge-One-Point
        # ID: Test-Edge-All-Same-Points
        (np.array([1, 1, 1]), np.array([2, 2, 2])),
    ],
)
def test_fit_edge_cases(x, y):
    # Arrange
    model = LinearRegression()

    # Act & Assert
    with pytest.raises(AssertionError):
        model.fit(x, y)


# Error cases
@pytest.mark.parametrize(
    "x, y, exception, message",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([1, 2]),
            AssertionError,
            "X must be a single column vector",
        ),  # ID: Test-Error-MultiColumn-X
        (
            np.array([1, 2]),
            np.array([[1, 2], [3, 4]]),
            AssertionError,
            "y must be a single column vector",
        ),  # ID: Test-Error-MultiColumn-y
        (
            np.array([1, 2]),
            "not an array",
            AssertionError,
            "X and y must be the same length",
        ),  # ID: Test-Error-NonArray-y
        (
            np.array([1, 2, 3]),
            np.array([1, 2]),
            AssertionError,
            "X and y must be the same length",
        ),  # ID: Test-Error-Different-Lengths
    ],
)
def test_fit_error_cases(x, y, exception, message):
    # Arrange
    model = LinearRegression()

    # Act & Assert
    with pytest.raises(exception) as exc_info:
        model.fit(x, y)
    assert message in str(exc_info.value)


def test_linear_regression_initialization():
    model = LinearRegression()
    assert not model.is_fitted


def test_linear_regression_fit():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    assert model.is_fitted


def test_linear_regression_predict():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    x_test = np.array([[6], [7], [8]])
    y_pred = model.predict(x_test)
    assert np.array_equal(y_pred, np.array([[12], [14], [16]]))


def test_linear_regression_intercept():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    assert model.intercept_() == 0.0


def test_linear_regression_slope():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    assert model.slope_() == 2.0


def test_linear_regression_invalid_input():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    x_invalid = np.array([[1, 2], [3, 4]])
    y_invalid = np.array([[1, 2], [3, 4]])
    x_invalid_pred = np.array([[1, 2], [3, 4]])

    with pytest.raises(AssertionError):
        model.fit(x_invalid, y)

    with pytest.raises(AssertionError):
        model.fit(x, y_invalid)

    with pytest.raises(AssertionError):
        model.predict(x_invalid_pred)


def test_linear_regression_call_before_fit():
    model = LinearRegression()

    with pytest.raises(AssertionError):
        model.intercept_()

    with pytest.raises(AssertionError):
        model.slope_()


def test_saving():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    model.save(MODEL_SAVE_PATH)
    assert os.path.isfile(MODEL_SAVE_PATH)
    assert os.path.getsize(MODEL_SAVE_PATH) > 0
    os.remove(MODEL_SAVE_PATH)


def test_loading():
    model = LinearRegression()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])
    model.fit(x, y)
    model.save(MODEL_SAVE_PATH)
    model_loaded = LinearRegression.load(MODEL_SAVE_PATH)
    assert model_loaded.is_fitted
    assert model_loaded.slope_() == model.slope_()
    assert model_loaded.intercept_() == model.intercept_()
    y_hat_loaded = model_loaded.predict(x)
    y_hat = model.predict(x)
    assert np.array_equal(y_hat, y_hat_loaded)
    os.remove(MODEL_SAVE_PATH)
