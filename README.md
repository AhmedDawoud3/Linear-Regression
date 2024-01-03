# Linear Regression Model (lreg.py)

This repository contains a simple implementation of a linear regression model in Python. The script `lreg.py` can be used to fit a linear regression model to a dataset and make predictions. The repository provides a basic example of how to use the model to make predictions.

# Getting Started

## Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.10+
- Jupyter Notebook (optional, for running the provided example)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/AhmedDawoud3/Linear-Regression
```

Change into the project directory:

```bash
cd linear-regression-model
```

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This will install all the Python packages needed to run lreg.py.

## Usage

### Example Usage

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("train.csv")

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(data[["x"]], data["y"])

# Get the slope and intercept of the model
model_slope = round(model.coef_[0], 4)
model_intercept = round(model.intercept_, 4)
print(f"{model_slope=}, {model_intercept=}")
# Output: model_slope=1.0007, model_intercept=-0.1073

# Make a prediction for a new input (e.g., x=6)
prediction = model.predict([[6]])
print(f"{prediction=}")
# Output: prediction=array([5.89667283])
```

This example demonstrates loading a dataset from a CSV file, training a linear regression model, and making a prediction for a new input (in this case, x=6).

Feel free to modify the code to suit your specific use case or integrate it into your own projects.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python.
