import pandas as pd
from sklearn.linear_model import LinearRegression as sk_LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore

from lreg import LinearRegression


def main():
    data = pd.read_csv("train.csv")

    model = LinearRegression()
    model.fit(data["x"], data["y"])

    model_slope = round(model.slope_(), 4)
    model_intercept = round(model.intercept_(), 4)
    print(f"{model_slope=}, {model_intercept=}")
    # model_slope=1.0007, model_intercept=-0.1073

    print(f"{model.predict(6)=}")
    # model.predict(6)=5.896672826836847

    sk_model = sk_LinearRegression()
    sk_model.fit(data.drop(columns=["y"]), data["y"])
    sk_model_slope = round(sk_model.coef_[0], 4)
    sk_model_intercept = round(sk_model.intercept_, 4)

    print(f"{sk_model_slope=}, {sk_model_intercept=}")
    # sk_model_slope=1.0007, sk_model_intercept=-0.1073

    test_data = pd.read_csv("test.csv")
    y_pred = model.predict(test_data["x"])
    y_sk_pred = sk_model.predict(test_data.drop(columns=["y"]))

    mae = mean_absolute_error(test_data["y"], y_pred)
    sk_mae = mean_absolute_error(test_data["y"], y_sk_pred)
    print(f"{mae=}, {sk_mae=}")
    # mae=2.4157718500412573, sk_mae=2.4157718500412595


if __name__ == "__main__":
    main()
