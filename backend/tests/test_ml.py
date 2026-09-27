import numpy as np
import pytest

from app.ml import tools_ml
from app.ml.from_scratch import LinearRegressionGD


def test_describe_dataset_reports_missing(housing):
    info = tools_ml.describe_dataset(housing)
    assert info["rows"] == 800
    assert info["missing"] == {"antiguedad": 20}


def test_correlations_rank_area_high(housing):
    corr = tools_ml.correlations(housing, "precio")["correlations"]
    assert next(iter(corr)) == "area_m2"


@pytest.mark.parametrize("model_type", tools_ml.MODEL_TYPES)
def test_train_regression(housing, model_type):
    res = tools_ml.train_model(housing, "precio", model_type)
    assert res["task"] == "regression"
    assert res["metrics"]["r2_test"] > 0.6


def test_train_classification_detected(churn):
    res = tools_ml.train_model(churn, "abandona", "tree")
    assert res["task"] == "classification"
    assert 0.5 < res["metrics"]["accuracy_test"] <= 1


def test_cross_validate(churn):
    res = tools_ml.cross_validate(churn, "abandona", "linear", folds=3)
    assert len(res["scores"]) == 3 and res["scoring"] == "accuracy"


def test_unknown_target_raises(housing):
    with pytest.raises(ValueError):
        tools_ml.train_model(housing, "no_existe")


def test_gradient_descent_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0 + rng.normal(0, 0.1, 200)
    model = LinearRegressionGD(learning_rate=0.1, epochs=1000).fit(X, y)
    np.testing.assert_allclose(model.w, [2.0, -1.0, 0.5], atol=0.05)
    assert abs(model.b - 3.0) < 0.05
    assert model.loss_history[-1] < model.loss_history[0]  # la pérdida baja
