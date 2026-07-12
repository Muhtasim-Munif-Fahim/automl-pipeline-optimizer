"""Tests for hyperparameter optimization."""

from __future__ import annotations

from automl_pipeline.optimization import get_param_grid, PARAM_GRIDS


class TestParamGrid:
    def test_random_forest_params(self):
        grid = get_param_grid("random_forest")
        assert grid is not None
        assert "n_estimators" in grid
        assert "max_depth" in grid

    def test_unknown_model(self):
        grid = get_param_grid("unknown_model")
        assert grid is None

    def test_all_models_have_grids(self):
        for model_name in [
            "logistic_regression", "random_forest", "gradient_boosting",
            "ridge", "lasso", "svm",
        ]:
            assert model_name in PARAM_GRIDS, f"{model_name} missing param grid"
