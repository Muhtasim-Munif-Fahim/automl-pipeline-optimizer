"""Tests for evaluation module."""

from __future__ import annotations

import numpy as np

from automl_pipeline.evaluation import calibration_curve_data, confusion_matrix_data


class TestConfusionMatrix:
    def test_binary(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        result = confusion_matrix_data(y_true, y_pred)
        assert "matrix" in result
        assert "accuracy" in result


class TestCalibrationCurve:
    def test_output_shape(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        result = calibration_curve_data(y_true, y_prob, n_bins=3)
        assert "prob_true" in result
        assert "prob_pred" in result
