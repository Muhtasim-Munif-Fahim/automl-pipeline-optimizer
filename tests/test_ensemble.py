"""Tests for ensemble module."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from automl_pipeline.ensemble import build_voting_ensemble, build_stacking_ensemble


class TestEnsemble:
    def test_voting_classifier(self):
        models = [("lr", LogisticRegression(max_iter=1000)), ("svm", SVC(probability=True))]
        ensemble = build_voting_ensemble(models, problem_type="classification")
        assert ensemble is not None

    def test_stacking_classifier(self):
        models = [("lr", LogisticRegression(max_iter=1000))]
        meta = LogisticRegression(max_iter=1000)
        ensemble = build_stacking_ensemble(models, meta, problem_type="classification", cv=2)
        assert ensemble is not None
