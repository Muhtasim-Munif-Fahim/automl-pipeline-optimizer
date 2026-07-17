"""Ensemble methods: voting, stacking, bagging wrappers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor


def build_voting_ensemble(models: list[tuple[str, Any]], problem_type: str = "classification", voting: str = "soft"):
    if "classification" in problem_type:
        return VotingClassifier(estimators=models, voting=voting)
    return VotingRegressor(estimators=models)


def build_stacking_ensemble(models: list[tuple[str, Any]], meta_model: Any, problem_type: str = "classification", cv: int = 5):
    if "classification" in problem_type:
        return StackingClassifier(estimators=models, final_estimator=meta_model, cv=cv)
    return StackingRegressor(estimators=models, final_estimator=meta_model, cv=cv)
