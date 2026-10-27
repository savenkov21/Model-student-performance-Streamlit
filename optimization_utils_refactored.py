from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import numpy as np
from bayes_opt import BayesianOptimization
from data_protocol import cv_folds, prepare_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"

CLASSIFICATION_SCORING = "f1_macro"
REGRESSION_SCORING = "neg_mean_squared_error"

CV_SPLITS = 5
RIDGE_ALPHA_GRID = np.logspace(-3, 3, 50)


@dataclass(frozen=True)
class OptimizationResult:
    best_params: Dict[str, Any]
    best_cv_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_cv_score": float(self.best_cv_score),
        }


def discretize_g3(y, q1, q3):
    """
    Перетворення неперервної цільової змінної у 3 класи
    за квартилями Q1, Q3.

    0: y <= Q1
    1: Q1 < y <= Q3
    2: y > Q3
    """
    y = np.asarray(y)
    cls = np.zeros_like(y, dtype=int)

    cls[y <= q1] = 0
    cls[(y > q1) & (y <= q3)] = 1
    cls[y > q3] = 2

    return cls


def _make_stratified_cv(random_state: int):
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)


def _make_kfold_cv(random_state: int):
    return KFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)


def _evaluate_cv(model, X_train, y_train, cv, scoring: str) -> float:
    folds=cv_folds(X_train,y_train,CV_SPLITS,42)
    scores = cross_val_score(
        estimator=prepare_model(model,calibrate=False), X=X_train, y=y_train,
        cv=folds, scoring=scoring, n_jobs=1, error_score="raise",
    )
    return float(np.mean(scores))


def _run_bayesian_optimization(
    objective_fn: Callable[..., float],
    pbounds: Mapping[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    random_state: int,
    verbose: int = 0,
) -> Dict[str, Any]:
    optimizer = BayesianOptimization(
        f=objective_fn,
        pbounds=dict(pbounds),
        random_state=random_state,
        verbose=verbose,
    )
    optimizer.maximize(init_points=int(init_points), n_iter=int(n_iter))
    return optimizer.max


def _result(best_params: Dict[str, Any], best_cv_score: float) -> Dict[str, Any]:
    return OptimizationResult(
        best_params=best_params,
        best_cv_score=float(best_cv_score),
    ).to_dict()


# =========================================================
# Strategy 1:
# direct_multiclass_classification
# Bayesian Optimization + StratifiedKFold
# objective = mean CV F1-macro
# =========================================================
def optimize_ridge_classifier_bo(
    X_train,
    y_train,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    cv = _make_stratified_cv(random_state)

    def ridge_cv(alpha):
        model = RidgeClassifier(alpha=float(alpha))
        return _evaluate_cv(model, X_train, y_train, cv, CLASSIFICATION_SCORING)

    best = _run_bayesian_optimization(
        objective_fn=ridge_cv,
        pbounds={"alpha": (0.001, 50.0)},
        init_points=init_points,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    best_params = {"alpha": float(best["params"]["alpha"])}
    return _result(best_params, best["target"])



def optimize_rf_classifier_bo(
    X_train,
    y_train,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    cv = _make_stratified_cv(random_state)

    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth < 1 else int(max_depth),
            min_samples_split=max(2, int(min_samples_split)),
            min_samples_leaf=max(1, int(min_samples_leaf)),
            random_state=random_state,
            n_jobs=-1,
        )
        return _evaluate_cv(model, X_train, y_train, cv, CLASSIFICATION_SCORING)

    best = _run_bayesian_optimization(
        objective_fn=rf_cv,
        pbounds={
            "n_estimators": (10, 200),
            "max_depth": (1, 20),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 4),
        },
        init_points=init_points,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    best_params = {
        "n_estimators": int(best["params"]["n_estimators"]),
        "max_depth": None if best["params"]["max_depth"] < 1 else int(best["params"]["max_depth"]),
        "min_samples_split": max(2, int(best["params"]["min_samples_split"])),
        "min_samples_leaf": max(1, int(best["params"]["min_samples_leaf"])),
    }
    return _result(best_params, best["target"])



def optimize_xgb_classifier_bo(
    X_train,
    y_train,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    cv = _make_stratified_cv(random_state)

    def xgb_cv(
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight,
        gamma,
        reg_lambda,
        reg_alpha,
    ):
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=max(1, int(max_depth)),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_weight=max(1, int(min_child_weight)),
            gamma=float(gamma),
            reg_lambda=float(reg_lambda),
            reg_alpha=float(reg_alpha),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        return _evaluate_cv(model, X_train, y_train, cv, CLASSIFICATION_SCORING)

    best = _run_bayesian_optimization(
        objective_fn=xgb_cv,
        pbounds={
            "n_estimators": (100, 600),
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "min_child_weight": (1, 10),
            "gamma": (0.0, 5.0),
            "reg_lambda": (0.0, 5.0),
            "reg_alpha": (0.0, 2.0),
        },
        init_points=init_points,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    best_params = {
        "n_estimators": int(best["params"]["n_estimators"]),
        "max_depth": max(1, int(best["params"]["max_depth"])),
        "learning_rate": float(best["params"]["learning_rate"]),
        "subsample": float(best["params"]["subsample"]),
        "colsample_bytree": float(best["params"]["colsample_bytree"]),
        "min_child_weight": max(1, int(best["params"]["min_child_weight"])),
        "gamma": float(best["params"]["gamma"]),
        "reg_lambda": float(best["params"]["reg_lambda"]),
        "reg_alpha": float(best["params"]["reg_alpha"]),
    }
    return _result(best_params, best["target"])


# =========================================================
# Strategy 2:
# regression_with_categorization
# Bayesian Optimization + KFold
# objective = mean CV NEGATIVE MSE
# =========================================================
def optimize_ridge_regressor_with_categorization(X_train, y_train_cont):
    scores=[_evaluate_cv(Ridge(alpha=float(alpha)),X_train,y_train_cont,CV_SPLITS,REGRESSION_SCORING) for alpha in RIDGE_ALPHA_GRID]
    best_idx=int(__import__('numpy').argmax(scores))
    best_alpha=float(RIDGE_ALPHA_GRID[best_idx]);best_cv_score=scores[best_idx]

    return _result({"alpha": best_alpha}, best_cv_score)



def optimize_rf_regressor_with_categorization_bo(
    X_train,
    y_train_cont,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    cv = _make_kfold_cv(random_state)

    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth < 1 else int(max_depth),
            min_samples_split=max(2, int(min_samples_split)),
            min_samples_leaf=max(1, int(min_samples_leaf)),
            random_state=random_state,
            n_jobs=-1,
        )
        return _evaluate_cv(model, X_train, y_train_cont, cv, REGRESSION_SCORING)

    best = _run_bayesian_optimization(
        objective_fn=rf_cv,
        pbounds={
            "n_estimators": (10, 200),
            "max_depth": (1, 20),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 4),
        },
        init_points=init_points,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    best_params = {
        "n_estimators": int(best["params"]["n_estimators"]),
        "max_depth": None if best["params"]["max_depth"] < 1 else int(best["params"]["max_depth"]),
        "min_samples_split": max(2, int(best["params"]["min_samples_split"])),
        "min_samples_leaf": max(1, int(best["params"]["min_samples_leaf"])),
    }
    return _result(best_params, best["target"])



def optimize_xgb_regressor_with_categorization_bo(
    X_train,
    y_train_cont,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    cv = _make_kfold_cv(random_state)

    def xgb_cv(
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight,
        gamma,
        reg_lambda,
        reg_alpha,
    ):
        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=max(1, int(max_depth)),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_weight=max(1, int(min_child_weight)),
            gamma=float(gamma),
            reg_lambda=float(reg_lambda),
            reg_alpha=float(reg_alpha),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        return _evaluate_cv(model, X_train, y_train_cont, cv, REGRESSION_SCORING)

    best = _run_bayesian_optimization(
        objective_fn=xgb_cv,
        pbounds={
            "n_estimators": (100, 600),
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "min_child_weight": (1, 10),
            "gamma": (0.0, 5.0),
            "reg_lambda": (0.0, 5.0),
            "reg_alpha": (0.0, 2.0),
        },
        init_points=init_points,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
    )

    best_params = {
        "n_estimators": int(best["params"]["n_estimators"]),
        "max_depth": max(1, int(best["params"]["max_depth"])),
        "learning_rate": float(best["params"]["learning_rate"]),
        "subsample": float(best["params"]["subsample"]),
        "colsample_bytree": float(best["params"]["colsample_bytree"]),
        "min_child_weight": max(1, int(best["params"]["min_child_weight"])),
        "gamma": float(best["params"]["gamma"]),
        "reg_lambda": float(best["params"]["reg_lambda"]),
        "reg_alpha": float(best["params"]["reg_alpha"]),
    }
    return _result(best_params, best["target"])


# =========================================================
# High-level wrapper for module_4.py
# =========================================================
def optimize_first_level_models(
    strategy_type,
    X_train,
    y_train,
    y_train_cont,
    init_points=10,
    n_iter=30,
    random_state=42,
):
    """
    Повертає результати оптимізації для трьох базових моделей.

    Output format:
    {
        "Ridge": {
            "best_params": {...},
            "best_cv_score": ...
        },
        "RandomForest": {
            "best_params": {...},
            "best_cv_score": ...
        },
        "XGBoost": {
            "best_params": {...},
            "best_cv_score": ...
        }
    }
    """
    if strategy_type == CLASSIFICATION_STRATEGY:
        return {
            "Ridge": optimize_ridge_classifier_bo(
                X_train=X_train,
                y_train=y_train,
                init_points=init_points,
                n_iter=n_iter,
                random_state=random_state,
            ),
            "RandomForest": optimize_rf_classifier_bo(
                X_train=X_train,
                y_train=y_train,
                init_points=init_points,
                n_iter=n_iter,
                random_state=random_state,
            ),
            "XGBoost": optimize_xgb_classifier_bo(
                X_train=X_train,
                y_train=y_train,
                init_points=init_points,
                n_iter=n_iter,
                random_state=random_state,
            ),
        }

    if strategy_type == REGRESSION_STRATEGY:
        return {
            "Ridge": optimize_ridge_regressor_with_categorization(
                X_train=X_train,
                y_train_cont=y_train_cont,
            ),
            "RandomForest": optimize_rf_regressor_with_categorization_bo(
                X_train=X_train,
                y_train_cont=y_train_cont,
                init_points=init_points,
                n_iter=n_iter,
                random_state=random_state,
            ),
            "XGBoost": optimize_xgb_regressor_with_categorization_bo(
                X_train=X_train,
                y_train_cont=y_train_cont,
                init_points=init_points,
                n_iter=n_iter,
                random_state=random_state,
            ),
        }

    raise ValueError(
        "Unsupported strategy_type. "
        f"Expected '{CLASSIFICATION_STRATEGY}' or '{REGRESSION_STRATEGY}', got: {strategy_type!r}"
    )
