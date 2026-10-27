# module_4_visual_refined.py

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from data_protocol import wrap_models, cv_folds, reset_downstream
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

from optimization_utils_refactored import optimize_first_level_models


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"

FIRST_LEVEL_STATE_KEYS = [
    "first_level_models",
    "first_level_results",
    "first_level_predictions",
    "first_level_predictions_cls",
    "first_level_params",
    "first_level_strategy_type",
    "optimization_results",
    "optimization_mode",
]

REQUIRED_SPLIT_KEYS = [
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "strategy_type",
    "split_ready",
    "Q1",
    "Q3",
]


@dataclass(frozen=True)
class TrainingContext:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    strategy_type: str
    q1: float
    q3: float
    y_train: pd.Series
    y_test: pd.Series
    y_train_cont: pd.Series
    y_test_cont: pd.Series
    target_col: str


@dataclass(frozen=True)
class OptimizationConfig:
    init_points: int
    n_iter: int


@dataclass(frozen=True)
class SelectedHyperparameters:
    ridge_alpha: float
    rf_params: dict[str, Any]
    xgb_params: dict[str, Any]


@dataclass(frozen=True)
class TrainingArtifacts:
    models: dict[str, Any]
    results: dict[str, dict[str, Any]]
    predictions: dict[str, np.ndarray]
    predictions_cls: dict[str, np.ndarray]
    params: dict[str, Any]


def discretize_g3(y, q1, q3):
    y = np.asarray(y)
    cls = np.zeros_like(y, dtype=int)
    cls[y <= q1] = 0
    cls[(y > q1) & (y <= q3)] = 1
    cls[y > q3] = 2
    return cls


def _results_to_json_bytes(results_dict: dict[str, Any]) -> BytesIO:
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(results_dict, indent=4, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return json_bytes


def _safe_classification_report(y_true, y_pred) -> dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def _strategy_label(strategy_type: str) -> str:
    labels = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return labels.get(strategy_type, strategy_type)


def _cv_metric_label(strategy_type: str) -> str:
    return "Середній CV F1-macro" if strategy_type == CLASSIFICATION_STRATEGY else "Середній CV score (neg. MSE)"


def _primary_metric_label(strategy_type: str) -> str:
    return "F1 weighted" if strategy_type == CLASSIFICATION_STRATEGY else "F1 weighted after discretization"


def _model_display_name(model_name: str) -> str:
    labels = {
        "Ridge": "Ridge",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "RidgeClassifier": "Ridge Classifier",
        "RandomForestClassifier": "Random Forest Classifier",
        "XGBClassifier": "XGBoost Classifier",
        "RandomForestRegressor": "Random Forest Regressor",
        "XGBRegressor": "XGBoost Regressor",
    }
    return labels.get(model_name, model_name)


def _params_to_dataframe(params_dict: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in params_dict.items():
        normalized = "None" if value is None else round(value, 6) if isinstance(value, float) else value
        rows.append({"Параметр": key, "Значення": normalized})
    return pd.DataFrame(rows)


def _metric(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _clear_first_level_state() -> None:
    for key in FIRST_LEVEL_STATE_KEYS:
        st.session_state.pop(key, None)


def _check_split_ready() -> bool:
    missing = [key for key in REQUIRED_SPLIT_KEYS if key not in st.session_state]
    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False
    if not st.session_state.get("split_ready", False):
        st.warning("Спочатку виконайте train/test split на сторінці попередньої обробки.")
        return False
    return True


def _build_training_context() -> TrainingContext:
    return TrainingContext(
        X_train=st.session_state.X_train_raw,
        X_test=st.session_state.X_test_raw,
        strategy_type=st.session_state.strategy_type,
        q1=float(st.session_state.Q1),
        q3=float(st.session_state.Q3),
        y_train=st.session_state.y_train,
        y_test=st.session_state.y_test,
        y_train_cont=st.session_state.y_train_cont,
        y_test_cont=st.session_state.y_test_cont,
        target_col=st.session_state.get("target_col", "—"),
    )


def _get_selected_hyperparameters(optimization_results: dict[str, Any]) -> SelectedHyperparameters:
    return SelectedHyperparameters(
        ridge_alpha=float(optimization_results["Ridge"]["best_params"]["alpha"]),
        rf_params=dict(optimization_results["RandomForest"]["best_params"]),
        xgb_params=dict(optimization_results["XGBoost"]["best_params"]),
    )


def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .m4-hero {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(70,130,180,0.08), rgba(255,255,255,0.02));
            margin-bottom: 0.8rem;
        }
        .m4-section-title {
            margin-top: 0.5rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m4-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }
        .m4-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            background: rgba(255,255,255,0.02);
            min-height: 110px;
        }
        .m4-card-kicker {
            font-size: 0.82rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.35rem;
        }
        .m4-card-value {
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.3rem;
        }
        .m4-card-note {
            font-size: 0.86rem;
            color: rgba(120,120,120,0.92);
        }
        .m4-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="m4-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="m4-subtle">{subtitle}</div>', unsafe_allow_html=True)


def _render_html_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f'''
        <div class="m4-card">
            <div class="m4-card-kicker">{title}</div>
            <div class="m4-card-value">{value}</div>
            <div class="m4-card-note">{note}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_hero(context: TrainingContext) -> None:
    strategy = _strategy_label(context.strategy_type)
    st.markdown(
        f'''
        <div class="m4-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Навчання базових моделей першого рівня</div>
            <div style="margin-bottom:0.6rem;">На цьому етапі виконується оптимізація гіперпараметрів для трьох базових моделей, а потім виконується їх навчання на підготовлених даних.</div>
            <span class="m4-pill">Random Forest</span>
            <span class="m4-pill">XGBoost</span>
            <span class="m4-pill">Ridge</span>
            <span class="m4-pill">Bayesian Optimization</span>
            <span class="m4-pill">k-fold cross-validation</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_data_snapshot(context: TrainingContext) -> None:
    _render_section_header(
        "1. Поточний стан даних",
        "Короткий огляд активної конфігурації, яка буде використана для оптимізації та навчання моделей.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Стратегія", _strategy_label(context.strategy_type))
    with c2:
        _render_html_card("Навчальна вибірка", f"{context.X_train.shape[0]} × {context.X_train.shape[1]}")
    with c3:
        _render_html_card("Тестова вибірка", f"{context.X_test.shape[0]} × {context.X_test.shape[1]}")
    with c4:
        _render_html_card("Цільова ознака", context.target_col)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    with st.expander("Показати деталі розбиття та категоризації", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Q1", _metric(context.q1))
        d2.metric("Q3", _metric(context.q3))
        d3.metric("Ознак після preprocessing", str(context.X_train.shape[1]))

        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.info(
                "Для прямої багатокласової класифікації неперервна цільова змінна спочатку категоризується за порогами Q1 і Q3, обчисленими на train-вибірці."
            )
        else:
            st.info(
                "Для регресійної стратегії моделі прогнозують неперервне значення, а класифікаційні метрики обчислюються додатково після категоризації прогнозів."
            )

        st.dataframe(pd.DataFrame({"Ознака": list(context.X_train.columns)}), width="stretch", hide_index=True)


def _render_bayesian_block(strategy_type: str) -> tuple[OptimizationConfig, bool, bool]:
    _render_section_header(
        "2. Bayesian Optimization",
        "Спочатку підбираються найкращі гіперпараметри для Ridge, Random Forest і XGBoost.",
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        with st.container(border=True):
            st.markdown("**Налаштування оптимізації**")
            c1, c2 = st.columns(2)
            with c1:
                init_points = st.slider(
                    "init_points",
                    min_value=3,
                    max_value=20,
                    value=int(st.session_state.get("bo_init_points", 10)),
                    step=1,
                    help="Кількість стартових випадкових конфігурацій.",
                )
            with c2:
                n_iter = st.slider(
                    "n_iter",
                    min_value=5,
                    max_value=40,
                    value=int(st.session_state.get("bo_n_iter", 30)),
                    step=1,
                    help="Кількість основних ітерацій пошуку.",
                )

            st.session_state.bo_init_points = init_points
            st.session_state.bo_n_iter = n_iter
            st.caption(f"Метрика оптимізації: {_cv_metric_label(strategy_type)}")

            b1, b2 = st.columns([1.6, 1])
            optimize_clicked = b1.button("Запустити оптимізацію", width="stretch", type="primary")
            clear_clicked = b2.button("Очистити", width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("**Коротко про метод**")
            st.write(
                "Bayesian Optimization не перебирає всі комбінації параметрів, а послідовно вибирає найперспективніші конфігурації на основі попередніх оцінок."
            )
            st.write(
                "Для класифікації оптимізується F1-macro у CV, а для регресійної стратегії — CV score на основі neg. MSE."
            )
            with st.expander("Детальніше", expanded=False):
                st.markdown(
                    """
1. Оцінюються стартові випадкові точки.
2. Будується наближена модель функції якості.
3. Acquisition function обирає наступну перспективну конфігурацію.
4. Після завершення ітерацій фіксується найкращий набір параметрів.
                    """
                )

    return OptimizationConfig(init_points=init_points, n_iter=n_iter), optimize_clicked, clear_clicked


def _render_optimization_results(optimization_results: dict[str, Any], strategy_type: str) -> None:
    _render_section_header(
        "3. Результати оптимізації",
        "Оптимальні конфігурації, які будуть використані для навчання моделей першого рівня.",
    )
    if not optimization_results:
        st.info("Після запуску оптимізації тут з'являться найкращі конфігурації для кожної моделі.")
        return

    summary_rows = []
    for model_name, result in optimization_results.items():
        summary_rows.append(
            {
                "Модель": _model_display_name(model_name),
                _cv_metric_label(strategy_type): round(float(result["best_cv_score"]), 4),
                "К-сть параметрів": len(result.get("best_params", {})),
            }
        )
    df = pd.DataFrame(summary_rows)

    c1, c2, c3 = st.columns(3)
    best_idx = df[_cv_metric_label(strategy_type)].idxmax()
    best_model = df.loc[best_idx, "Модель"]
    c1.metric("Оптимізовано моделей", str(len(df)))
    c2.metric("Найкращий CV результат", _metric(df.loc[best_idx, _cv_metric_label(strategy_type)]))
    c3.metric("Лідер оптимізації", best_model)

    st.dataframe(df, width="stretch", hide_index=True)

    tabs = st.tabs([_model_display_name(name) for name in optimization_results.keys()])
    for tab, (model_name, result) in zip(tabs, optimization_results.items()):
        with tab:
            left, right = st.columns([0.8, 1.2])
            left.metric(_cv_metric_label(strategy_type), _metric(result["best_cv_score"]))
            right.dataframe(_params_to_dataframe(result.get("best_params", {})), width="stretch", hide_index=True)


def _render_training_controls() -> tuple[bool, bool]:
    _render_section_header(
        "4. Навчання базових моделей",
        "Після оптимізації гіперпараметрів виконайте навчання Ridge, Random Forest і XGBoost на train-вибірці.",
    )
    with st.container(border=True):
        c1, c2 = st.columns([1.6, 1])
        train_clicked = c1.button("Навчити моделі першого рівня", width="stretch", type="primary")
        clear_clicked = c2.button("Очистити результати навчання", width="stretch")
    return train_clicked, clear_clicked


def _build_classification_models(params: SelectedHyperparameters) -> dict[str, Any]:
    return wrap_models({
        "RidgeClassifier": RidgeClassifier(alpha=params.ridge_alpha),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=int(params.rf_params["n_estimators"]),
            max_depth=params.rf_params["max_depth"],
            min_samples_split=int(params.rf_params["min_samples_split"]),
            min_samples_leaf=int(params.rf_params["min_samples_leaf"]),
            random_state=42,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=int(params.xgb_params["n_estimators"]),
            max_depth=int(params.xgb_params["max_depth"]),
            learning_rate=float(params.xgb_params["learning_rate"]),
            subsample=float(params.xgb_params["subsample"]),
            colsample_bytree=float(params.xgb_params["colsample_bytree"]),
            min_child_weight=int(params.xgb_params["min_child_weight"]),
            gamma=float(params.xgb_params["gamma"]),
            reg_lambda=float(params.xgb_params["reg_lambda"]),
            reg_alpha=float(params.xgb_params["reg_alpha"]),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    })


def _build_regression_models(params: SelectedHyperparameters) -> dict[str, Any]:
    return wrap_models({
        "Ridge": Ridge(alpha=params.ridge_alpha),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=int(params.rf_params["n_estimators"]),
            max_depth=params.rf_params["max_depth"],
            min_samples_split=int(params.rf_params["min_samples_split"]),
            min_samples_leaf=int(params.rf_params["min_samples_leaf"]),
            random_state=42,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=int(params.xgb_params["n_estimators"]),
            max_depth=int(params.xgb_params["max_depth"]),
            learning_rate=float(params.xgb_params["learning_rate"]),
            subsample=float(params.xgb_params["subsample"]),
            colsample_bytree=float(params.xgb_params["colsample_bytree"]),
            min_child_weight=int(params.xgb_params["min_child_weight"]),
            gamma=float(params.xgb_params["gamma"]),
            reg_lambda=float(params.xgb_params["reg_lambda"]),
            reg_alpha=float(params.xgb_params["reg_alpha"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    })


def _train_single_classifier(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))
    result = {
        "model_name": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "F1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": _safe_classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classes": labels.tolist(),
    }
    return model, np.asarray(y_pred), result


def _train_single_regressor_with_categorization(model, model_name, X_train, X_test, y_train_cont, y_test_cont, q1, q3):
    model.fit(X_train, y_train_cont)
    y_pred_cont = model.predict(X_test)
    mse = mean_squared_error(y_test_cont, y_pred_cont)
    y_test_cls = discretize_g3(y_test_cont, q1, q3)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    labels_cls = np.unique(np.concatenate([np.asarray(y_test_cls), np.asarray(y_pred_cls)]))
    result = {
        "model_name": model_name,
        "MAE": float(mean_absolute_error(y_test_cont, y_pred_cont)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test_cont, y_pred_cont)),
        "Accuracy_after_discretization": float(accuracy_score(y_test_cls, y_pred_cls)),
        "Balanced_accuracy_after_discretization": float(balanced_accuracy_score(y_test_cls, y_pred_cls)),
        "F1_weighted_after_discretization": float(f1_score(y_test_cls, y_pred_cls, average="weighted")),
        "classification_report_after_discretization": _safe_classification_report(y_test_cls, y_pred_cls),
        "confusion_matrix_after_discretization": confusion_matrix(y_test_cls, y_pred_cls, labels=labels_cls).tolist(),
        "classes_after_discretization": labels_cls.tolist(),
    }
    return model, np.asarray(y_pred_cont), np.asarray(y_pred_cls), result


def _train_models_for_strategy(context: TrainingContext, params: SelectedHyperparameters) -> TrainingArtifacts:
    reset_downstream(5)
    trained_models: dict[str, Any] = {}
    all_results: dict[str, dict[str, Any]] = {}
    all_predictions: dict[str, np.ndarray] = {}
    all_predictions_cls: dict[str, np.ndarray] = {}

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        models = _build_classification_models(params)
        for model_name, model in models.items():
            trained_model, y_pred, result = _train_single_classifier(
                model=model,
                model_name=model_name,
                X_train=context.X_train,
                X_test=context.X_test,
                y_train=context.y_train,
                y_test=context.y_test,
            )
            trained_models[model_name] = trained_model
            all_results[model_name] = result
            all_predictions[model_name] = np.asarray(y_pred)
            all_predictions_cls[model_name] = np.asarray(y_pred)
    else:
        models = _build_regression_models(params)
        for model_name, model in models.items():
            trained_model, y_pred_cont, y_pred_cls, result = _train_single_regressor_with_categorization(
                model=model,
                model_name=model_name,
                X_train=context.X_train,
                X_test=context.X_test,
                y_train_cont=context.y_train_cont,
                y_test_cont=context.y_test_cont,
                q1=context.q1,
                q3=context.q3,
            )
            trained_models[model_name] = trained_model
            all_results[model_name] = result
            all_predictions[model_name] = np.asarray(y_pred_cont)
            all_predictions_cls[model_name] = np.asarray(y_pred_cls)

    return TrainingArtifacts(
        models=trained_models,
        results=all_results,
        predictions=all_predictions,
        predictions_cls=all_predictions_cls,
        params={
            "source": "bayesian_optimization",
            "ridge_alpha": params.ridge_alpha,
            "rf_params": params.rf_params,
            "xgb_params": params.xgb_params,
        },
    )


def _save_training_artifacts(context: TrainingContext, artifacts: TrainingArtifacts) -> None:
    st.session_state.first_level_models = artifacts.models
    st.session_state.first_level_results = artifacts.results
    st.session_state.first_level_predictions = artifacts.predictions
    st.session_state.first_level_predictions_cls = artifacts.predictions_cls
    st.session_state.first_level_params = artifacts.params
    st.session_state.first_level_strategy_type = context.strategy_type
    st.session_state.optimization_mode = "bayesian_optimization"


def _build_comparison_dataframe(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    if strategy_type == CLASSIFICATION_STRATEGY:
        return pd.DataFrame(
            {
                model_name: {
                    "Accuracy": model_res["Accuracy"],
                    "Balanced Accuracy": model_res["Balanced_accuracy"],
                    "F1 weighted": model_res["F1_weighted"],
                }
                for model_name, model_res in results.items()
            }
        ).T
    return pd.DataFrame(
        {
            model_name: {
                "MAE": model_res["MAE"],
                "MSE": model_res["MSE"],
                "RMSE": model_res["RMSE"],
                "R2": model_res["R2"],
                "Accuracy after discretization": model_res["Accuracy_after_discretization"],
                "Balanced accuracy after discretization": model_res["Balanced_accuracy_after_discretization"],
                "F1 weighted after discretization": model_res["F1_weighted_after_discretization"],
            }
            for model_name, model_res in results.items()
        }
    ).T


def _plot_confusion_matrix(cm, class_labels, model_name, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    title = f"Confusion Matrix: {_model_display_name(model_name)}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_actual_vs_predicted(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Actual vs Predicted: {_model_display_name(model_name)}")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_results_summary(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    _render_section_header(
        "5. Порівняння результатів",
        "Підсумок і таблиця метрик.",
    )
    comparison_df = _build_comparison_dataframe(results, strategy_type)
    comparison_df.index = [_model_display_name(name) for name in comparison_df.index]
    metric_name = _primary_metric_label(strategy_type)
    best_model = comparison_df[metric_name].idxmax()
    best_value = comparison_df.loc[best_model, metric_name]

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_html_card("Оцінено моделей", str(len(comparison_df)), "Кількість базових моделей першого рівня")
    with c2:
        _render_html_card("Найкраща модель", best_model, "Лідер за головною метрикою поточної стратегії")
    with c3:
        _render_html_card(metric_name, _metric(best_value), "Найвище значення серед усіх порівнюваних моделей")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    
    
    display_df = comparison_df.reset_index().rename(columns={"index": "Модель"})
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Модель": st.column_config.TextColumn("Модель", width="large"),
            },
        )

    return comparison_df


def _render_classification_model_details(selected_model: str, selected_result: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric(selected_result["Accuracy"]))
    c2.metric("Balanced Accuracy", _metric(selected_result["Balanced_accuracy"]))
    c3.metric("F1 weighted", _metric(selected_result["F1_weighted"]))
    st.dataframe(pd.DataFrame(selected_result["classification_report"]).T, width="stretch")
    _plot_confusion_matrix(np.asarray(selected_result["confusion_matrix"]), selected_result["classes"], selected_model)


def _render_regression_model_details(selected_model: str, selected_result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _metric(selected_result["MAE"]))
    c2.metric("MSE", _metric(selected_result["MSE"]))
    c3.metric("RMSE", _metric(selected_result["RMSE"]))
    c4.metric("R²", _metric(selected_result["R2"]))
    _plot_actual_vs_predicted(np.asarray(st.session_state.y_test_cont), st.session_state.first_level_predictions[selected_model], selected_model)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy after discretization", _metric(selected_result["Accuracy_after_discretization"]))
    m2.metric("Balanced accuracy after discretization", _metric(selected_result["Balanced_accuracy_after_discretization"]))
    m3.metric("F1 weighted after discretization", _metric(selected_result["F1_weighted_after_discretization"]))
    st.dataframe(pd.DataFrame(selected_result["classification_report_after_discretization"]).T, width="stretch")
    _plot_confusion_matrix(
        np.asarray(selected_result["confusion_matrix_after_discretization"]),
        selected_result["classes_after_discretization"],
        selected_model,
        title_suffix="after discretization",
    )


def _render_model_details_tabs(results: dict[str, dict[str, Any]], strategy_type: str) -> None:
    _render_section_header(
        "6. Деталі моделей",
        "Тут можна переглянути метрики, classification report і візуалізації для кожної базової моделі.",
    )
    tabs = st.tabs([_model_display_name(name) for name in results.keys()])
    for tab, model_name in zip(tabs, results.keys()):
        with tab:
            st.markdown(f"**{_model_display_name(model_name)}**")
            selected_result = results[model_name]
            if strategy_type == CLASSIFICATION_STRATEGY:
                _render_classification_model_details(model_name, selected_result)
            else:
                _render_regression_model_details(model_name, selected_result)


def _render_export_section(results: dict[str, dict[str, Any]], strategy_type: str) -> None:
    _render_section_header(
        "7. Експорт результатів",
        "Збереження підсумків першого рівня для подальшого аналізу.",
    )
    payload = {
        "strategy_type": strategy_type,
        "target_col": st.session_state.get("target_col"),
        "Q1": st.session_state.get("Q1"),
        "Q3": st.session_state.get("Q3"),
        "optimization_results": st.session_state.get("optimization_results"),
        "first_level_params": st.session_state.get("first_level_params", {}),
        "models": results,
    }
    left, right = st.columns([1.2, 0.8])
    with left:
        st.download_button(
            on_click="ignore",
            label="Завантажити результати (JSON)",
            data=_results_to_json_bytes(payload),
            file_name="first_level_models_results.json",
            mime="application/json",
            width="stretch",
        )
    with right:
        st.info("JSON містить параметри оптимізації, результати всіх моделей та службову інформацію для наступних етапів.")


def _run_optimization(context: TrainingContext, config: OptimizationConfig) -> None:
    with st.spinner("Триває Bayesian Optimization..."):
        st.session_state.optimization_results = optimize_first_level_models(
            strategy_type=context.strategy_type,
            X_train=context.X_train,
            y_train=context.y_train,
            y_train_cont=context.y_train_cont,
            init_points=config.init_points,
            n_iter=config.n_iter,
        )
    st.success("Оптимізацію гіперпараметрів завершено!")


def _run_first_level_training(context: TrainingContext, optimization_results: dict[str, Any]) -> None:
    if optimization_results is None:
        st.warning("Спочатку виконайте Bayesian Optimization.")
        return
    params = _get_selected_hyperparameters(optimization_results)
    with st.spinner("Триває навчання базових моделей..."):
        artifacts = _train_models_for_strategy(context, params)
        _save_training_artifacts(context, artifacts)
    st.success("Базові моделі 1-го рівня успішно навчені!")


def _render_results_if_ready(context: TrainingContext) -> None:
    if "first_level_results" not in st.session_state:
        st.info("Після навчання тут з’явиться порівняння моделей, детальні метрики та експорт результатів.")
        return
    if st.session_state.get("first_level_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть базові моделі.")
        return
    results = st.session_state.first_level_results
    _render_results_summary(results, context.strategy_type)
    _render_model_details_tabs(results, context.strategy_type)
    _render_export_section(results, context.strategy_type)


def page_training_first_level_models() -> None:
    _inject_page_style()
    if not _check_split_ready():
        return

    context = _build_training_context()
    _render_hero(context)
    _render_data_snapshot(context)

    optimization_config, optimize_clicked, clear_opt_clicked = _render_bayesian_block(context.strategy_type)
    if clear_opt_clicked:
        st.session_state.pop("optimization_results", None)
        st.success("Результати оптимізації очищено!")

    if optimize_clicked:
        _run_optimization(context, optimization_config)

    optimization_results = st.session_state.get("optimization_results")
    _render_optimization_results(optimization_results, context.strategy_type)

    train_clicked, clear_train_clicked = _render_training_controls()
    if clear_train_clicked:
        _clear_first_level_state()
        st.success("Результати базових моделей очищено!")
        return

    if train_clicked:
        _run_first_level_training(context, optimization_results)

    _render_results_if_ready(context)
