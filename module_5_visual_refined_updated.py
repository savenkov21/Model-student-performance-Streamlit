from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from data_protocol import wrap_models, cv_folds, reset_downstream

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
RANDOM_STATE = 42


@dataclass(frozen=True)
class MetaFeatureContext:
    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    strategy_type: str
    params: Mapping[str, object]
    q1: float | None
    q3: float | None
    target_col: str | None


@dataclass(frozen=True)
class MetaFeatureConfig:
    cv_folds: int
    passthrough: bool


@dataclass(frozen=True)
class MetaFeatureArtifacts:
    meta_X_train: pd.DataFrame
    meta_X_test: pd.DataFrame
    feature_names: List[str]
    base_model_names: List[str]
    full_train_models: Dict[str, object]
    generation_params: Dict[str, object]


# =========================================================
# State / validation helpers
# =========================================================
def _check_ready_for_metafeatures() -> bool:
    required_keys = [
        "X_train",
        "X_test",
        "strategy_type",
        "first_level_params",
        "first_level_strategy_type",
        "Q1",
        "Q3",
    ]
    missing = [key for key in required_keys if key not in st.session_state]

    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False

    if st.session_state.get("first_level_strategy_type") != st.session_state.get("strategy_type"):
        st.warning("Стратегія змінилася. Перенавчіть базові моделі на сторінці 'Навчання базових моделей'.")
        return False

    return True



def _clear_metafeatures_state() -> None:
    keys = [
        "meta_X_train",
        "meta_X_test",
        "meta_feature_names",
        "meta_generation_params",
        "meta_base_models",
        "meta_full_train_models",
        "meta_ready",
        "meta_strategy_type",
    ]
    for key in keys:
        st.session_state.pop(key, None)



def _results_to_json_bytes(results_dict: Mapping[str, object]) -> BytesIO:
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(results_dict, indent=4, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return json_bytes



def _load_context() -> MetaFeatureContext:
    return MetaFeatureContext(
        X_train=st.session_state.X_train_raw,
        X_test=st.session_state.X_test_raw,
        strategy_type=st.session_state.strategy_type,
        params=st.session_state.first_level_params,
        q1=st.session_state.get("Q1"),
        q3=st.session_state.get("Q3"),
        target_col=st.session_state.get("target_col"),
    )



def _strategy_label(strategy_type: str) -> str:
    mapping = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return mapping.get(strategy_type, strategy_type)



def _short_model_name(name: str) -> str:
    mapping = {
        "RidgeClassifier": "Ridge",
        "RandomForestClassifier": "Random Forest",
        "XGBClassifier": "XGBoost",
        "Ridge": "Ridge",
        "RandomForestRegressor": "Random Forest",
        "XGBRegressor": "XGBoost",
    }
    return mapping.get(name, name)



def _bool_label(value: bool) -> str:
    return "Так" if bool(value) else "Ні"



def _metric(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


# =========================================================
# Model factories
# =========================================================
def _get_classification_models(ridge_alpha, rf_params, xgb_params) -> Dict[str, object]:
    ridge_base = RidgeClassifier(alpha=float(ridge_alpha))
    ridge_calibrated = CalibratedClassifierCV(estimator=ridge_base, cv=5)

    return wrap_models({
        "RidgeClassifier": ridge_calibrated,
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_split=int(rf_params["min_samples_split"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            min_child_weight=int(xgb_params["min_child_weight"]),
            gamma=float(xgb_params["gamma"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    })



def _get_regression_models(ridge_alpha, rf_params, xgb_params) -> Dict[str, object]:
    return wrap_models({
        "Ridge": Ridge(alpha=float(ridge_alpha)),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_split=int(rf_params["min_samples_split"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            min_child_weight=int(xgb_params["min_child_weight"]),
            gamma=float(xgb_params["gamma"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    })



def _build_base_models(strategy_type: str, params: Mapping[str, object]) -> Dict[str, object]:
    ridge_alpha = params["ridge_alpha"]
    rf_params = params["rf_params"]
    xgb_params = params["xgb_params"]

    if strategy_type == CLASSIFICATION_STRATEGY:
        return _get_classification_models(ridge_alpha, rf_params, xgb_params)
    if strategy_type == REGRESSION_STRATEGY:
        return _get_regression_models(ridge_alpha, rf_params, xgb_params)
    raise ValueError(f"Невідома стратегія: {strategy_type}")


# =========================================================
# Prediction helpers
# =========================================================
def _predict_for_meta_classification(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)



def _predict_for_meta_regression(model, X) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


# =========================================================
# OOF generation
# =========================================================
def _generate_oof_metafeatures_classification(
    models: Mapping[str, object],
    X_train,
    y_train,
    X_test,
    cv: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[str]]:
    X_train_np = X_train.copy()
    X_test_np = X_test.copy()
    y_train_np = np.asarray(y_train)

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    classes_ = np.unique(y_train_np)
    n_classes = len(classes_)

    meta_train_blocks = []
    meta_test_blocks = []
    feature_names: List[str] = []
    fitted_models_on_full_train: Dict[str, object] = {}

    for model_name, base_model in models.items():
        oof_pred = np.zeros((len(X_train_np), n_classes), dtype=float)
        test_fold_preds = []
        fold_models = []

        for train_idx, valid_idx in cv_folds(X_train_np,y_train_np,cv,RANDOM_STATE):
            X_tr = X_train_np.iloc[train_idx]
            X_val = X_train_np.iloc[valid_idx]
            y_tr = y_train_np[train_idx]

            model = clone(base_model)
            model.fit(X_tr, y_tr)
            fold_models.append(model)

            oof_pred[valid_idx, :] = _predict_for_meta_classification(model, X_val)
            test_fold_preds.append(_predict_for_meta_classification(model, X_test_np))

        st.session_state.setdefault("fold_prediction_models",{})[model_name]=fold_models
        meta_train_blocks.append(oof_pred)
        meta_test_blocks.append(np.mean(np.stack(test_fold_preds, axis=0), axis=0))
        feature_names.extend([f"meta_{model_name}_proba_{cls}" for cls in classes_])

        full_model = clone(base_model)
        full_model.fit(X_train_np, y_train_np)
        fitted_models_on_full_train[model_name] = full_model

    meta_X_train = pd.DataFrame(
        np.hstack(meta_train_blocks),
        columns=feature_names,
        index=getattr(X_train, "index", None),
    )
    meta_X_test = pd.DataFrame(
        np.hstack(meta_test_blocks),
        columns=feature_names,
        index=getattr(X_test, "index", None),
    )
    return meta_X_train, meta_X_test, fitted_models_on_full_train, feature_names



def _generate_oof_metafeatures_regression(
    models: Mapping[str, object],
    X_train,
    y_train,
    X_test,
    cv: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[str]]:
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = len(models)

    meta_train = np.zeros((n_train, n_models), dtype=float)
    meta_test = np.zeros((n_test, n_models), dtype=float)

    splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    X_train_np = X_train.copy()
    X_test_np = X_test.copy()
    y_train_np = np.asarray(y_train)

    fitted_models_on_full_train: Dict[str, object] = {}
    feature_names: List[str] = []

    for model_idx, (model_name, base_model) in enumerate(models.items()):
        oof_pred = np.zeros(n_train, dtype=float)
        test_fold_preds = []
        fold_models = []

        for train_idx, valid_idx in cv_folds(X_train_np,y_train_np,cv,RANDOM_STATE):
            X_tr = X_train_np.iloc[train_idx]
            X_val = X_train_np.iloc[valid_idx]
            y_tr = y_train_np[train_idx]

            model = clone(base_model)
            model.fit(X_tr, y_tr)
            fold_models.append(model)

            oof_pred[valid_idx] = _predict_for_meta_regression(model, X_val)
            test_fold_preds.append(_predict_for_meta_regression(model, X_test_np))

        st.session_state.setdefault("fold_prediction_models",{})[model_name]=fold_models
        meta_train[:, model_idx] = oof_pred
        meta_test[:, model_idx] = np.mean(np.vstack(test_fold_preds), axis=0)
        feature_names.append(f"meta_{model_name}")

        full_model = clone(base_model)
        full_model.fit(X_train_np, y_train_np)
        fitted_models_on_full_train[model_name] = full_model

    meta_X_train = pd.DataFrame(meta_train, columns=feature_names, index=getattr(X_train, "index", None))
    meta_X_test = pd.DataFrame(meta_test, columns=feature_names, index=getattr(X_test, "index", None))
    return meta_X_train, meta_X_test, fitted_models_on_full_train, feature_names


# =========================================================
# Assembly helpers
# =========================================================
def _append_original_features(meta_X_train: pd.DataFrame, meta_X_test: pd.DataFrame, X_train, X_test):
    X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
    meta_train_df = meta_X_train.reset_index(drop=True)
    meta_test_df = meta_X_test.reset_index(drop=True)

    combined_train = pd.concat([X_train_df, meta_train_df], axis=1)
    combined_test = pd.concat([X_test_df, meta_test_df], axis=1)
    return combined_train, combined_test, combined_train.columns.tolist()



def _build_metafeatures(context: MetaFeatureContext, config: MetaFeatureConfig) -> MetaFeatureArtifacts:
    reset_downstream(6)
    st.session_state.fold_prediction_models={}
    base_models = _build_base_models(context.strategy_type, context.params)

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        target = st.session_state.y_train
        meta_X_train, meta_X_test, full_train_models, feature_names = _generate_oof_metafeatures_classification(
            models=base_models,
            X_train=context.X_train,
            y_train=target,
            X_test=context.X_test,
            cv=config.cv_folds,
        )
    else:
        target = st.session_state.y_train_cont
        meta_X_train, meta_X_test, full_train_models, feature_names = _generate_oof_metafeatures_regression(
            models=base_models,
            X_train=context.X_train,
            y_train=target,
            X_test=context.X_test,
            cv=config.cv_folds,
        )

    if config.passthrough:
        meta_X_train, meta_X_test, feature_names = _append_original_features(
            meta_X_train=meta_X_train,
            meta_X_test=meta_X_test,
            X_train=st.session_state.X_train,
            X_test=st.session_state.X_test,
        )

    return MetaFeatureArtifacts(
        meta_X_train=meta_X_train,
        meta_X_test=meta_X_test,
        feature_names=feature_names,
        base_model_names=list(base_models.keys()),
        full_train_models=full_train_models,
        generation_params={
            "cv_folds": int(config.cv_folds),
            "passthrough": bool(config.passthrough),
        },
    )



def _save_metafeatures(artifacts: MetaFeatureArtifacts, strategy_type: str) -> None:
    st.session_state.meta_X_train = artifacts.meta_X_train
    st.session_state.meta_X_test = artifacts.meta_X_test
    st.session_state.meta_feature_names = artifacts.feature_names
    st.session_state.meta_generation_params = artifacts.generation_params
    st.session_state.meta_base_models = artifacts.base_model_names
    st.session_state.meta_full_train_models = artifacts.full_train_models
    st.session_state.meta_ready = True
    st.session_state.meta_strategy_type = strategy_type


# =========================================================
# Rendering helpers
# =========================================================
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



def _render_hero(context: MetaFeatureContext) -> None:
    strategy = _strategy_label(context.strategy_type)
    status_text = (
        "Метаознаки вже сформовані для поточної стратегії. Можна переглянути результат або перебудувати їх."
        if st.session_state.get("meta_ready") and st.session_state.get("meta_strategy_type") == context.strategy_type
        else "На цьому етапі будуються OOF-прогнози базових моделей першого рівня. Саме вони стають вхідними ознаками для ансамблів другого рівня."
    )
    st.markdown(
        f'''
        <div class="m4-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Формування метаознак</div>
            <div style="margin-bottom:0.6rem;">{status_text}</div>
            <span class="m4-pill">OOF predictions</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )



def _render_overview_cards(context: MetaFeatureContext) -> None:
    _render_section_header(
        "1. Поточний стан даних",
        "Короткий огляд активної конфігурації, яка буде використана для побудови OOF-метаознак.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Стратегія", _strategy_label(context.strategy_type))
    with c2:
        _render_html_card("Навчальна вибірка", f"{len(context.X_train)} × {pd.DataFrame(context.X_train).shape[1]}")
    with c3:
        _render_html_card("Тестова вибірка", f"{len(context.X_test)} × {pd.DataFrame(context.X_test).shape[1]}")
    with c4:
        _render_html_card("Цільова ознака", context.target_col or "—")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    with st.expander("Показати деталі конфігурації", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Q1", _metric(context.q1) if context.q1 is not None else "—")
        d2.metric("Q3", _metric(context.q3) if context.q3 is not None else "—")
        d3.metric("Ознак 1-го рівня", str(pd.DataFrame(context.X_train).shape[1]))

        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.info(
                "Для класифікаційної стратегії метаознаками стають імовірності класів, отримані з OOF-прогнозів базових моделей."
            )
        else:
            st.info(
                "Для регресійної стратегії метаознаками стають неперервні OOF-прогнози базових моделей; за потреби до них можуть бути додані вихідні ознаки."
            )

        st.markdown(
            f"**Базові моделі:** {', '.join(['Ridge', 'Random Forest', 'XGBoost'])}  \\\n**Призначення етапу:** побудова стабільних метаознак без leakage."
        )



def _render_theory_block() -> None:
    with st.expander("Коротке теоретичне пояснення", expanded=False):
        st.markdown(
            """
            **OOF (out-of-fold) метаознаки** — це прогнози базових моделей, сформовані так,
            щоб метамодель не бачила прогнозів, отриманих на тих самих даних, на яких модель навчалась.

            **Логіка побудови:**
            1. Навчальна вибірка ділиться на `k` folds.
            2. На кожному кроці базова модель навчається на `k-1` folds.
            3. Потім вона прогнозує на відкладеному fold.
            4. Сукупність таких прогнозів формує метаознаки для `train`.
            5. Для `test` прогноз усереднюється між усіма fold-моделями.

            У класифікації метаознаками є **ймовірності класів**, а в регресії —
            **неперервні прогнози** базових моделей.
            """
        )



def _render_controls() -> Tuple[MetaFeatureConfig, bool, bool]:
    _render_section_header(
        "2. Налаштування генерації",
        "Задайте параметри OOF-генерації та, за потреби, додайте вихідні ознаки першого рівня до метапростору.",
    )
    left, right = st.columns([1.15, 0.85])

    with left:
        with st.container(border=True):
            st.markdown("**Параметри генерації**")
            cv_folds = st.slider(
                "Кількість folds для OOF-прогнозів",
                min_value=3,
                max_value=10,
                value=int(st.session_state.get("meta_cv_folds", 5)),
                step=1,
                help="Більша кількість folds дає стабільніші OOF-ознаки, але збільшує час обчислень.",
            )

            passthrough = st.checkbox(
                "Додати початкові ознаки до метаознак (passthrough)",
                value=bool(st.session_state.get("meta_passthrough", False)),
                help="До OOF-метаознак буде додано вихідні ознаки першого рівня.",
            )

            st.session_state.meta_cv_folds = int(cv_folds)
            st.session_state.meta_passthrough = bool(passthrough)
            st.caption("Метод побудови: OOF-прогнози базових моделей із k-fold розбиттям.")

            b1, b2 = st.columns([1.5, 1])
            build_clicked = b1.button("Сформувати метаознаки", width="stretch", type="primary")
            clear_clicked = b2.button("Очистити", width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("**Коротко про результат**")
            st.write(
                "Після запуску буде сформовано `meta_X_train` для навчання метамоделі та `meta_X_test` для подальшого оцінювання ансамблів другого рівня."
            )
            st.write(
                "У класифікації метаознаками є ймовірності класів, а в регресії — неперервні прогнози базових моделей."
            )
            with st.expander("Поточна конфігурація", expanded=False):
                c1, c2 = st.columns(2)
                c1.metric("CV folds", str(cv_folds))
                c2.metric("Passthrough", _bool_label(bool(passthrough)))

    return MetaFeatureConfig(cv_folds=int(cv_folds), passthrough=bool(passthrough)), build_clicked, clear_clicked



def _build_summary_payload() -> Dict[str, Any]:
    meta_x_train = st.session_state.get("meta_X_train")
    meta_x_test = st.session_state.get("meta_X_test")
    feature_names = st.session_state.get("meta_feature_names", [])
    params = st.session_state.get("meta_generation_params", {})
    base_models = st.session_state.get("meta_base_models", [])

    return {
        "meta_train_shape": getattr(meta_x_train, "shape", None),
        "meta_test_shape": getattr(meta_x_test, "shape", None),
        "feature_count": len(feature_names),
        "base_model_count": len(base_models),
        "cv_folds": params.get("cv_folds"),
        "passthrough": params.get("passthrough", False),
    }



def _render_result_summary() -> None:
    if "meta_X_train" not in st.session_state or "meta_X_test" not in st.session_state:
        return

    summary = _build_summary_payload()
    train_shape = summary["meta_train_shape"] or (0, 0)
    test_shape = summary["meta_test_shape"] or (0, 0)

    _render_section_header(
        "3. Короткий результат",
        "Підсумок сформованого метапростору та готовності до навчання моделей другого рівня.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Метаознак", str(summary["feature_count"]), "Кількість сформованих метаознак")
    with c2:
        _render_html_card("Базових моделей", str(summary["base_model_count"]), "Кількість джерел OOF-прогнозів")
    with c3:
        _render_html_card("Навчальна вибірка", f"{train_shape[0]} × {train_shape[1]}", "Розмір матриці для метамоделі")
    with c4:
        _render_html_card("Тестова вибірка", f"{test_shape[0]} × {test_shape[1]}", "Розмір тестової матриці метаознак")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    
    summary_df = pd.DataFrame(
        [
            {"Показник": "cv_folds", "Значення": summary["cv_folds"]},
            {"Показник": "passthrough", "Значення": _bool_label(bool(summary["passthrough"]))},
            {"Показник": "Готовність", "Значення": "Метаознаки сформовані"},
        ]
    )
    st.dataframe(summary_df.astype(str), width="stretch", hide_index=True)


def _render_preview_tabs() -> None:
    if "meta_X_train" not in st.session_state or "meta_X_test" not in st.session_state:
        return

    meta_x_train = st.session_state.meta_X_train
    meta_x_test = st.session_state.meta_X_test
    feature_names = st.session_state.get("meta_feature_names", [])
    base_models = st.session_state.get("meta_base_models", [])
    params = st.session_state.get("meta_generation_params", {})

    _render_section_header(
        "4. Перегляд результатів",
        "Доступний попередній перегляд матриць метаознак, їх структури та службової інформації про генерацію.",
    )
    tab1, tab2, tab3 = st.tabs(["Preview train/test", "Структура метаознак", "Службова інформація"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown(f"**meta_X_train:** `{meta_x_train.shape}`")
            st.dataframe(meta_x_train.head(10), width="stretch")
        with right:
            st.markdown(f"**meta_X_test:** `{meta_x_test.shape}`")
            st.dataframe(meta_x_test.head(10), width="stretch")

    with tab2:
        preview_df = pd.DataFrame(
            {
                "Метаознака": feature_names,
                "Тип блоку": [
                    "Probability / прогноз" if str(name).startswith("meta_") else "Passthrough"
                    for name in feature_names
                ],
            }
        )
        st.dataframe(
            preview_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Метаознака": st.column_config.TextColumn("Метаознака", width="large"),
                "Тип блоку": st.column_config.TextColumn("Тип блоку", width="medium"),
            },
        )

    with tab3:
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            with st.container(border=True):
                st.markdown("**Базові моделі**")
                st.write(", ".join(_short_model_name(name) for name in base_models) if base_models else "—")
        with info_col2:
            with st.container(border=True):
                st.markdown("**Параметри генерації**")
                st.json(params)



def _render_export_section(strategy_type: str) -> None:
    _render_section_header(
        "5. Експорт результатів",
        "Збереження опису метаознак і конфігурації генерації для наступних етапів дослідження.",
    )
    json_bytes = _results_to_json_bytes(
        {
            "strategy_type": strategy_type,
            "meta_generation_params": st.session_state.get("meta_generation_params", {}),
            "meta_base_models": st.session_state.get("meta_base_models", []),
            "meta_feature_names": st.session_state.get("meta_feature_names", []),
            "meta_X_train_shape": list(st.session_state.meta_X_train.shape),
            "meta_X_test_shape": list(st.session_state.meta_X_test.shape),
        }
    )

    left, right = st.columns([1.2, 0.8])
    with left:
        st.download_button(
            on_click="ignore",
            label="Завантажити опис метаознак (JSON)",
            data=json_bytes,
            file_name="metafeatures_description.json",
            mime="application/json",
            width="stretch",
        )
    with right:
        st.info("JSON містить параметри генерації, список базових моделей, назви метаознак і розміри матриць метаознак.")


# =========================================================
# Main page
# =========================================================
def page_formation_metafeatures() -> None:
    _inject_page_style()
    if not _check_ready_for_metafeatures():
        return

    context = _load_context()
    _render_hero(context)
    _render_overview_cards(context)
    _render_theory_block()

    config, build_clicked, clear_clicked = _render_controls()

    if clear_clicked:
        _clear_metafeatures_state()
        st.success("Метаознаки очищено.")
        return

    if build_clicked:
        with st.spinner("Триває формування OOF-метаознак..."):
            artifacts = _build_metafeatures(context=context, config=config)
            _save_metafeatures(artifacts, strategy_type=context.strategy_type)
        st.success("Метаознаки успішно сформовано!")

    if st.session_state.get("meta_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Сформуйте метаознаки повторно.")
        return

    if not st.session_state.get("meta_ready", False):
        return

    _render_result_summary()
    _render_preview_tabs()
    _render_export_section(context.strategy_type)
