#module_2_visual_refined_v2.py

import datetime
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from data_protocol import split_indices, frame_digest, reset_downstream

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# =========================================================
# Constants
# =========================================================
STRATEGY_LABELS = {
    "direct_multiclass_classification": "Пряма багатокласова класифікація",
    "regression_with_categorization": "Регресія з подальшою категоризацією",
}

DEFAULT_PREPROCESSING_CONFIG = {
    "scaler_type": "StandardScaler",
    "numeric_imputer_strategy": "mean",
    "categorical_imputer_strategy": "most_frequent",
    "onehot_drop": "if_binary",
    "test_size": 0.30,
    "random_state": 42,
}

NUMERIC_IMPUTER_LABELS = {
    "Середнє значення": "mean",
    "Медіана": "median",
}

CATEGORICAL_IMPUTER_LABELS = {
    "Найчастіше значення": "most_frequent",
    "Окреме службове значення": "constant",
}

ONEHOT_ENCODING_LABELS = {
    "Повне кодування": "none",
    "Скорочене кодування для бінарних ознак": "if_binary",
}

SPLIT_STATE_KEYS = [
    # module_2
    "strategy_type",
    "target_col",
    "feature_cols",
    "feature_names_after_preprocessing",
    "preprocessing_config",
    "preprocessor",
    "X_raw",
    "X_train_raw",
    "X_test_raw",
    "X",
    "y",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "y_cont",
    "y_train_cont",
    "y_test_cont",
    "y_train_cls",
    "y_test_cls",
    "Q1",
    "Q3",
    "test_size",
    "random_state",
    "stratify_used",
    "split_ready",
    # module_4
    "first_level_models",
    "first_level_results",
    "first_level_predictions",
    "first_level_predictions_cls",
    "first_level_params",
    "first_level_strategy_type",
    "optimization_results",
    "optimization_mode",
    # module_5
    "meta_X_train",
    "meta_X_test",
    "meta_feature_names",
    "meta_generation_params",
    "meta_base_models",
    "meta_full_train_models",
    "meta_ready",
    "meta_strategy_type",
    # module_6
    "second_level_models",
    "second_level_model",
    "second_level_results",
    "second_level_predictions",
    "second_level_predictions_cls",
    "second_level_params",
    "second_level_strategy_type",
    # module_7
    "strategy_snapshots",
]


# =========================================================
# Data classes
# =========================================================
@dataclass
class SplitConfig:
    target_col: str
    strategy_type: str
    scaler_type: str
    numeric_imputer_strategy: str
    categorical_imputer_strategy: str
    onehot_drop: str
    test_size: float
    random_state: int


@dataclass
class SplitArtifacts:
    strategy_type: str
    target_col: str
    feature_cols: list[str]
    feature_names_after_preprocessing: list[str]
    preprocessing_config: dict[str, Any]
    preprocessor: ColumnTransformer
    X_raw: pd.DataFrame
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_active: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    y_cont: pd.Series
    y_train_cont: pd.Series
    y_test_cont: pd.Series
    y_train_cls: pd.Series
    y_test_cls: pd.Series
    q1: float
    q3: float
    test_size: float
    random_state: int
    stratify_used: bool


# =========================================================
# Column helpers
# =========================================================
def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


# =========================================================
# State helpers
# =========================================================
def _invalidate_split_state() -> None:
    for key in SPLIT_STATE_KEYS:
        st.session_state.pop(key, None)


def _save_split_to_session(artifacts: SplitArtifacts) -> None:
    signature=frame_digest(artifacts.X_raw.assign(__target__=artifacts.y_cont))+str((artifacts.test_size,artifacts.random_state,artifacts.preprocessing_config))
    previous=st.session_state.get('split_signature')
    if previous!=signature:
        st.session_state.strategy_snapshots={}
        st.session_state.evaluation_runs={}
        st.session_state.figure_exports={}
    reset_downstream(4)
    st.session_state.split_signature=signature
    st.session_state.strategy_type = artifacts.strategy_type
    st.session_state.target_col = artifacts.target_col
    st.session_state.feature_cols = artifacts.feature_cols
    st.session_state.feature_names_after_preprocessing = artifacts.feature_names_after_preprocessing
    st.session_state.preprocessing_config = artifacts.preprocessing_config
    st.session_state.preprocessor = artifacts.preprocessor

    st.session_state.X_raw = artifacts.X_raw.copy()
    st.session_state.X_train_raw = artifacts.X_train_raw.copy()
    st.session_state.X_test_raw = artifacts.X_test_raw.copy()

    st.session_state.X_train = artifacts.X_train.copy()
    st.session_state.X_test = artifacts.X_test.copy()
    st.session_state.X = pd.concat([artifacts.X_train, artifacts.X_test]).sort_index().copy()

    st.session_state.y = artifacts.y_active.copy()
    st.session_state.y_train = artifacts.y_train.copy()
    st.session_state.y_test = artifacts.y_test.copy()

    st.session_state.y_cont = artifacts.y_cont.copy()
    st.session_state.y_train_cont = artifacts.y_train_cont.copy()
    st.session_state.y_test_cont = artifacts.y_test_cont.copy()

    st.session_state.y_train_cls = artifacts.y_train_cls.copy()
    st.session_state.y_test_cls = artifacts.y_test_cls.copy()

    st.session_state.Q1 = float(artifacts.q1)
    st.session_state.Q3 = float(artifacts.q3)
    st.session_state.test_size = float(artifacts.test_size)
    st.session_state.random_state = int(artifacts.random_state)
    st.session_state.stratify_used = bool(artifacts.stratify_used)
    st.session_state.visual_before=artifacts.X_train_raw.assign(**{artifacts.target_col:artifacts.y_train_cont})
    st.session_state.visual_after=artifacts.X_train.assign(**{artifacts.target_col:artifacts.y_train_cont})
    st.session_state.split_ready = True


# =========================================================
# ML helpers
# =========================================================
def discretize_g3(y, q1: float, q3: float) -> np.ndarray:
    y = np.asarray(y)
    classes = np.zeros_like(y, dtype=int)
    classes[y <= q1] = 0
    classes[(y > q1) & (y <= q3)] = 1
    classes[y > q3] = 2
    return classes


def _make_stratify_bins(y: pd.Series, n_bins: int = 3) -> pd.Series | None:
    y = pd.Series(y).reset_index(drop=True)

    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None

    bins = pd.Series(bins)
    if bins.nunique(dropna=True) < 2:
        return None

    min_count = bins.value_counts().min()
    if pd.isna(min_count) or min_count < 2:
        return None

    return bins


def _build_preprocessor(
    X_train_raw: pd.DataFrame,
    *,
    scaler_type: str,
    numeric_imputer_strategy: str,
    categorical_imputer_strategy: str,
    onehot_drop: str,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    binary_cols = [c for c in X_train_raw if c == "subject_math"]
    numeric_cols = [c for c in _numeric_columns(X_train_raw) if c not in binary_cols]
    categorical_cols = _categorical_columns(X_train_raw)

    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]
    if scaler != "passthrough":
        numeric_steps.append(("scaler", scaler))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None if onehot_drop == "none" else "if_binary",
                ),
            ),
        ]
    )

    transformers = []
    if binary_cols:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), binary_cols))
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("У X немає доступних ознак для preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def _transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)
    return X_train, X_test, feature_names


def _prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def _validate_split_inputs(X_raw: pd.DataFrame, y_cont: pd.Series) -> list[str]:
    errors: list[str] = []

    if X_raw.empty:
        errors.append("Матриця ознак X порожня. Потрібен хоча б один предиктор.")
    if y_cont.isna().any():
        errors.append("Цільова змінна містить пропущені значення.")
    if not pd.api.types.is_numeric_dtype(y_cont):
        errors.append(
            "Для двох стратегій цільова ознака має бути числовою "
            "(щоб можна було обчислити квартилі Q1 і Q3)."
        )

    return errors


def build_split_artifacts(df: pd.DataFrame, config: SplitConfig) -> SplitArtifacts:
    X_raw, y_cont = _prepare_xy(df, config.target_col)
    errors = _validate_split_inputs(X_raw, y_cont)
    if errors:
        raise ValueError("\n".join(errors))

    # One strategy-independent group split, deterministic for the same data and seed.
    train_idx,test_idx=split_indices(X_raw,float(config.test_size),int(config.random_state))
    X_train_raw=X_raw.iloc[train_idx].copy();X_test_raw=X_raw.iloc[test_idx].copy()
    y_train_cont=y_cont.iloc[train_idx].copy();y_test_cont=y_cont.iloc[test_idx].copy()
    stratify_used=False

    q1 = float(np.percentile(y_train_cont, 25))
    q3 = float(np.percentile(y_train_cont, 75))

    y_train_cls = pd.Series(discretize_g3(y_train_cont, q1, q3), index=X_train_raw.index)
    y_test_cls = pd.Series(discretize_g3(y_test_cont, q1, q3), index=X_test_raw.index)

    preprocessor, num_cols, cat_cols = _build_preprocessor(
        X_train_raw,
        scaler_type=config.scaler_type,
        numeric_imputer_strategy=config.numeric_imputer_strategy,
        categorical_imputer_strategy=config.categorical_imputer_strategy,
        onehot_drop=config.onehot_drop,
    )
    X_train, X_test, feature_names = _transform_with_preprocessor(
        preprocessor,
        X_train_raw,
        X_test_raw,
    )

    if config.strategy_type == "direct_multiclass_classification":
        y_train = y_train_cls.copy()
        y_test = y_test_cls.copy()
        y_active = pd.concat([y_train, y_test]).sort_index()
    else:
        y_train = pd.Series(y_train_cont, index=X_train.index)
        y_test = pd.Series(y_test_cont, index=X_test.index)
        y_active = pd.Series(y_cont).copy()

    preprocessing_config = {
        "scaler_type": config.scaler_type,
        "numeric_imputer_strategy": config.numeric_imputer_strategy,
        "categorical_imputer_strategy": config.categorical_imputer_strategy,
        "onehot_drop": config.onehot_drop,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "n_output_features": len(feature_names),
    }

    return SplitArtifacts(
        strategy_type=config.strategy_type,
        target_col=config.target_col,
        feature_cols=[col for col in df.columns if col != config.target_col],
        feature_names_after_preprocessing=feature_names,
        preprocessing_config=preprocessing_config,
        preprocessor=preprocessor,
        X_raw=X_raw,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_test=X_test,
        y_active=y_active,
        y_train=y_train,
        y_test=y_test,
        y_cont=pd.Series(y_cont).copy(),
        y_train_cont=pd.Series(y_train_cont, index=X_train.index),
        y_test_cont=pd.Series(y_test_cont, index=X_test.index),
        y_train_cls=y_train_cls,
        y_test_cls=y_test_cls,
        q1=q1,
        q3=q3,
        test_size=float(config.test_size),
        random_state=int(config.random_state),
        stratify_used=stratify_used,
    )


def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .m2-state-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            background: rgba(255,255,255,0.02);
            min-height: 112px;
        }
        .m2-state-kicker {
            font-size: 0.82rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.35rem;
        }
        .m2-state-value {
            font-size: 1.16rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.28rem;
        }
        .m2-state-note {
            font-size: 0.86rem;
            color: rgba(120,120,120,0.92);
        }
        .m2-section-title {
            margin-top: 0.45rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m2-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }
        .m2-hero {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(70,130,180,0.08), rgba(255,255,255,0.02));
            margin-bottom: 0.8rem;
        }
        .m2-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .m2-panel {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.75rem;
        }
        .m2-panel-title {
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .m2-panel-note {
            font-size: 0.9rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.7rem;
        }
        .m2-mini-note {
            font-size: 0.88rem;
            color: rgba(120,120,120,0.95);
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="m2-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="m2-subtle">{subtitle}</div>', unsafe_allow_html=True)


def _render_html_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="m2-state-card">
            <div class="m2-state-kicker">{title}</div>
            <div class="m2-state-value">{value}</div>
            <div class="m2-state-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# UI helpers
# =========================================================
def _render_header() -> None:
    st.markdown(
        '''
        <div class="m2-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Попередня обробка даних та формування вибірок</div>
            <div style="margin-bottom:0.6rem;">На цьому етапі обирається <b>target</b>, налаштовується preprocessing-pipeline і формується коректний <b>train/test split</b> для подальшого моделювання.</div>
            <span class="m2-pill">Очищення даних</span>
            <span class="m2-pill">Трансформація даних</span>
            <span class="m2-pill">Розбиття набору даних на навчальну та тестову вибірки</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_dataset_overview(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    rows_with_nan = int(df.isna().any(axis=1).sum())
    dup_count = int(df.duplicated().sum())
    numeric_count = len(_numeric_columns(df))
    categorical_count = len(_categorical_columns(df))

    _render_section_header(
        "1. Поточний набір даних",
        "Швидкий огляд структури датасету перед очищенням, трансформацією та розбиттям на вибірки.",
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Рядки", rows)
    c2.metric("Стовпці", cols)
    c3.metric("Числові", numeric_count)
    c4.metric("Категоріальні", categorical_count)
    c5.metric("Рядки з NaN", rows_with_nan)
    c6.metric("Дублікати", dup_count)

    if rows_with_nan > 0 or dup_count > 0:
        st.warning(
            "У наборі є пропуски або однакові рядки. Пропуски заповнюються лише за train. Однакові скорочені профілі можуть належати різним учням; їх видалення не є обов’язковим."
        )
    else:
        st.success("Базова перевірка пройдена: явних пропусків у рядках і дублікатів не знайдено.")


def _render_cleaning_panel() -> None:
    _render_section_header(
        "2. Очищення набору даних",
        "Видалення дублікатів, відновлення набору даних та параметри обробки пропущених значень.",
    )
    
    st.markdown('<div class="m2-panel">', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-title">Видалення дублікатів</div>', unsafe_allow_html=True)
    
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Видалити дублікати", width="stretch"):
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates()
            _invalidate_split_state()
            st.success(f"Видалено дублікатів: {before - len(st.session_state.df)}")
    with a2:
        if st.button("Відновити", width="stretch"):
            st.session_state.df = st.session_state.df_orig.copy()
            st.session_state.df_before = st.session_state.df_orig.copy()
            _invalidate_split_state()
            st.success("Оригінальний набір даних відновлено.")
    st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('<div class="m2-panel">', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-title">Обробка пропущених значень</div>', unsafe_allow_html=True)
    
    df_current = st.session_state.df
    has_missing_rows = bool(df_current.isna().any(axis=1).sum())

    default_num_imputer = st.session_state.get("preprocessing_config", {}).get(
            "numeric_imputer_strategy", DEFAULT_PREPROCESSING_CONFIG["numeric_imputer_strategy"]
        )
    numeric_imputer_reverse = {v: k for k, v in NUMERIC_IMPUTER_LABELS.items()}
    default_num_label = numeric_imputer_reverse.get(default_num_imputer, "Середнє значення")

    default_cat_imputer = st.session_state.get("preprocessing_config", {}).get(
            "categorical_imputer_strategy", DEFAULT_PREPROCESSING_CONFIG["categorical_imputer_strategy"]
        )
    categorical_imputer_reverse = {v: k for k, v in CATEGORICAL_IMPUTER_LABELS.items()}
    default_cat_label = categorical_imputer_reverse.get(default_cat_imputer, "Найчастіше значення")

    if has_missing_rows:
        st.selectbox(
                "Заповнення пропусків у числових ознаках",
                options=list(NUMERIC_IMPUTER_LABELS.keys()),
                index=list(NUMERIC_IMPUTER_LABELS.keys()).index(default_num_label),
                key="numeric_imputer_label",
            )

        st.selectbox(
                "Заповнення пропусків у категоріальних ознаках",
                options=list(CATEGORICAL_IMPUTER_LABELS.keys()),
                index=list(CATEGORICAL_IMPUTER_LABELS.keys()).index(default_cat_label),
                key="categorical_imputer_label",
            )

        st.markdown('<div class="m2-mini-note">Після зміни параметрів імпутації повторно сформуйте train/test split, щоб оновити preprocessing-пайплайн.</div>', unsafe_allow_html=True)
    else:
        st.info("У поточному наборі даних пропущених значень немає, тому обробка пропущених значень на цьому етапі не потрібна.")
        st.session_state["numeric_imputer_label"] = default_num_label
        st.session_state["categorical_imputer_label"] = default_cat_label

        st.selectbox(
                "Заповнення пропусків у числових ознаках",
                options=["Не застосовується"],
                index=0,
                disabled=True,
                key="numeric_imputer_label_disabled",
            )
        st.selectbox(
                "Заповнення пропусків у категоріальних ознаках",
                options=["Не застосовується"],
                index=0,
                disabled=True,
                key="categorical_imputer_label_disabled",
            )
        
        st.markdown('</div>', unsafe_allow_html=True)


def _render_dataset_preview(df: pd.DataFrame) -> None:
    _render_section_header(
        "3. Попередній перегляд даних",
        "Перевірка вмісту таблиці, типів стовпців і можливість експорту поточного датасету.",
    )

    preview_tab1, preview_tab2, preview_tab3 = st.tabs([
        "Перші рядки",
        "Типи стовпців",
        "Експорт CSV",
    ])

    with preview_tab1:
        st.dataframe(df.head(20), width="stretch")

    with preview_tab2:
        dtype_df = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "unique": df.nunique(dropna=False).values,
        })
        st.dataframe(dtype_df, width="stretch", hide_index=True)

    with preview_tab3:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"current_dataset_{ts}.csv"
        st.download_button(
            on_click="ignore",
            label="Завантажити поточний CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            width="stretch",
        )


def _render_strategy_form(df_current: pd.DataFrame) -> SplitConfig | None:
    _render_section_header(
        "4. Налаштування стратегії дослідження та трансформація даних",
        "Вибір цільової ознаки, сценарію дослідження, параметрів розбиття набору даних та налаштування трансформації ознак, які будуть застосовані після формування вибірок.",
    )

    all_cols = df_current.columns.tolist()
    if len(all_cols) < 2:
        st.warning("У наборі даних замало стовпців для формування X і y.")
        return None

    default_target = (
        st.session_state.target_col
        if "target_col" in st.session_state and st.session_state.target_col in all_cols
        else all_cols[-1]
    )

    

    
    st.markdown('<div class="m2-panel">', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-title">Конфігурація експерименту</div>', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-note">Визначте цільову ознаку, активну стратегію моделювання та параметри майбутнього розбиття на вибірки.</div>', unsafe_allow_html=True)

    target_col = st.selectbox(
            "Цільова ознака (target)",
            options=all_cols,
            index=all_cols.index(default_target),
        )
        
    strategy_type = st.radio(
            "Стратегія дослідження",
            options=list(STRATEGY_LABELS.keys()),
            index=0 if st.session_state.get("strategy_type", "direct_multiclass_classification") == "direct_multiclass_classification" else 1,
            format_func=lambda x: STRATEGY_LABELS[x],
        )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        test_size = st.slider(
                "Частка тестової вибірки",
            min_value=0.10,
            max_value=0.40,
            value=float(st.session_state.get("test_size", DEFAULT_PREPROCESSING_CONFIG["test_size"])),
            step=0.05,
            )
    with col_t2:
        random_state = st.number_input(
                "Початкове значення random state",
            min_value=0,
            max_value=9999,
            value=int(st.session_state.get("random_state", DEFAULT_PREPROCESSING_CONFIG["random_state"])),
            step=1,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="m2-panel">', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-title">Трансформація даних</div>', unsafe_allow_html=True)
    st.markdown('<div class="m2-panel-note">Масштабування та кодування ознак.</div>', unsafe_allow_html=True)

    scaler_type = st.radio(
            "Масштабування числових ознак",
        options=["None", "StandardScaler", "MinMaxScaler"],
        index=["None", "StandardScaler", "MinMaxScaler"].index(
            st.session_state.get("preprocessing_config", {}).get(
                    "scaler_type", DEFAULT_PREPROCESSING_CONFIG["scaler_type"]
                )
            ),
            horizontal=True,
        )
    st.info("Масштабування впливає лише на числові ознаки й виконується після обробки пропущених значень.")

    numeric_imputer_label = st.session_state.get("numeric_imputer_label", "Середнє значення")
    categorical_imputer_label = st.session_state.get("categorical_imputer_label", "Найчастіше значення")
    numeric_imputer_strategy = NUMERIC_IMPUTER_LABELS[numeric_imputer_label]
    categorical_imputer_strategy = CATEGORICAL_IMPUTER_LABELS[categorical_imputer_label]

    default_onehot = st.session_state.get("preprocessing_config", {}).get(
            "onehot_drop", DEFAULT_PREPROCESSING_CONFIG["onehot_drop"]
        )
    onehot_reverse = {v: k for k, v in ONEHOT_ENCODING_LABELS.items()}
    onehot_label = st.selectbox(
            "Кодування категоріальних ознак",
        options=list(ONEHOT_ENCODING_LABELS.keys()),
        index=list(ONEHOT_ENCODING_LABELS.keys()).index(
            onehot_reverse.get(default_onehot, "Скорочене кодування для бінарних ознак")
            ),
        )
    onehot_drop = ONEHOT_ENCODING_LABELS[onehot_label]
    st.markdown('</div>', unsafe_allow_html=True)

    feature_cols = [col for col in all_cols if col != target_col]
    numeric_preview = [col for col in feature_cols if col in _numeric_columns(df_current)]
    categorical_preview = [col for col in feature_cols if col in _categorical_columns(df_current)]

    with st.expander("Показати деталі попередньої обробки", expanded=False):
        st.write(
                "Pipeline fit виконується лише на train. Для числових ознак застосовуються нормалізація (Min–Max Scaler) або стандартизація, а для категоріальних — кодування (One-Hot Encoding)."
            )
        st.write(
                "Для обох стратегій межі Q1 і Q3 обчислюються тільки на навчальній вибірці, тому тестові дані не впливають на категоризацію."
            )
        details_df = pd.DataFrame(
                {
                    "Параметр": [
                        "Масштабування",
                        "Обробка пропущених числових ознак",
                        "Обробка пропущених категоріальних ознак",
                        "One-Hot Encoding",
                        "Test size",
                        "Random state",
                    ],
                    "Поточне значення": [
                        scaler_type,
                        numeric_imputer_label,
                        categorical_imputer_label,
                        onehot_label,
                        f"{test_size:.2f}",
                        str(int(random_state)),
                    ],
                }
            )
        st.dataframe(details_df, width="stretch", hide_index=True)

    with st.expander("Список необроблених ознак і їх типи", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write("**Усі необроблені ознаки**")
            st.write(feature_cols if feature_cols else "—")
        with col_f2:
            st.write("**Розподіл за типами**")
            st.write({
                "numeric": numeric_preview,
                "categorical": categorical_preview,
            })

    return SplitConfig(
        target_col=target_col,
        strategy_type=strategy_type,
        scaler_type=scaler_type,
        numeric_imputer_strategy=numeric_imputer_strategy,
        categorical_imputer_strategy=categorical_imputer_strategy,
        onehot_drop=onehot_drop,
        test_size=float(test_size),
        random_state=int(random_state),
    )

def _render_split_actions(df_current: pd.DataFrame, config: SplitConfig | None) -> None:
    if config is None:
        return

    _render_section_header(
        "5. Формування навчальної та тестової вибірки",
        "Після підтвердження конфігурації буде побудовано етап попередньої обробки даних і створено вибірки для навчання та оцінювання.",
    )
    c1, c2 = st.columns([1.3, 0.8])

    with c1:
        if st.button("Підготувати ознаки й розподілити дані на вибірки", type="primary", width="stretch"):
            try:
                artifacts = build_split_artifacts(df_current, config)
                _save_split_to_session(artifacts)
                st.success(
                    f"Дані підготовлено для стратегії: {STRATEGY_LABELS[config.strategy_type]}."
                )
            except Exception as exc:
                st.error(f"Помилка під час формування вибірок і preprocessing: {exc}")

    with c2:
        if st.button("Скинути розбиття на вибірки", width="stretch"):
            _invalidate_split_state()
            st.success("Train/test split, preprocessing pipeline і похідні стани очищено.")


def _render_split_summary() -> None:
    if not st.session_state.get("split_ready", False):
        return

    _render_section_header(
        "6. Поточний стан даних",
        "Підсумок сформованої конфігурації, яка буде використана на наступних етапах моделювання.",
    )

    strategy_label = STRATEGY_LABELS.get(st.session_state.get("strategy_type", "—"), "—")
    n_features = len(st.session_state.get("feature_names_after_preprocessing", []))
    train_shape = f"{st.session_state.X_train.shape[0]} × {st.session_state.X_train.shape[1]}"
    test_shape = f"{st.session_state.X_test.shape[0]} × {st.session_state.X_test.shape[1]}"
    target_name = st.session_state.get("target_col", "—")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Стратегія", strategy_label)
    with c2:
        _render_html_card("Навчальна вибірка", train_shape)
    with c3:
        _render_html_card("Тестова вибірка", test_shape)
    with c4:
        _render_html_card("Цільова ознака", target_name)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    with st.expander("Показати деталі попередньої обробки", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Q1", f"{st.session_state.get('Q1', 0.0):.4f}")
        p2.metric("Q3", f"{st.session_state.get('Q3', 0.0):.4f}")
        p3.metric("Ознак після попередньої обробки", str(n_features))
        p4.metric("Stratify", "так" if st.session_state.get("stratify_used", False) else "ні")

        config = st.session_state.get("preprocessing_config", {})
        if config:
            st.info(
                "Preprocessing-пайплайн fit виконується лише на train-вибірці. Це зменшує ризик data leakage перед навчанням моделей першого рівня."
            )
            st.json(config)

    summary_tab1, summary_tab2, summary_tab3, summary_tab4 = st.tabs([
        "Порівняння raw vs processed",
        "Target preview",
        "Класи після дискретизації",
        "Назви ознак після preprocessing",
    ])

    with summary_tab1:
        left, right = st.columns(2)
        with left:
            st.markdown("**X_train_raw (head)**")
            st.dataframe(st.session_state.X_train_raw.head(), width="stretch")
        with right:
            st.markdown("**X_train після pipeline (head)**")
            st.dataframe(st.session_state.X_train.head(), width="stretch")

    with summary_tab2:
        y_preview = pd.DataFrame({
            "y_train": st.session_state.y_train.head(10),
            "y_train_cont": st.session_state.y_train_cont.head(10),
        })
        st.dataframe(y_preview, width="stretch")

    with summary_tab3:
        cls_preview = pd.DataFrame({
            "y_train_cls": st.session_state.y_train_cls.head(10),
            "y_test_cls": st.session_state.y_test_cls.head(10),
        })
        st.dataframe(cls_preview, width="stretch")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Розподіл класів у train**")
            st.dataframe(
                st.session_state.y_train_cls.value_counts().sort_index().rename_axis("class").reset_index(name="count"),
                width="stretch",
                hide_index=True,
            )
        with dist_col2:
            st.markdown("**Розподіл класів у test**")
            st.dataframe(
                st.session_state.y_test_cls.value_counts().sort_index().rename_axis("class").reset_index(name="count"),
                width="stretch",
                hide_index=True,
            )

    with summary_tab4:
        feature_df = pd.DataFrame({
            "feature_name": st.session_state.get("feature_names_after_preprocessing", []),
        })
        st.dataframe(feature_df, width="stretch", hide_index=True)

    st.success("Попередню обробку даних налаштовано, можна переходити до навчання моделей першого рівня.")


# =========================================================
# Main page
# =========================================================
def page_preprocessing() -> None:
    _inject_page_style()

    if "df" not in st.session_state:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return

    _render_header()

    if "df_before" not in st.session_state:
        st.session_state.df_before = st.session_state.df.copy()

    df_current = st.session_state.df.copy()

    _render_dataset_overview(df_current)
    _render_cleaning_panel()
    _render_dataset_preview(df_current)
    config = _render_strategy_form(df_current)
    _render_split_actions(df_current, config)
    _render_split_summary()
