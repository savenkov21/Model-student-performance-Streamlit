from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"


@dataclass
class EvaluationContext:
    strategy_type: str
    target_col: str
    q1: Any
    q3: Any
    meta_ready: bool
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_results: Dict[str, Dict[str, Any]]
    first_level_params: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    second_level_params: Dict[str, Any]


@dataclass
class EvaluationArtifacts:
    comparison_df: pd.DataFrame
    best_model: Dict[str, Any]
    final_report: Dict[str, Any]


def _results_to_json_bytes(results_dict):
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(results_dict, indent=4, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return json_bytes


def _strategy_label(strategy_type):
    mapping = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return mapping.get(strategy_type, strategy_type)


def _strategy_short_label(strategy_type):
    return "Класифікація" if strategy_type == CLASSIFICATION_STRATEGY else "Регресія + категоризація"


def _get_primary_metric(strategy_type):
    return "F1 weighted" if strategy_type == CLASSIFICATION_STRATEGY else "F1 weighted after discretization"


def _check_ready_for_evaluation():
    required_keys = [
        "strategy_type",
        "first_level_results",
        "first_level_strategy_type",
        "second_level_results",
        "second_level_strategy_type",
    ]
    missing = [key for key in required_keys if key not in st.session_state]

    if missing:
        st.warning(
            "Спочатку виконайте попередні етапи pipeline: підготовку даних, "
            "навчання моделей першого рівня та ансамблів другого рівня."
        )
        return False

    strategy_type = st.session_state.get("strategy_type")

    if st.session_state.get("first_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть базові моделі на попередньому етапі.")
        return False

    if st.session_state.get("second_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть ансамблі другого рівня.")
        return False

    return True


def _load_context() -> Optional[EvaluationContext]:
    if not _check_ready_for_evaluation():
        return None

    return EvaluationContext(
        strategy_type=st.session_state.get("strategy_type"),
        target_col=st.session_state.get("target_col", "—"),
        q1=st.session_state.get("Q1", "—"),
        q3=st.session_state.get("Q3", "—"),
        meta_ready=bool(st.session_state.get("meta_ready", False)),
        first_level_results=st.session_state.get("first_level_results", {}),
        second_level_results=st.session_state.get("second_level_results", {}),
        first_level_params=st.session_state.get("first_level_params", {}),
        meta_generation_params=st.session_state.get("meta_generation_params", {}),
        second_level_params=st.session_state.get("second_level_params", {}),
    )


def _classification_row(model_name, result, level, family):
    return {
        "Strategy": "Direct multiclass classification",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "Accuracy": float(result["Accuracy"]),
        "Balanced Accuracy": float(result["Balanced_accuracy"]),
        "F1 weighted": float(result["F1_weighted"]),
    }


def _regression_row(model_name, result, level, family):
    return {
        "Strategy": "Regression with categorization",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "MAE": float(result["MAE"]),
        "MSE": float(result["MSE"]),
        "RMSE": float(result["RMSE"]),
        "R2": float(result["R2"]),
        "Accuracy after discretization": float(result["Accuracy_after_discretization"]),
        "Balanced accuracy after discretization": float(result["Balanced_accuracy_after_discretization"]),
        "F1 weighted after discretization": float(result["F1_weighted_after_discretization"]),
    }


def _build_comparison_dataframe(strategy_type, first_level_results, second_level_results):
    rows: List[Dict[str, Any]] = []

    if strategy_type == CLASSIFICATION_STRATEGY:
        for model_name, result in first_level_results.items():
            rows.append(_classification_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(_classification_row(model_name, result, "Другий рівень", "Ансамбль"))
    else:
        for model_name, result in first_level_results.items():
            rows.append(_regression_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(_regression_row(model_name, result, "Другий рівень", "Ансамбль"))

    return pd.DataFrame(rows)


def _select_best_model(strategy_type, comparison_df):
    metric = _get_primary_metric(strategy_type)
    best_idx = comparison_df[metric].idxmax()
    best_row = comparison_df.loc[best_idx]
    return {
        "name": best_row["Model"],
        "level": best_row["Level"],
        "family": best_row["Family"],
        "metric": metric,
        "value": float(best_row[metric]),
    }


def _save_strategy_snapshot(context: EvaluationContext, comparison_df, best_model):
    if "strategy_snapshots" not in st.session_state:
        st.session_state.strategy_snapshots = {}

    st.session_state.strategy_snapshots[context.strategy_type] = {
        "strategy_label": _strategy_label(context.strategy_type),
        "target_col": context.target_col,
        "Q1": context.q1,
        "Q3": context.q3,
        "comparison_df": comparison_df.to_dict(orient="records"),
        "best_model": best_model,
        "first_level_results": context.first_level_results,
        "second_level_results": context.second_level_results,
        "first_level_params": context.first_level_params,
        "meta_generation_params": context.meta_generation_params,
        "second_level_params": context.second_level_params,
    }


def _build_final_report(context: EvaluationContext, comparison_df, best_model):
    return {
        "strategy_type": context.strategy_type,
        "strategy_label": _strategy_label(context.strategy_type),
        "target_col": context.target_col,
        "Q1": context.q1,
        "Q3": context.q3,
        "best_model": best_model,
        "first_level_params": context.first_level_params,
        "meta_generation_params": context.meta_generation_params,
        "second_level_params": context.second_level_params,
        "first_level_results": context.first_level_results,
        "second_level_results": context.second_level_results,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "strategy_snapshots": st.session_state.get("strategy_snapshots", {}),
    }


def _build_artifacts(context: EvaluationContext) -> EvaluationArtifacts:
    comparison_df = _build_comparison_dataframe(
        strategy_type=context.strategy_type,
        first_level_results=context.first_level_results,
        second_level_results=context.second_level_results,
    )
    best_model = _select_best_model(context.strategy_type, comparison_df)
    _save_strategy_snapshot(context, comparison_df, best_model)
    from statistical_analysis import capture_run
    capture_run(comparison_df)
    final_report = _build_final_report(context, comparison_df, best_model)
    return EvaluationArtifacts(comparison_df=comparison_df, best_model=best_model, final_report=final_report)


def _metric_value(value):
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


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
            margin-top: 0.55rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m4-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .m4-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
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
            word-break: break-word;
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
        .m4-note {
            border-left: 4px solid rgba(70,130,180,0.55);
            padding: 0.75rem 0.9rem;
            background: rgba(70,130,180,0.06);
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .m4-mini-note {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.5rem;
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
        f"""
        <div class="m4-card">
            <div class="m4-card-kicker">{title}</div>
            <div class="m4-card-value">{value}</div>
            <div class="m4-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_info_note(text: str) -> None:
    st.markdown(f"<div class='m4-note'>{text}</div>", unsafe_allow_html=True)


def _render_theory_note(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="m4-mini-note">
            <div style="font-weight:700; margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:0.95rem;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_section(context: EvaluationContext, artifacts: EvaluationArtifacts):
    st.markdown(
        f"""
        <div class="m4-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Фінальне оцінювання та порівняння результатів</div>
            <div style="margin-bottom:0.6rem;">
                На цьому етапі підсумовуються результати базових моделей і ансамблів другого рівня,
                визначається найкраща модель поточної стратегії та формується фінальний звіт експерименту.
            </div>
            <span class="m4-pill">{_strategy_label(context.strategy_type)}</span>
            <span class="m4-pill">Оцінювання якості моделей</span>
            <span class="m4-pill">Порівняння стратегій</span>
            <span class="m4-pill">Теорія метрик</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Стратегія", _strategy_short_label(context.strategy_type), "Активна схема моделювання")
    with c2:
        _render_html_card("Цільова ознака", str(context.target_col), "Target поточного експерименту")
    with c3:
        _render_html_card("Базові моделі", str(len(context.first_level_results)), "Кількість моделей першого рівня")
    with c4:
        _render_html_card("Ансамблі 2-го рівня", str(len(context.second_level_results)), "Кількість ансамблевих моделей")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        _render_html_card("Найкраща модель", str(artifacts.best_model["name"]), "Лідер за головною метрикою")
    with b2:
        _render_html_card("Рівень", str(artifacts.best_model["level"]), "Перший або другий рівень")
    with b3:
        _render_html_card("Головна метрика", str(artifacts.best_model["metric"]), "Критерій фінального порівняння")
    with b4:
        _render_html_card("Значення", f"{artifacts.best_model['value']:.4f}", "Найкращий результат поточної стратегії")


def _render_experiment_snapshot(context: EvaluationContext):
    _render_section_header(
        "1. Поточний стан експерименту",
        "Короткий огляд конфігурації, на основі якої виконується фінальна оцінка моделей.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_html_card("Q1", _metric_value(context.q1), "Нижній поріг категоризації")
    with c2:
        _render_html_card("Q3", _metric_value(context.q3), "Верхній поріг категоризації")
    with c3:
        _render_html_card("Метаознаки", "Сформовано" if context.meta_ready else "Ні", "Стан етапу stacking")
    with c4:
        _render_html_card(
            "Параметри етапів",
            f"{int(bool(context.first_level_params)) + int(bool(context.meta_generation_params)) + int(bool(context.second_level_params))}/3",
            "Заповнені службові конфігурації",
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    with st.expander("Показати деталі конфігурації", expanded=False):
        overview_df = pd.DataFrame(
            [
                {"Параметр": "Стратегія", "Значення": _strategy_label(context.strategy_type)},
                {"Параметр": "Target", "Значення": context.target_col},
                {"Параметр": "Q1", "Значення": context.q1},
                {"Параметр": "Q3", "Значення": context.q3},
                {"Параметр": "Метаознаки сформовано", "Значення": context.meta_ready},
            ]
        )
        st.dataframe(overview_df.astype(str), width="stretch", hide_index=True)
        st.info(
            "Для класифікації головною метрикою підсумкового порівняння є F1 weighted. "
            "Для регресійної стратегії використовується F1 weighted after discretization, "
            "тобто після переведення неперервних прогнозів у класи за межами Q1 і Q3."
        )


def _prepare_ranking_df(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ranking_df = comparison_df.copy()
    ranking_df = ranking_df.sort_values(metric, ascending=False).reset_index(drop=True)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)
    return ranking_df


def _render_comparison_section(context: EvaluationContext, artifacts: EvaluationArtifacts):
    metric = _get_primary_metric(context.strategy_type)
    _render_section_header(
        "2. Порівняння результатів",
        "Зведений підсумок для моделей першого рівня та ансамблів другого рівня в межах поточної стратегії.",
    )

    ranking_df = _prepare_ranking_df(artifacts.comparison_df, metric)
    best_value = float(ranking_df.loc[0, metric])
    second_value = float(ranking_df.loc[1, metric]) if len(ranking_df) > 1 else best_value
    gap = best_value - second_value

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_html_card("Найкраща модель", artifacts.best_model["name"], "Лідер загального рейтингу")
    with c2:
        _render_html_card("Тип моделі", artifacts.best_model["family"], "Базова модель або ансамбль")
    with c3:
        _render_html_card("Відрив від 2-го місця", f"{gap:.4f}", f"За метрикою {metric}")

    _render_info_note(
        f"Фінальне ранжування виконується за метрикою <b>{metric}</b>. "
        f"Найкращий результат у поточній стратегії показала модель <b>{artifacts.best_model['name']}</b> "
        f"({artifacts.best_model['level']}, {artifacts.best_model['family']})."
    )

    st.markdown("#### Рейтинг моделей")
    summary_df = ranking_df[["Rank", "Model", "Level", "Family", metric]].copy()
    st.dataframe(summary_df.astype(str), width="stretch", hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    first_level_df = ranking_df[ranking_df["Level"] == "Перший рівень"].copy()
    second_level_df = ranking_df[ranking_df["Level"] == "Другий рівень"].copy()

    left, right = st.columns(2)
    with left:
        st.markdown("#### Базові моделі")
        if first_level_df.empty:
            st.info("Немає даних для базових моделей.")
        else:
            st.dataframe(
                first_level_df[["Rank", "Model", "Family", metric]],
                width="stretch",
                hide_index=True,
            )
    with right:
        st.markdown("#### Ансамблі другого рівня")
        if second_level_df.empty:
            st.info("Немає даних для ансамблів другого рівня.")
        else:
            st.dataframe(
                second_level_df[["Rank", "Model", "Family", metric]],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Показати повну таблицю метрик", expanded=False):
        display_df = artifacts.comparison_df.copy()
        st.dataframe(display_df, width="stretch", hide_index=True)


def _render_classification_metrics(result):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric_value(result["Accuracy"]))
    c2.metric("Balanced Accuracy", _metric_value(result["Balanced_accuracy"]))
    c3.metric("F1 weighted", _metric_value(result["F1_weighted"]))


def _render_regression_metrics(result):
    st.markdown("**Регресійні метрики**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _metric_value(result["MAE"]))
    c2.metric("MSE", _metric_value(result["MSE"]))
    c3.metric("RMSE", _metric_value(result["RMSE"]))
    c4.metric("R²", _metric_value(result["R2"]))

    st.markdown("**Після категоризації**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric_value(result["Accuracy_after_discretization"]))
    c2.metric("Balanced Accuracy", _metric_value(result["Balanced_accuracy_after_discretization"]))
    c3.metric("F1 weighted", _metric_value(result["F1_weighted_after_discretization"]))


def _classification_metric_rows(result):
    return pd.DataFrame(
        [
            {"Метрика": "Accuracy", "Значення": float(result["Accuracy"])},
            {"Метрика": "Balanced Accuracy", "Значення": float(result["Balanced_accuracy"])},
            {"Метрика": "F1 weighted", "Значення": float(result["F1_weighted"])},
        ]
    )


def _regression_metric_rows(result):
    return pd.DataFrame(
        [
            {"Група": "Регресія", "Метрика": "MAE", "Значення": float(result["MAE"])},
            {"Група": "Регресія", "Метрика": "MSE", "Значення": float(result["MSE"])},
            {"Група": "Регресія", "Метрика": "RMSE", "Значення": float(result["RMSE"])},
            {"Група": "Регресія", "Метрика": "R²", "Значення": float(result["R2"])},
            {"Група": "Після категоризації", "Метрика": "Accuracy", "Значення": float(result["Accuracy_after_discretization"])},
            {"Група": "Після категоризації", "Метрика": "Balanced Accuracy", "Значення": float(result["Balanced_accuracy_after_discretization"])},
            {"Група": "Після категоризації", "Метрика": "F1 weighted", "Значення": float(result["F1_weighted_after_discretization"])},
        ]
    )


def _render_result_block(strategy_type, title, result):
    st.markdown(f"### {title}")

    if strategy_type == CLASSIFICATION_STRATEGY:
        _render_classification_metrics(result)

        metric_df = _classification_metric_rows(result)
        st.dataframe(metric_df, width="stretch", hide_index=True)

        if "used_probability_groups" in result:
            st.caption("Використані probability-блоки")
            st.write(result["used_probability_groups"])

        if "classification_report" in result:
            with st.expander("Показати classification report"):
                st.text(result["classification_report"])
        return

    _render_regression_metrics(result)

    metric_df = _regression_metric_rows(result)
    st.dataframe(metric_df, width="stretch", hide_index=True)

    if "classification_report_after_discretization" in result:
        with st.expander("Показати звіт після категоризації"):
            st.text(result["classification_report_after_discretization"])


def _ordered_second_level_names(strategy_type, second_level_results):
    preferred = ["SoftVoting", "Stacking"] if strategy_type == CLASSIFICATION_STRATEGY else ["Voting", "Stacking"]
    existing = [name for name in preferred if name in second_level_results]
    remaining = [name for name in second_level_results.keys() if name not in existing]
    return existing + remaining


def _render_details_section(context: EvaluationContext):
    _render_section_header(
        "3. Деталі моделей",
        "Розгорнуті метрики для базових моделей і ансамблів другого рівня.",
    )

    second_level_names = _ordered_second_level_names(context.strategy_type, context.second_level_results)
    tab_titles = ["Базові моделі"] + second_level_names[:2] if len(second_level_names) >= 2 else ["Базові моделі", "Ансамблі 2-го рівня"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        for model_name, result in context.first_level_results.items():
            with st.container(border=True):
                _render_result_block(context.strategy_type, model_name, result)

    if len(second_level_names) >= 2:
        with tabs[1]:
            with st.container(border=True):
                _render_result_block(
                    context.strategy_type,
                    second_level_names[0],
                    context.second_level_results[second_level_names[0]],
                )
        with tabs[2]:
            with st.container(border=True):
                _render_result_block(
                    context.strategy_type,
                    second_level_names[1],
                    context.second_level_results[second_level_names[1]],
                )
        if len(second_level_names) > 2:
            with st.expander("Додаткові ансамблі другого рівня"):
                for extra_name in second_level_names[2:]:
                    with st.container(border=True):
                        _render_result_block(
                            context.strategy_type,
                            extra_name,
                            context.second_level_results[extra_name],
                        )
        return

    with tabs[1]:
        for ensemble_name, result in context.second_level_results.items():
            with st.container(border=True):
                _render_result_block(context.strategy_type, ensemble_name, result)


def _build_cross_strategy_best_df(snapshots):
    direct_best = snapshots[CLASSIFICATION_STRATEGY]["best_model"]
    reg_best = snapshots[REGRESSION_STRATEGY]["best_model"]
    return pd.DataFrame(
        [
            {
                "Strategy": snapshots[CLASSIFICATION_STRATEGY]["strategy_label"],
                "Best model": direct_best["name"],
                "Level": direct_best["level"],
                "Family": direct_best["family"],
                "Metric": direct_best["metric"],
                "Value": direct_best["value"],
            },
            {
                "Strategy": snapshots[REGRESSION_STRATEGY]["strategy_label"],
                "Best model": reg_best["name"],
                "Level": reg_best["level"],
                "Family": reg_best["family"],
                "Metric": reg_best["metric"],
                "Value": reg_best["value"],
            },
        ]
    )


def _build_cross_strategy_second_level_df(snapshots):
    rows = []

    for model_name, result in snapshots[CLASSIFICATION_STRATEGY]["second_level_results"].items():
        rows.append(
            {
                "Strategy": "Класифікація",
                "Model": model_name,
                "Metric": "F1 weighted",
                "Value": float(result["F1_weighted"]),
            }
        )

    for model_name, result in snapshots[REGRESSION_STRATEGY]["second_level_results"].items():
        rows.append(
            {
                "Strategy": "Регресія + категоризація",
                "Model": model_name,
                "Metric": "F1 weighted after discretization",
                "Value": float(result["F1_weighted_after_discretization"]),
            }
        )

    return pd.DataFrame(rows)


def _render_cross_strategy_comparison():
    _render_section_header(
        "4. Міжстратегічне порівняння",
        "Порівняння найкращих результатів двох стратегій у табличному вигляді.",
    )

    snapshots = st.session_state.get("strategy_snapshots", {})
    needed = {CLASSIFICATION_STRATEGY, REGRESSION_STRATEGY}

    if not needed.issubset(set(snapshots.keys())):
        st.info(
            "Щоб побачити порівняння двох стратегій, виконайте фінальне оцінювання окремо "
            "для прямої класифікації та для регресії з подальшою категоризацією."
        )
        return

    final_df = _build_cross_strategy_best_df(snapshots)
    second_level_df = _build_cross_strategy_second_level_df(snapshots)

    st.markdown("#### Найкращі моделі двох стратегій")
    st.dataframe(final_df, width="stretch", hide_index=True)

    st.markdown("#### Ансамблі другого рівня")
    left, right = st.columns(2)

    with left:
        st.markdown("**Класифікація**")
        cls_df = second_level_df[second_level_df["Strategy"] == "Класифікація"].copy()
        if cls_df.empty:
            st.info("Немає snapshot для класифікації.")
        else:
            st.dataframe(cls_df[["Model", "Metric", "Value"]], width="stretch", hide_index=True)

    with right:
        st.markdown("**Регресія + категоризація**")
        reg_df = second_level_df[second_level_df["Strategy"] == "Регресія + категоризація"].copy()
        if reg_df.empty:
            st.info("Немає snapshot для регресійної стратегії.")
        else:
            st.dataframe(reg_df[["Model", "Metric", "Value"]], width="stretch", hide_index=True)

    with st.expander("Показати повну таблицю міжстратегічного порівняння", expanded=False):
        st.dataframe(second_level_df, width="stretch", hide_index=True)


def _render_theory_section():
    _render_section_header(
        "5. Теоретичне пояснення метрик",
        "Короткі пояснення змісту метрик класифікації та регресії, які використовуються на сторінці.",
    )

    tab_cls, tab_reg = st.tabs(["Метрики класифікації", "Метрики регресії"])

    with tab_cls:
        _render_theory_note(
            "Accuracy",
            "Частка правильно класифікованих об'єктів серед усіх об'єктів тестової вибірки. "
            "Метрика проста для інтерпретації, але може бути неінформативною за дисбалансу класів.",
        )
        _render_theory_note(
            "Balanced Accuracy",
            "Середнє значення recall по всіх класах. Дає більш збалансовану оцінку, "
            "коли класи представлені нерівномірно.",
        )
        _render_theory_note(
            "Precision",
            "Показує, яка частка об'єктів, віднесених моделлю до певного класу, "
            "дійсно належить до цього класу.",
        )
        _render_theory_note(
            "Recall",
            "Показує, яку частку об'єктів певного справжнього класу модель змогла коректно знайти.",
        )
        _render_theory_note(
            "F1 weighted",
            "Зважене середнє F1-міри за всіма класами. Поєднує precision і recall та "
            "враховує кількість об'єктів у кожному класі. Саме ця метрика використовується "
            "як головна для фінального порівняння моделей класифікації.",
        )
        _render_theory_note(
            "Classification report",
            "Табличний текстовий звіт зі значеннями precision, recall, F1-score і support "
            "для кожного класу окремо, а також середніми значеннями.",
        )

    with tab_reg:
        _render_theory_note(
            "MAE",
            "Середня абсолютна помилка прогнозу. Показує, на скільки одиниць цільової змінної "
            "у середньому помиляється модель. Має ті самі одиниці вимірювання, що й target.",
        )
        _render_theory_note(
            "MSE",
            "Середня квадратична помилка. Сильніше штрафує великі відхилення, ніж MAE. "
            "Має квадрат одиниць вимірювання цільової змінної.",
        )
        _render_theory_note(
            "RMSE",
            "Корінь із середньоквадратичної помилки. Має ті самі одиниці вимірювання, "
            "що й target, тому зручніший для практичної інтерпретації, ніж MSE.",
        )
        _render_theory_note(
            "R²",
            "Коефіцієнт детермінації. Показує, яку частку варіації цільової змінної "
            "пояснює модель. Це безрозмірна метрика; чим ближче до 1, тим краще.",
        )
        _render_theory_note(
            "Accuracy / Balanced Accuracy / F1 weighted after discretization",
            "Для регресійної стратегії неперервні прогнози додатково переводяться у класи "
            "за порогами Q1 і Q3. Після цього обчислюються класифікаційні метрики, що дає "
            "змогу зіставити якість регресійної стратегії з прямою багатокласовою класифікацією.",
        )
        _render_theory_note(
            "Чому метрики регресії не варто зображати на одному графіку",
            "MAE і RMSE мають одиниці цільової змінної, MSE має квадрат цих одиниць, а R² є "
            "безрозмірною метрикою. Тому спільний bar chart для них методологічно некоректний.",
        )


def _render_export_section(artifacts: EvaluationArtifacts):
    _render_section_header(
        "6. Експорт підсумкового звіту",
        "Збереження фінального JSON-звіту з параметрами, метриками та snapshots стратегій.",
    )
    json_bytes = _results_to_json_bytes(artifacts.final_report)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.download_button(
            on_click="ignore",
            label="Завантажити фінальний звіт (JSON)",
            data=json_bytes,
            file_name="final_evaluation_report.json",
            mime="application/json",
            width="stretch",
        )
    with right:
        st.info(
            "JSON містить підсумкову таблицю порівняння, параметри етапів та snapshots "
            "для міжстратегічного аналізу."
        )

    with st.expander("Попередній перегляд JSON", expanded=False):
        st.json(artifacts.final_report)


def page_evaluation_results():
    _inject_page_style()
    context = _load_context()
    if context is None:
        return

    artifacts = _build_artifacts(context)

    _render_hero_section(context, artifacts)
    _render_experiment_snapshot(context)
    _render_comparison_section(context, artifacts)
    _render_details_section(context)
    _render_cross_strategy_comparison()
    _render_theory_section()
    from evaluation_extras import render_extended_evaluation
    render_extended_evaluation()
    _render_export_section(artifacts)
    from export_tools import all_exports_zip
    if st.button("Підготувати повний архів таблиць і графіків"):
        st.session_state.complete_export=all_exports_zip()
    if "complete_export" in st.session_state:
        st.download_button("Завантажити всі результати (ZIP)",st.session_state.complete_export,"experiment_results.zip","application/zip",on_click="ignore")