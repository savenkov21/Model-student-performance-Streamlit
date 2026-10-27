import streamlit as st, pandas as pd
import hashlib
from pathlib import Path
from data_protocol import register_data

from module_3 import _draw_panels, _numeric_columns

def _load_df(_file) -> pd.DataFrame:
    """Load a DataFrame from a file, trying different encodings if necessary."""
    try:
        df = pd.read_csv(_file, sep=None, engine="python", encoding="utf-8-sig")
        return df
    except UnicodeDecodeError: 
        return pd.read_csv(_file, sep=None, engine='python', encoding='latin-1')

def _reset_ml_state():
    """Очистити все, що пов'язано з ML-пайплайном, коли завантажили новий файл."""
    preserved = {'file_name','df_orig','df','df_before','log','source_hash','row_groups','grouped_data'}
    for key in list(st.session_state):
        if key not in preserved and not key.startswith('FormSubmitter:'):
            st.session_state.pop(key,None)


def page_analysis():
    st.subheader("Завантаження та аналіз даних")
    st.write("Завантажте CSV-файл та отримайте базовий огляд даних.")
    
    file = st.file_uploader("Виберіть CSV-файл", type=["csv"])
    
    use_bundled=st.button("Завантажити вбудовані португальські дані")
    if file is not None or use_bundled:
        data=file.getvalue() if file is not None and not use_bundled else (Path(__file__).parent/'DataSet/Student_Performance_Portugal_Numeric.csv').read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        if digest!=st.session_state.get('source_hash'):
            from io import BytesIO
            df_loaded=_load_df(BytesIO(data))
            _reset_ml_state()
            st.session_state.file_name=file.name if file is not None and not use_bundled else 'Student_Performance_Portugal_Numeric.csv'
            st.session_state.source_hash=digest
            st.session_state.df_orig=df_loaded.copy();st.session_state.df=df_loaded.copy();st.session_state.df_before=df_loaded.copy()
            st.session_state.log=[]
            register_data(df_loaded)
            st.success("Набір даних успішно завантажено!")

    if "df" in st.session_state:
        df = st.session_state.df
        
        st.markdown("### Попередній перегляд даних")
        st.dataframe(df.head(20), width="stretch")
        
        st.markdown("### Основна інформація")
        st.write(f"Кількість рядків: **{df.shape[0]}**, кількість стовпців: **{df.shape[1]}**")
        st.write(df.dtypes.astype(str).to_frame("dtype"))
        
        st.markdown("### Швидка перевірка якості даних")
        missing = df.isna().sum()
        dup_count = df.duplicated().sum()
        rows_with_nan = df.isna().any(axis=1).sum()
        
        st.write(f"Рядків з пропущеними значеннями: {rows_with_nan}")
        st.write("Розподіл пропущених значень по стовпцях:")
        if (missing > 0).any():
            st.write(missing[missing > 0])
        else:
            st.info("Немає пропущених значень.")
        
        st.write(f"Кількість дублікатів: **{dup_count}**")
        
        st.markdown("### Статистичні показники числових змінних")
        num_cols = _numeric_columns(df)
        if num_cols:
            st.write(df[num_cols].describe().T)
        else:
            st.info("Числових колонок не знайдено.")
            
        if num_cols:
            st.markdown("### Візуалізація числових змінних")
            plot_type = st.selectbox("Тип графіка", options=["Histogram", "Boxplot", "Violin"], index=0)
            selected_cols = st.multiselect("Виберіть стовпці для графіків", options=num_cols, default=num_cols)
            if st.button("Побудувати графіки", width="stretch"):
                _draw_panels(df, selected_cols, plot_type, mode_tag="before_norm")
            
    else:
        st.info("Поки що не завантажено жодного файлу. Використайте кнопку вище для завантаження.")
