import streamlit as st
from export_tools import install_figure_capture, render_figure_downloads

from module_1 import page_analysis
from module_2_visual_refined_v2_imputer_na_logic import page_preprocessing
from module_3 import page_visualization
from module_4_visual_refined_updated import page_training_first_level_models
from module_5_visual_refined_updated import page_formation_metafeatures
from module_6_visual_refined_updated import page_training_second_level_model
from module_7_visual_refined_updated import page_evaluation_results

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Model student performance",
    page_icon="🔎",
    layout="centered"  # or "wide"
)
st.title("🔎 Model student performance")

# ----------------------------
# Sidebar: stage selection
# ----------------------------
with st.sidebar:
    st.header("Навігація")
    page = st.radio("Виберіть етап:", 
        options=["Завантаження та первинний аналіз даних", "Попередня обробка даних та формування вибірок", 
                 "Візуалізація", "Навчання базових моделей першого рівня", 
                 "Формування метаознак", "Навчання моделей другого рівня", "Оцінювання та порівняння моделей"], 
        index=0)
    
    st.markdown("---")
    st.caption("Tip: switch to 'wide' layout from the top-right menu if needed.")
    
page_id=["Завантаження та первинний аналіз даних", "Попередня обробка даних та формування вибірок", "Візуалізація", "Навчання базових моделей першого рівня", "Формування метаознак", "Навчання моделей другого рівня", "Оцінювання та порівняння моделей"].index(page)+1
install_figure_capture(page_id)

if page == "Завантаження та первинний аналіз даних":
    page_analysis()
    
elif page == "Попередня обробка даних та формування вибірок":
    page_preprocessing()
    
elif page == "Візуалізація":
    page_visualization()
    
elif page == "Навчання базових моделей першого рівня":
    page_training_first_level_models()

elif page == "Формування метаознак":
    page_formation_metafeatures()

elif page == "Навчання моделей другого рівня":
    page_training_second_level_model()

elif page == "Оцінювання та порівняння моделей":
    page_evaluation_results()
render_figure_downloads(page_id)
