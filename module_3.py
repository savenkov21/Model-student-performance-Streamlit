import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from io import BytesIO
from scipy.stats import pearsonr,spearmanr,pointbiserialr
from data_protocol import ORDINAL,BINARY,CONTINUOUS


def _numeric_columns(df):return df.select_dtypes(include='number').columns.tolist()


def _draw_panels(df,cols,kind,mode_tag):
    cols=[c for c in cols if c in df]
    if not cols:st.info('Виберіть ознаки.');return
    ncols=min(3,len(cols));nrows=ceil(len(cols)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,3.6*nrows),squeeze=False)
    for ax,col in zip(axes.flat,cols):
        values=df[col].dropna()
        if kind=='Histogram':ax.hist(values,bins=15,color='salmon' if mode_tag=='before_norm' else 'skyblue',edgecolor='white');ax.set_ylabel('Count')
        elif kind=='Boxplot':ax.boxplot(values,vert=True);ax.set_ylabel(col)
        else:sns.violinplot(y=values,ax=ax,color='skyblue');ax.set_ylabel(col)
        ax.set_title(col);ax.tick_params(labelsize=9)
    for ax in list(axes.flat)[len(cols):]:fig.delaxes(ax)
    fig.suptitle('Before preprocessing' if mode_tag=='before_norm' else 'After preprocessing')
    fig.tight_layout();st.pyplot(fig)


def visualization_frames():
    if not st.session_state.get('split_ready',False):return st.session_state.df.copy(),None
    return st.session_state.visual_before.copy(),st.session_state.visual_after.copy()


def association_matrix(df,cols,mode):
    matrix=pd.DataFrame(np.eye(len(cols)),index=cols,columns=cols);methods=matrix.astype(object)
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i==j:methods.loc[a,b]='self';continue
            values=df[[a,b]].dropna()
            if mode=='Spearman' or a in ORDINAL or b in ORDINAL:
                r,_=spearmanr(values[a],values[b]);label='Spearman'
            elif a in BINARY or b in BINARY:
                binary=a if a in BINARY else b;other=b if binary==a else a
                r,_=pointbiserialr(values[binary],values[other]);label='Point-biserial'
            else:r,_=pearsonr(values[a],values[b]);label='Pearson'
            matrix.loc[a,b]=r;methods.loc[a,b]=label
    return matrix,methods


def page_visualization():
    if 'df' not in st.session_state:st.warning('Спочатку завантажте дані.');return
    before,after=visualization_frames()
    st.subheader('Розподіли до та після preprocessing')
    st.caption('Показано ту саму навчальну частину. Ціль не масштабується. Стандартизація змінює шкалу, але не форму розподілу; порядкові коди не стають неперервними вимірюваннями.')
    if after is None:st.info('Для порівняння сформуйте вибірки на етапі 2.')
    modes=['До нормалізації','Після нормалізації','Порівняти до / після']
    mode=st.radio('Режим візуалізації:',modes,horizontal=True)
    kind=st.selectbox('Тип графіка',['Histogram','Boxplot','Violin'])
    cols=st.multiselect('Виберіть стовпці для візуалізації',_numeric_columns(before),default=[c for c in ['age','absences','G1'] if c in before] or _numeric_columns(before)[:3])
    if mode in [modes[0],modes[2]]:_draw_panels(before,cols,kind,'before_norm')
    if mode in [modes[1],modes[2]]:
        if after is not None:_draw_panels(after,cols,kind,'after_norm')
    if after is not None:
        common=[c for c in before if c in after and c!=st.session_state.target_col]
        summary=pd.DataFrame({'Mean before':before[common].mean(),'SD before (ddof=0)':before[common].std(ddof=0),'Mean after':after[common].mean(),'SD after (ddof=0)':after[common].std(ddof=0)})
        st.markdown('**Перевірка перетворення ознак на train**');st.dataframe(summary)
        st.download_button('Зберегти перевірку масштабування',summary.to_csv(index_label='Feature'),'scaling_check.csv','text/csv',on_click='ignore')
        if st.session_state.preprocessing_config.get('scaler_type')=='StandardScaler':st.caption('Для несталих стандартизованих ознак: mean ≈ 0, SD ≈ 1. subject_math залишається 0/1.')
    st.subheader('Кореляційний аналіз з урахуванням типів ознак')
    st.caption('Кількісні: age, absences, G1 і target. Порядкові: Medu, Fedu, traveltime, studytime, failures. Бінарна: subject_math. Аналіз виконується на початкових train-значеннях.')
    method=st.selectbox('Метод кореляції',['За типом ознак','Spearman','Pearson (лише кількісні)'])
    options=[c for c in _numeric_columns(before) if method!='Pearson (лише кількісні)' or c not in ORDINAL+BINARY]
    selected=st.multiselect('Ознаки для матриці',options,default=options)
    if len(selected)>=2:
        corr,methods=association_matrix(before,selected,method)
        fig,ax=plt.subplots(figsize=(10,8));sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',vmin=-1,vmax=1,ax=ax);ax.set_title('Associations: '+method);fig.tight_layout();st.pyplot(fig)
        st.download_button('Матриця асоціацій CSV',corr.to_csv(index_label='Feature'),'associations.csv','text/csv',on_click='ignore')
        with st.expander('Метод для кожної пари ознак'):st.dataframe(methods)
        target=st.session_state.get('target_col',before.columns[-1])
        if target in corr:
            values=corr[target].drop(target).sort_values()
            fig,ax=plt.subplots(figsize=(8,5));ax.barh(values.index,values.values);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Association with target');fig.tight_layout();st.pyplot(fig)
            st.caption('Це сила асоціації, а не автоматичний відбір ознак чи доказ причинного впливу. Всі вибрані предиктори залишаються в моделях.')
