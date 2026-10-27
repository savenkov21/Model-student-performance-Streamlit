from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix,log_loss,roc_auc_score
from statistical_analysis import normalize,render_statistics


def new_meta(run,X,classification):
    cols={}
    for name,models in run['fold_models'].items():
        if classification:
            p=np.mean([normalize(m.predict_proba(X)) for m in models],axis=0)
            for c in range(p.shape[1]):cols[f'meta_{name}_proba_{c}']=p[:,c]
        else:cols[f'meta_{name}']=np.mean([m.predict(X) for m in models],axis=0)
    meta=pd.DataFrame(cols,index=X.index)
    missing=[c for c in run['meta_columns'] if c not in meta]
    if missing:
        prep=run['preprocessor'];transformed=pd.DataFrame(prep.transform(X),columns=prep.get_feature_names_out(),index=X.index)
        meta=pd.concat([transformed,meta],axis=1)
    return meta[run['meta_columns']]


def new_prediction(run,name,X,classification):
    if name in run['first_models']:
        model=run['first_models'][name]
        return normalize(model.predict_proba(X)) if classification else model.predict(X)
    meta=new_meta(run,X,classification)
    if name=='Stacking':
        model=run['second_models'][name]
        return normalize(model.predict_proba(meta)) if classification else model.predict(meta)
    if classification:
        from module_6_visual_refined_updated import _soft_voting_from_meta_proba
        return _soft_voting_from_meta_proba(meta,[0,1,2],run['second_params']['SoftVoting']['weights'])[0]
    weights=run['second_params']['Voting']['weights'];values=meta.filter(regex='^meta_').to_numpy()
    if isinstance(weights,str) and weights=='equal':weights=None
    return values.mean(1) if weights is None else np.average(values,axis=1,weights=weights)


def render_extended_evaluation():
    strategy=st.session_state.strategy_type;run=st.session_state.evaluation_runs[strategy]
    cls=strategy=='direct_multiclass_classification'
    st.subheader('Графіки моделей та прогнозів')
    names=list(run['predictions'])
    name=st.selectbox('Модель для графіків',names,key='extra_plot_model')
    fig,ax=plt.subplots(figsize=(5,4));cm=confusion_matrix(run['y_cls'],run['predictions'][name],labels=[0,1,2]);ax.imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Predicted class',ylabel='Actual class',title=name);fig.tight_layout();st.pyplot(fig)
    if not cls:
        fig,ax=plt.subplots(figsize=(6,5));p=run['continuous'][name];y=run['y_cont'];ax.scatter(y,p,alpha=.5,s=18);low=min(y.min(),p.min());high=max(y.max(),p.max());ax.plot([low,high],[low,high],'r--');ax.set(xlabel='Actual Performance Index',ylabel='Predicted Performance Index',title=name);fig.tight_layout();st.pyplot(fig)
        fig,ax=plt.subplots(figsize=(7,4));ax.scatter(p,y-p,alpha=.5,s=18);ax.axhline(0,color='red',ls='--');ax.set(xlabel='Predicted grade',ylabel='Residual',title=name);fig.tight_layout();st.pyplot(fig)
    elif run['probabilities']:
        rows=[]
        for n,p in run['probabilities'].items():
            y=run['y_cls'];onehot=np.eye(3)[y]
            rows.append({'Model':n,'Log loss':log_loss(y,p,labels=[0,1,2]),'Brier multiclass':float(np.mean(np.sum((onehot-p)**2,axis=1))),'ROC AUC macro OVR':roc_auc_score(y,p,multi_class='ovr',average='macro')})
        st.markdown('**Оцінювання ймовірнісних прогнозів**');st.dataframe(pd.DataFrame(rows))
        pn=st.selectbox('Модель для ймовірностей',list(run['probabilities']),key='proba_model')
        i=int(st.number_input('Номер тестового запису',min_value=0,max_value=len(run['y_cls'])-1,value=0))
        fig,ax=plt.subplots(figsize=(6,3));ax.bar(['Low','Medium','High'],run['probabilities'][pn][i]);ax.set_ylim(0,1);ax.set(ylabel='Probability',title=f'{pn}: test record {i}, true class {run["y_cls"][i]}');fig.tight_layout();st.pyplot(fig)
    render_statistics()
    st.subheader('Прогнозування нових даних')
    ns=st.selectbox('Стратегія для прогнозування',list(st.session_state.evaluation_runs),format_func=lambda s:'Класифікація' if s=='direct_multiclass_classification' else 'Регресія з категоризацією',key='prediction_strategy')
    nr=st.session_state.evaluation_runs[ns];ncls=ns=='direct_multiclass_classification'
    nm=st.selectbox('Модель для нового запису',list(nr['predictions']),key='prediction_model')
    sample=st.session_state.X_train_raw.iloc[[0]][nr['features']].reset_index(drop=True)
    edited=st.data_editor(sample,num_rows='fixed',key='new_record_editor')
    uploaded=st.file_uploader('Або CSV з новими записами та тими самими предикторами',type='csv',key='new_prediction_csv')
    if st.button('Сформувати прогноз'):
        try:
            X=pd.read_csv(uploaded,sep=None,engine='python') if uploaded is not None else edited
            X=X[nr['features']];p=new_prediction(nr,nm,X,ncls);out=X.copy()
            if ncls:
                out['predicted_class']=p.argmax(1)
                for c in range(3):out[f'P{c}']=p[:,c]
            else:out['predicted_grade']=p;out['predicted_class']=np.digitize(p,[nr['Q1'],nr['Q3']],right=True)
            st.session_state.new_prediction_output=out
        except Exception as e:st.error(str(e))
    if 'new_prediction_output' in st.session_state:
        out=st.session_state.new_prediction_output;st.dataframe(out)
        st.download_button('Завантажити нові прогнози CSV',out.to_csv(index=False),'new_predictions.csv','text/csv',on_click='ignore')
