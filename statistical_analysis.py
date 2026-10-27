from itertools import combinations
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binomtest,chi2
from sklearn.metrics import log_loss,roc_auc_score


def normalize(p):
    p=np.asarray(p,float);return p/p.sum(1,keepdims=True)


def capture_run(table):
    cls=st.session_state.strategy_type=='direct_multiclass_classification'
    first=st.session_state.first_level_predictions_cls
    second=st.session_state.second_level_predictions_cls
    predictions={**{k:np.asarray(v).copy() for k,v in first.items()},**{k:np.asarray(v).copy() for k,v in second.items()}}
    probabilities={};continuous={}
    if cls:
        for name,model in st.session_state.first_level_models.items():
            if hasattr(model,'predict_proba'):probabilities[name]=normalize(model.predict_proba(st.session_state.X_test_raw))
        probabilities['SoftVoting']=normalize(st.session_state.second_level_predictions['SoftVoting'])
        probabilities['Stacking']=normalize(st.session_state.second_level_models['Stacking'].predict_proba(st.session_state.meta_X_test))
    else:continuous={**st.session_state.first_level_predictions,**st.session_state.second_level_predictions}
    from data_protocol import groups_for
    run={'signature':st.session_state.split_signature,'table':table.copy(),'test_index':st.session_state.X_test_raw.index.to_numpy().copy(),
         'y_cont':st.session_state.y_test_cont.to_numpy().copy(),'y_cls':st.session_state.y_test_cls.to_numpy().copy(),
         'groups':groups_for(st.session_state.X_test_raw),'predictions':predictions,'probabilities':probabilities,'continuous':continuous,
         'first_models':st.session_state.first_level_models.copy(),'fold_models':st.session_state.get('fold_prediction_models',{}).copy(),
         'meta_columns':list(st.session_state.meta_X_test.columns),'second_models':st.session_state.second_level_models.copy(),
         'second_params':st.session_state.second_level_params.copy(),'preprocessor':st.session_state.preprocessor,
         'features':list(st.session_state.X_train_raw.columns),'Q1':st.session_state.Q1,'Q3':st.session_state.Q3}
    old=st.session_state.get('evaluation_runs',{}).get(st.session_state.strategy_type)
    if old is None or any(k not in old['predictions'] or not np.array_equal(v,old['predictions'][k]) for k,v in predictions.items()):
        for key in ['statistical_tables','statistics_token','complete_export']:
            st.session_state.pop(key,None)
    st.session_state.setdefault('evaluation_runs',{})[st.session_state.strategy_type]=run


def class_metrics(y,p):
    cm=np.bincount(3*np.asarray(y,int)+np.asarray(p,int),minlength=9).reshape(3,3)
    n=cm.sum();sup=cm.sum(1);pre=cm.sum(0);tp=cm.diagonal()
    recall=np.divide(tp,sup,out=np.zeros(3),where=sup>0)
    f1=np.divide(2*tp,sup+pre,out=np.zeros(3),where=(sup+pre)>0)
    return {'Accuracy':float(tp.sum()/n),'Balanced Accuracy':float(recall[sup>0].mean()),'F1 macro':float(f1.mean()),'F1 weighted':float(np.dot(f1,sup)/n)}


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adj


def compute_statistics(runs,B=2000,mode='stratified',seed=42):
    entries={};first=next(iter(runs.values()));y=first['y_cls'];groups=first['groups']
    for strategy,r in runs.items():
        if not np.array_equal(first['test_index'],r['test_index']) or not np.array_equal(y,r['y_cls']):raise ValueError('Стратегії мають різні тестові записи або класи. Повторіть їх на спільному split.')
        for name,p in r['predictions'].items():entries[f'{strategy}/{name}']=np.asarray(p)
    point={k:class_metrics(y,p) for k,p in entries.items()};sample={k:{m:[] for m in v} for k,v in point.items()};rng=np.random.default_rng(seed)
    groupidx=[np.flatnonzero(groups==g) for g in np.unique(groups)];classidx=[np.flatnonzero(y==c) for c in range(3)]
    for _ in range(B):
        if mode=='stratified':idx=np.concatenate([rng.choice(v,len(v),replace=True) for v in classidx if len(v)])
        else:idx=np.concatenate([groupidx[k] for k in rng.integers(0,len(groupidx),len(groupidx))])
        if len(np.unique(y[idx]))<3:continue
        for name,p in entries.items():
            for metric,v in class_metrics(y[idx],p[idx]).items():sample[name][metric].append(v)
    intervals=[]
    for name,met in sample.items():
        for m,vals in met.items():
            low,high=np.percentile(vals,[2.5,97.5]);intervals.append({'Model':name,'Metric':m,'Estimate':point[name][m],'CI low':low,'CI high':high,'B valid':len(vals),'Bootstrap':mode})
    pairs=[];tests=[]
    # Group representatives: selection independent of target and models.
    rng=np.random.default_rng(seed);independent_idx=np.array([rng.choice(v) for v in groupidx])
    for a,b in combinations(entries,2):
        for metric in ['F1 weighted','F1 macro']:
            delta=np.array(sample[a][metric])-np.array(sample[b][metric]);low,high=np.percentile(delta,[2.5,97.5])
            pairs.append({'A':a,'B':b,'Metric':metric,'Delta A-B':point[a][metric]-point[b][metric],'CI low':low,'CI high':high,'CI excludes zero':bool(low>0 or high<0)})
        for unit,idx in [('records',np.arange(len(y))),('profile groups',independent_idx)]:
            ac=entries[a][idx]==y[idx];bc=entries[b][idx]==y[idx];n10=int(np.sum(ac&~bc));n01=int(np.sum(~ac&bc));n=n10+n01
            exact=n<25;p=1. if n==0 else binomtest(n10,n,.5).pvalue if exact else chi2.sf(max(abs(n10-n01)-1,0)**2/n,1)
            tests.append({'A':a,'B':b,'b':n10,'c':n01,'p raw':p,'Method':'exact' if exact else 'chi2 corrected','Unit':unit,'N':len(idx)})
    mc=pd.DataFrame(tests)
    for unit in mc.Unit.unique():
        mask=mc.Unit==unit;mc.loc[mask,'p Holm']=holm(mc.loc[mask,'p raw'])
    mc['Reject Holm 0.05']=mc['p Holm']<.05
    wald=[]
    for name,p in entries.items():
        acc=point[name]['Accuracy'];err=1.96*np.sqrt(acc*(1-acc)/len(y));wald.append({'Model':name,'Accuracy':acc,'Wald low':max(0,acc-err),'Wald high':min(1,acc+err),'N':len(y)})
    return {'bootstrap_CI':pd.DataFrame(intervals),'paired_bootstrap_F1':pd.DataFrame(pairs),'mcnemar_Holm':mc,'accuracy_Wald':pd.DataFrame(wald)}


def render_statistics():
    runs=st.session_state.get('evaluation_runs',{})
    if not runs:return
    st.subheader('Статистичне порівняння моделей')
    st.caption('Зберігаються результати кожної стратегії після відкриття цієї сторінки. Для спільного порівняння пройдіть етапи 2–7 для другої стратегії з тими самими параметрами split і preprocessing.')
    B=int(st.number_input('Кількість bootstrap-повторень',min_value=100,max_value=10000,value=2000,step=100))
    mode=st.selectbox('Bootstrap',['Стратифікований за класами (як у роботі)','За профільними групами'])
    st.caption('У стратифікованій процедурі кожен запис є одиницею перевибору. Груповий варіант враховує повторні записи учнів. Для Мак-Немара доступні результати за всіма записами та за одним записом кожної профільної групи; останні менше залежать від повторних предметів.')
    token=hashlib.sha256(b''.join(r['y_cls'].tobytes()+b''.join(v.tobytes() for v in r['predictions'].values()) for r in runs.values())).hexdigest()+str((B,mode))
    if st.button('Обчислити довірчі інтервали та парні тести'):
        with st.spinner('Розрахунок статистики…'):
            st.session_state.statistical_tables=compute_statistics(runs,B,'stratified' if mode.startswith('Стратифікований') else 'cluster')
            st.session_state.statistics_token=token
    if st.session_state.get('statistics_token')==token:
        st.caption('Мак-Немар: точний двосторонній тест за b+c<25, інакше χ² з поправкою неперервності. Holm — для всіх пар моделей у відповідній одиниці аналізу. Парні bootstrap-CI є окремими 95% інтервалами, не одночасними. Wald — додаткове нормальне наближення для записів.')
        for name,table in st.session_state.statistical_tables.items():
            with st.expander(name,expanded=name=='mcnemar_Holm'):st.dataframe(table)
    elif 'statistical_tables' in st.session_state:st.info('Моделі або параметри статистики змінилися. Повторіть розрахунок інтервалів і тестів.')
