"""Data identity, shared group splits and fold-local preprocessing."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold

ROOT=Path(__file__).parent
ORDINAL=['Medu','Fedu','traveltime','studytime','failures']
BINARY=['subject_math']
CONTINUOUS=['age','absences','G1']


def frame_digest(df):
    return hashlib.sha256(df.to_csv(index=True).encode()).hexdigest()


def register_data(df):
    bundled=pd.read_csv(ROOT/'DataSet/Student_Performance_Portugal_Numeric.csv')
    if df.reset_index(drop=True).equals(bundled):
        groups=json.loads((ROOT/'DataSet/student_groups.json').read_text())['groups']
        st.session_state.row_groups=pd.Series(groups,index=df.index)
        st.session_state.grouped_data=True
    else:
        st.session_state.row_groups=pd.Series(np.arange(len(df)),index=df.index)
        st.session_state.grouped_data=False


def groups_for(X):
    mapping=st.session_state.get('row_groups')
    if mapping is None or not X.index.isin(mapping.index).all():return np.asarray(X.index)
    return mapping.loc[X.index].to_numpy()


def split_indices(X,test_size,seed):
    return next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(X,groups=groups_for(X)))


def cv_folds(X,y,n_splits=5,seed=42):
    labels=st.session_state.get('y_train_cls')
    if labels is not None and X.index.isin(labels.index).all():labels=labels.loc[X.index].to_numpy()
    else:
        labels=np.asarray(y)
        if np.unique(labels).size>3:labels=np.digitize(labels,np.percentile(labels,[25,75]),right=True)
    groups=groups_for(X)
    return list(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(X,labels,groups))


class GroupCalibratedRidge(ClassifierMixin,BaseEstimator):
    def __init__(self,estimator,preprocessor,group_map,seed=42):
        self.estimator=estimator;self.preprocessor=preprocessor;self.group_map=group_map;self.seed=seed
    def fit(self,X,y):
        groups=np.array([self.group_map.get(int(i),int(i)) for i in X.index])
        folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=self.seed).split(X,y,groups))
        pipe=Pipeline([('preprocessor',clone(self.preprocessor)),('model',clone(self.estimator))])
        self.model_=CalibratedClassifierCV(estimator=pipe,cv=folds,method='sigmoid').fit(X,y)
        self.classes_=self.model_.classes_
        self.n_features_in_=X.shape[1]
        return self
    def predict_proba(self,X):
        p=np.asarray(self.model_.predict_proba(X),float);return p/p.sum(axis=1,keepdims=True)
    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(axis=1)]


def prepare_model(estimator,calibrate=True):
    prep=st.session_state.get('preprocessor')
    if prep is None:return estimator
    if isinstance(estimator,CalibratedClassifierCV):estimator=estimator.estimator
    if isinstance(estimator,RidgeClassifier) and calibrate:
        mapping=st.session_state.get('row_groups',pd.Series(dtype=int)).to_dict()
        return GroupCalibratedRidge(estimator,clone(prep),mapping,int(st.session_state.get('random_state',42)))
    return Pipeline([('preprocessor',clone(prep)),('model',estimator)])


def wrap_models(models):return {name:prepare_model(model) for name,model in models.items()}


def reset_downstream(start=4):
    keys=[]
    if start<=4:keys+=['first_level_models','first_level_results','first_level_predictions','first_level_predictions_cls','first_level_params','first_level_strategy_type','optimization_results']
    if start<=5:keys+=['meta_X_train','meta_X_test','meta_feature_names','meta_generation_params','meta_base_models','meta_full_train_models','meta_ready','meta_strategy_type','fold_prediction_models']
    if start<=6:keys+=['second_level_models','second_level_model','second_level_results','second_level_predictions','second_level_predictions_cls','second_level_params','second_level_strategy_type']
    for key in keys:st.session_state.pop(key,None)
