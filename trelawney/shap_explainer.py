from typing import List, Optional, Dict

import pandas as pd
import shap as shap
import sklearn
import numpy as np
import operator
from keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from trelawney.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        self._explainer = None
        self.class_names = class_names
        self._model_to_explain = None
        self._model_type = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self._model_to_explain = model
        self._model_type = type(self._model_to_explain)
        if isinstance(self._model_type, LogisticRegression):
            self._explainer = shap.LinearExplainer(self._model_to_explain, data=x_train)
        elif isinstance(self._model_type, (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)):
            self._explainer = shap.TreeExplainer(self._model_to_explain, data=x_train)
        elif isinstance(self._model_type, models.Model):
            self._explainer = shap.DeepExplainer(self._model_to_explain, data=x_train)
        else:
            self._explainer = shap.KernelExplainer(self._model_to_explain, data=x_train)

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        shap_values = self._explainer.shap_values(x_explain)
        n_cols = n_cols or len(x_explain.columns)
        res = []
        for individual_sample in tqdm(range(len(x_explain))):
            individual_explanation = {col: col_values for col, col_values in zip(x_explain.columns, shap_values[individual_sample].T)}
            individual_explanation = sorted(individual_explanation.items(), key=operator.itemgetter(1), reverse=True)[:n_cols]
            res.append(dict(individual_explanation))
        return res

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        shap_values = self._explainer.shap_values(x_explain)
        shap_dict = dict(zip(x_explain.columns.to_list(), list(np.mean(abs(shap_values), axis=0).tolist())))
        kept_shap_bar_cols = dict(sorted(shap_dict.items(), key=lambda x: np.abs(x[1]), reverse=True,)[:n_cols])
        return kept_shap_bar_cols
