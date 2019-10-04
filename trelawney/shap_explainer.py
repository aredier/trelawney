from typing import List, Optional, Dict

import pandas as pd
import shap as shap
import sklearn
from keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from trelawney.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        self._explainer = None
        self.class_names = class_names
        self.categorical_features = categorical_features
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self._model_to_explain = model
        if self._model_to_explain == LogisticRegression:
            self._explainer = shap.LinearExplainer(self._model_to_explain, data = x_train, feature_dependence="independent")
        elif self._model_to_explain in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier]:
            self._explainer = shap.TreeExplainer(self._model_to_explain, data = x_train)
        elif self._model_to_explain == models.Model:
            self._explainer = shap.DeepExplainer(self._model_to_explain, data = x_train)
        else :
            raise NotImplementedError('For now, this package has only been implemented for LogisticRegression,'
                                      'DecisionTreeClassifier, RandomForestClassifier, XGBClassifier and NN')
"""

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        shap_values = self._explainer.shap_values(x_explain)
        return shap_values

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        pass

    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass
"""
