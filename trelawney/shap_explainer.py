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
            self._explainer = shap.LinearExplainer(self._model_to_explain, data=x_train, feature_dependence="independent")
        elif self._model_to_explain in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier]:
            self._explainer = shap.TreeExplainer(self._model_to_explain, data=x_train)
        elif self._model_to_explain == models.Model:
            self._explainer = shap.DeepExplainer(self._model_to_explain, data=x_train)
        else:
            raise NotImplementedError('For now, this package has only been implemented for LogisticRegression,'
                                      'DecisionTreeClassifier, RandomForestClassifier, XGBClassifier and NN')

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        shap_values = self._explainer.shap_values()
        res = []
        for individual_sample in tqdm(x_explain.iterrows()):
            individual_explanation = {col: np.mean(np.abs(col_values)) for col, col_values in zip(X_test.columns, shap_values_XGB_test[0].T)}
            individual_explanation = sorted(individual_explanation.items(), key=operator.itemgetter(1),reverse=True)
            res.append(dict(individual_explanation.as_list()))
        return [shap_values, res]

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        shap_dict = dict(zip(X_test.columns.to_list(), list(np.mean(abs(shap_values_XGB_test), axis = 0).tolist())))
        kept_shap_bar_cols = dict(sorted(shap_dict.items(), key=lambda x: np.abs(x[1]), reverse=True,)[:N_COMPS])


"""
    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass
"""
