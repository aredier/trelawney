from typing import List, Optional, Dict, Union

import pandas as pd
import shap as shap
import sklearn
import numpy as np
import operator
import keras.models
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble.forest import ForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree.tree import BaseDecisionTree
from tqdm import tqdm
from xgboost import XGBClassifier

from trelawney.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):
    """
    SHAP is a `package <https://github.com/slundberg/shap>` to estimate Shapley values on an individual prediction.

    Shapley come from sport analitics by trying to devide the outcome of a match betweeen the players of team by
    comparing the estimated outcome of the match with all the combinaton of players. For model interpretation, we
    consider that all the features are the players and the probability of the classifier the outcome of the game
    """

    def __init__(self):
        super().__init__()
        self._explainer = None

    def _find_right_explainer(self, x_train):
        if isinstance(self._model_to_explain, LogisticRegression):
            return shap.LinearExplainer(self._model_to_explain, data=x_train.values)
        if isinstance(self._model_to_explain, (BaseDecisionTree, ForestClassifier, XGBClassifier)):
            return shap.TreeExplainer(self._model_to_explain)
        if isinstance(self._model_to_explain, KerasClassifier):
            return shap.DeepExplainer(self._model_to_explain.model, data=x_train.values)
        return shap.KernelExplainer(self._model_to_explain, data=x_train.values)

    def fit(self, model: sklearn.base.BaseEstimator, x_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            y_train: pd.DataFrame):
        x_train = self._get_dataframe_from_mixed_input(x_train)
        super().fit(model, x_train, y_train)
        self._explainer = self._find_right_explainer(x_train)
        return self

    def _get_shap_values(self, x_explain):
        shap_values = self._explainer.shap_values(x_explain.values)
        if isinstance(shap_values, list):
            # if we have a list this means there is one output per class, we take the positive class
            # TODO class should be a parameter
            return shap_values[1]
        return shap_values

    def explain_local(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                      n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        super().explain_local(x_explain)
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        shap_values = self._get_shap_values(x_explain)
        n_cols = n_cols or len(x_explain.columns)
        res = []
        for individual_sample in tqdm(range(len(x_explain))):

            # creating the contributions of all the features to a prediction
            individual_explanation = list(zip(x_explain.columns, shap_values[individual_sample].T))

            # limiting to n_cols
            total_mvmt = sum(map(operator.itemgetter(1), individual_explanation))
            skewed_individual_explanation = sorted(individual_explanation, key=operator.itemgetter(1),
                                                   reverse=True)[:n_cols]
            skewed_individual_explanation.append(
                ('rest', total_mvmt - sum(map(operator.itemgetter(1), skewed_individual_explanation)))
            )
            res.append(dict(skewed_individual_explanation))
        return res

    def feature_importance(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                           n_cols: Optional[int] = None) -> Dict[str, float]:
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        shap_values = self._get_shap_values(x_explain)
        shap_dict = dict(zip(x_explain.columns.to_list(), np.mean(np.abs(shap_values), axis=0).reshape(-1).tolist()))
        kept_shap_bar_cols = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True,)[:n_cols])
        return kept_shap_bar_cols
