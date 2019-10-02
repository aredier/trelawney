import abc
from typing import List, Optional

import sklearn
import pandas as pd


class BaseExplainer(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    @abc.abstractmethod
    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        pass

    def filtered_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass

    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass

    @abc.abstractmethod
    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        pass

    def explain_filtered_local(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        pass
