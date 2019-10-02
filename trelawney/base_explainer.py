import abc
import operator
from typing import List, Optional, Dict

import sklearn
import pandas as pd


class BaseExplainer(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    @abc.abstractmethod
    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        pass

    @staticmethod
    def _filter_and_limit_dict(col_importance_dic: Dict[str, float], cols: List[str], n_cols: int):
        return dict(sorted(
            filter(
                lambda col_and_importance: col_and_importance[0] in cols,
                col_importance_dic.items()
            ),
            key=operator.itemgetter(1),
            reverse=True
        )[:n_cols])

    def filtered_feature_importance(self, x_explain: pd.DataFrame, cols: List[str],
                                    n_cols: Optional[int] = None) -> Dict[str, float]:
        return self._filter_and_limit_dict(self.feature_importance(x_explain), cols, n_cols)

    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        raise NotImplementedError('graphing functionalities not implemented yet')

    @abc.abstractmethod
    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        pass

    def explain_filtered_local(self, x_explain: pd.DataFrame, cols: List[str],
                               n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        return [
            self._filter_and_limit_dict(sample_importance_dict, cols, n_cols)
            for sample_importance_dict in self.explain_local(x_explain)
        ]

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        raise NotImplementedError('graphing functionalities not implemented yet')
