"""
module that provides the base explainer class from which all future explainers will inherit
"""
import abc
import operator
from typing import List, Optional, Dict

import sklearn
import pandas as pd


class BaseExplainer(abc.ABC):
    """
    the base explainer class. this is an abstract class so you will need to define some behaviors when implementing your
    new explainer. In order to do so, override:

    - the `fit` method that defines how (if needed) the explainer should be fited
    - the `feature_importance` method that extracts the relative importance of each feature on a dataset globally
    - the `explain_local` method that extracts the relative impact of each feature on the final decisionfor every sample
      in a dataset
    """

    @abc.abstractmethod
    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        fits the explainer if needed

        :param model: the TRAINED model the explainer will need to shed light on
        :param x_train: the dataset the model was trained on originally
        :param y_train: the target the model was trained on originally
        """
        pass

    @abc.abstractmethod
    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns a relative importance of each feature on the predictions of the model (the explainer was fitted on) for
        `x_explain` globally. The output will be a dict with the importance for each column/feature in `x_explain`
        (limited to n_cols)

        if some importance are negative, this means they are negatively correlated with the output and absolute value
        represents the relative importance

        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return (ordered by importance)
        """
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
        """same as `feature_importance` but applying a filter first (on the name of the column)"""

        return self._filter_and_limit_dict(self.feature_importance(x_explain), cols, n_cols)

    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        raise NotImplementedError('graphing functionalities not implemented yet')

    @abc.abstractmethod
    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        explains each individual predictions made on x_explain. BEWARE this is usually quite slow on large datasets

        :param x_explain: the samples to explain
        :param n_cols: the number of columns to limit the explanation to
        """
        pass

    def explain_filtered_local(self, x_explain: pd.DataFrame, cols: List[str],
                               n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """same as `explain_local` but applying a filter on each explanation on the features"""
        
        return [
            self._filter_and_limit_dict(sample_importance_dict, cols, n_cols)
            for sample_importance_dict in self.explain_local(x_explain)
        ]

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: List[str], n_cols: Optional[int] = None):
        raise NotImplementedError('graphing functionalities not implemented yet')
