"""
module that provides the base explainer class from which all future explainers will inherit
"""
import abc
import operator
from typing import List, Optional, Dict

import sklearn
import pandas as pd
import plotly.graph_objs as go


class BaseExplainer(abc.ABC):
    """
    the base explainer class. this is an abstract class so you will need to define some behaviors when implementing your
    new explainer. In order to do so, override:

    - the `fit` method that defines how (if needed) the explainer should be fited
    - the `feature_importance` method that extracts the relative importance of each feature on a dataset globally
    - the `explain_local` method that extracts the relative impact of each feature on the final decisionfor every sample
      in a dataset
    """

    def __init__(self):
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        prepares the explainer by saving all the information it needs and fitting necessary models

        :param model: the TRAINED model the explainer will need to shed light on
        :param x_train: the dataset the model was trained on originally
        :param y_train: the target the model was trained on originally
        """
        self._model_to_explain = model

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
        og_mvmt = sum(col_importance_dic.values())
        sorted_and_filtered = dict(sorted(
            filter(
                lambda col_and_importance: col_and_importance[0] in cols,
                col_importance_dic.items()
            ),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n_cols])
        sorted_and_filtered['rest'] = sorted_and_filtered.get('rest', 0.) + (og_mvmt - sum(sorted_and_filtered.values()))
        return sorted_and_filtered

    def filtered_feature_importance(self, x_explain: pd.DataFrame, cols: Optional[List[str]],
                                    n_cols: Optional[int] = None) -> Dict[str, float]:
        """same as `feature_importance` but applying a filter first (on the name of the column)"""
        return self._filter_and_limit_dict(self.feature_importance(x_explain), cols, n_cols)

    def graph_feature_importance(self, x_explain: pd.DataFrame, cols: Optional[List[str]] = None, n_cols: Optional[int] = None,
                                 irrelevant_cols: Optional[List[str]] = None):
        cols = cols or x_explain.columns.to_list()
        irrelevant_cols = irrelevant_cols or []
        feature_importance = self.filtered_feature_importance(x_explain, cols, n_cols)
        rest = feature_importance.pop('rest')
        sorted_feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True)
        colors = ['#3B577C' if col not in irrelevant_cols else '#8292AF'
                  for col in map(operator.itemgetter(0), sorted_feature_importance)]
        colors.append('#0077ff')
        plot = go.Bar(x=list(map(operator.itemgetter(0), sorted_feature_importance)) + ['rest'],
                      y=list(map(operator.itemgetter(1), sorted_feature_importance)) + [rest],
                      marker_color=colors)
        return go.Figure(plot)

    @abc.abstractmethod
    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        explains each individual predictions made on x_explain. BEWARE this is usually quite slow on large datasets

        :param x_explain: the samples to explain
        :param n_cols: the number of columns to limit the explanation to
        """
        if not isinstance(x_explain, pd.DataFrame):
            raise TypeError('{} is not supported, please use dataframes'.format(type(x_explain)))

    def explain_filtered_local(self, x_explain: pd.DataFrame, cols: List[str],
                               n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """same as `explain_local` but applying a filter on each explanation on the features"""

        return [
            self._filter_and_limit_dict(sample_importance_dict, cols, n_cols)
            for sample_importance_dict in self.explain_local(x_explain)
        ]

    def graph_local_explanation(self, x_explain: pd.DataFrame, cols: Optional[List[str]] = None,
                                n_cols: Optional[int] = None) -> go.Figure:
        """
        creates a waterfall plotly figure to represent the influance of each feature on the final decision for a single
        prediction of the model.

        You can filter the columns you want to see in your graph and limit the final number of columns you want to see.
        If you choose to do so the filter will be applied first and of those filtered columns at most `n_cols` will be
        kept

        :param x_explain: the example of the model this must be a dataframe with a single ow
        :param cols: the columns to keep if you want to filter (if None - default) all the columns will be kept
        :param n_cols: the number of columns to limit the graph to. (if None - default) all the columns will be kept

        :raises ValueError: if x_explain doesn't have the right shape
        """
        if x_explain.shape[0] != 1:
            raise ValueError('can only explain single observations, if you only have one sample, use reshape(1, -1)')
        cols = cols or x_explain.columns.to_list()
        importance_dict = self.explain_filtered_local(x_explain, cols=cols, n_cols=n_cols)[0]

        output_value = self._model_to_explain.predict_proba(x_explain.values)[0, 1]
        start_value = output_value - sum(importance_dict.values())
        rest = importance_dict.pop('rest')

        sorted_importances = sorted(importance_dict.items(), key=lambda importance: abs(importance[1]), reverse=True)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=['absolute', *['relative' for _ in importance_dict], 'relative', 'absolute'],
            y=[start_value, *map(operator.itemgetter(1), sorted_importances), rest, output_value],
            textposition="outside",
            #     text = ["+60", "+80", "", "-40", "-20", "Total"],
            x=['start_value', *map(operator.itemgetter(0), sorted_importances), 'rest', 'output_value'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title="explanation",
            showlegend=True
        )
        return fig
