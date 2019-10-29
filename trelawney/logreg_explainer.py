"""
Module that provides the LogRegExplainer class base on the BaseExplainer class
"""
import operator
from typing import Optional, List, Dict, Union

import pandas as pd
import numpy as np
import sklearn
import plotly.graph_objs as go
from trelawney.base_explainer import BaseExplainer
from trelawney.colors import BLUE, GREY


class LogRegExplainer(BaseExplainer):
    """
    The LogRegExplainer class is composed of 3 methods:
    - fit: get the right model
    - feature_importance (global interpretation)
    - graph_odds_ratio (visualisation of the ranking of the features, based on their odds ratio)
    """

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        """
        initialize class_names, categorical_features and model_to_explain
        """
        self.class_names = class_names
        self._model_to_explain = None
        self._feature_names = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            y_train: pd.DataFrame):
        x_train = self._get_dataframe_from_mixed_input(x_train)
        self._model_to_explain = model
        self._feature_names = x_train.columns

    def feature_importance(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                           n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns the absolute value (i.e. magnitude) of the coefficient of each feature as a dict.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        res = dict(zip(x_explain.columns, np.abs(self._model_to_explain.coef_[0])))
        return res

    def _compute_odds_ratio(self):
        res = pd.DataFrame({'variables': self._feature_names,
                            'odds_ratio': np.exp(self._model_to_explain.coef_[0])}).set_index('variables')
        return res

    def graph_odds_ratio(self, n_cols: Optional[int] = 10, ascending: bool = False,
                         irrelevant_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        returns a plot of the top k features, based on the magnitude of their odds ratio.
        :n_cols: number of features to plot
        :ascending: order of the ranking of the magnitude of the coefficients
        """
        irrelevant_cols = irrelevant_cols or []
        all_odds = self._compute_odds_ratio()
        odds_ratio_dict = dict(zip(all_odds.index, all_odds.odds_ratio))
        dataset_to_plot = sorted(odds_ratio_dict.items(), key=lambda x: abs(x[1]), reverse=not ascending)[:n_cols]

        colors = [BLUE if col not in irrelevant_cols else GREY
                  for col in map(operator.itemgetter(0), dataset_to_plot)]
        plot = go.Bar(x=list(map(operator.itemgetter(0), dataset_to_plot)),
                      y=list(map(operator.itemgetter(1), dataset_to_plot)),
                      marker_color=colors)
        fig = go.Figure(plot)
        fig.update_layout(
            title='Top {} variables, based on their odds ratio'.format(n_cols),
            showlegend=True
        )
        return fig

    def explain_local(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                      n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        returns local relative importance of features for a specific observation.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        raise NotImplementedError('no consensus on which values can explain the path followed by an observation')
