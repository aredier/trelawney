import operator
from typing import Optional

import pandas as pd
import sklearn
import numpy as np

from trelawney.base_explainer import BaseExplainer


class FakeExplainer(BaseExplainer):

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        return dict(sorted(
            ((col, np.mean(np.abs(x_explain.loc[:, col]))) for col in x_explain.columns),
            key=operator.itemgetter(1),
            reverse=True
        )[:n_cols])

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        return [
            dict(sorted(sample_explanation.items(), key=operator.itemgetter(1), reverse=True)[:n_cols])
            for sample_explanation in x_explain.abs().to_dict(orient='records')
            ]


def test_explainer_basic():

    explainer = FakeExplainer()
    assert explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])) == {
        'var_1': 5., 'var_2': 2.5
    }
    assert explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1) == {
        'var_1': 5.
    }

    assert explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])) == [
        {'var_1': 10., 'var_2': 0.},
        {'var_2': 5., 'var_1': 0.}
    ]
    assert explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1) == [
        {'var_1': 10.},
        {'var_2': 5.}
    ]


def test_explainer_filter():

    explainer = FakeExplainer()
    assert explainer.filtered_feature_importance(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3']) == {'var_1': 5., 'var_3': 3.5}

    assert explainer.filtered_feature_importance(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        n_cols=1, cols=['var_1', 'var_3']) == {'var_1': 5.}

    assert explainer.explain_filtered_local(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3']) == [
        {'var_1': 10., 'var_3': 4.},
        {'var_3': 3., 'var_1': 0.}
    ]
    assert explainer.explain_filtered_local(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3'], n_cols=1) == [
               {'var_1': 10.},
               {'var_3': 3.}
           ]



