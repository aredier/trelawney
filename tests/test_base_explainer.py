import operator
from typing import Optional, Dict, Tuple, List

import pandas as pd
import sklearn
import numpy as np

from trelawney.base_explainer import BaseExplainer


class FakeExplainer(BaseExplainer):

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    @staticmethod
    def _regularize(importance_dict: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        total = sum(map(operator.itemgetter(1), importance_dict))
        return [
            (key, -(2 * (i % 2) - 1) * (value / total))
            for i, (key, value) in enumerate(importance_dict)
        ]

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        return dict(self._regularize(sorted(
            ((col, np.mean(np.abs(x_explain.loc[:, col]))) for col in x_explain.columns),
            key=operator.itemgetter(1),
            reverse=True
        ))[:n_cols])

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        return [
            dict(self._regularize(
                sorted(sample_explanation.items(), key=operator.itemgetter(1), reverse=True)
            )[:n_cols])
            for sample_explanation in x_explain.abs().to_dict(orient='records')
            ]


def test_explainer_basic():

    explainer = FakeExplainer()
    assert explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])) == {
        'var_1': 5 / 7.5, 'var_2': -2.5 / 7.5
    }
    assert explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1) == {
        'var_1': 5. / 7.5
    }

    assert explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])) == [
        {'var_1': 1., 'var_2': 0.},
        {'var_2': 1., 'var_1': 0.}
    ]
    assert explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1) == [
        {'var_1': 1.},
        {'var_2': 1.}
    ]


def test_explainer_filter():

    explainer = FakeExplainer()
    assert explainer.filtered_feature_importance(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3']) == {'var_1': 10 / 22, 'var_3': -7 / 22}

    assert explainer.filtered_feature_importance(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        n_cols=1, cols=['var_1', 'var_3']) == {'var_1': 10 / 22}

    assert explainer.explain_filtered_local(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3']) == [
        {'var_1': 10 / 14, 'var_3': -4 / 14},
        {'var_3': -3 / 8, 'var_1': 0.}
    ]
    assert explainer.explain_filtered_local(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
        cols=['var_1', 'var_3'], n_cols=1) == [
               {'var_1': 10 / 14},
               {'var_3': -3 / 8}
           ]



