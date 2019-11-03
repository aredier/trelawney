import operator
from typing import Optional, Dict, Tuple, List, Union

import pandas as pd
import pytest
import sklearn
import numpy as np
import plotly.graph_objs as go

from trelawney.base_explainer import BaseExplainer


class FakeExplainer(BaseExplainer):

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        return super().fit(model, x_train, y_train)

    @staticmethod
    def _regularize(importance_dict: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        total = sum(map(operator.itemgetter(1), importance_dict))
        return [
            (key, -(2 * (i % 2) - 1) * (value / total))
            for i, (key, value) in enumerate(importance_dict)
        ]

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        importance = self._regularize(sorted(
            ((col, np.mean(np.abs(x_explain.loc[:, col]))) for col in x_explain.columns),
            key=operator.itemgetter(1),
            reverse=True
        ))
        total_mvmt = sum(map(operator.itemgetter(1), importance))
        res = dict(importance[:n_cols])
        res['rest'] = total_mvmt - sum(res.values())
        return res

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None):
        res = []
        for sample_explanation in x_explain.abs().to_dict(orient='records'):
            importance = self._regularize(sorted(sample_explanation.items(), key=operator.itemgetter(1), reverse=True))

            total_mvmt = sum(map(operator.itemgetter(1), importance))
            res_ind = dict(importance[:n_cols])
            res_ind['rest'] = total_mvmt - sum(res_ind.values())
            res.append(res_ind)

        return res


def _float_error_resilient_compare(left: Union[List[Dict], Dict], right: Union[List[Dict], Dict]):
    assert len(left) == len(right)
    if isinstance(left, list):
        return [_float_error_resilient_compare(ind_left, ind_right) for ind_left, ind_right in zip(left, right)]
    for key, value in left.items():
        assert key in right
        assert abs(value - right[key]) < 0.0001


def test_explainer_basic():

    explainer = FakeExplainer()
    _float_error_resilient_compare(
        explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])),
        {'var_1': 5 / 7.5, 'var_2': -2.5 / 7.5, 'rest': 0.}
    )
    _float_error_resilient_compare(
        explainer.feature_importance(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1),
        {'var_1': 5. / 7.5, 'rest':  -2.5 / 7.5}
    )

    _float_error_resilient_compare(
        explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2'])),
        [{'var_1': 1., 'var_2': 0., 'rest': 0.}, {'var_2': 1., 'var_1': 0., 'rest': 0.}]
    )

    _float_error_resilient_compare(
        explainer.explain_local(pd.DataFrame([[10, 0], [0, -5]], columns=['var_1', 'var_2']), n_cols=1),
        [{'var_1': 1., 'rest': 0.},{'var_2': 1, 'rest': 0.}]
    )


def test_explainer_filter():

    explainer = FakeExplainer()
    _float_error_resilient_compare(
        explainer.filtered_feature_importance(pd.DataFrame(
            [[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']), cols=['var_1', 'var_3']
        ),
        {'var_1': 10 / 22, 'var_3': -7 / 22, 'rest': 5 / 22}
    )

    _float_error_resilient_compare(
        explainer.filtered_feature_importance(
            pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']), n_cols=1,
            cols=['var_1', 'var_3']
        ),
        {'var_1': 10 / 22, 'rest': -2 / 22}
    )

    _float_error_resilient_compare(
        explainer.explain_filtered_local(
            pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']), cols=['var_1', 'var_3']
        ),
        [{'var_1': 10 / 14, 'var_3': -4 / 14, 'rest': 0.}, {'var_3': -3 / 8, 'var_1': 0., 'rest': 5 / 8}]
    )

    _float_error_resilient_compare(
        explainer.explain_filtered_local(
            pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']),
            cols=['var_1', 'var_3'], n_cols=1
        ),
        [{'var_1': 10 / 14, 'rest': -4 / 14}, {'var_3': -3 / 8, 'rest': 5 / 8}]
    )


def test_local_graph(FakeClassifier, fake_dataset):

    model = FakeClassifier()
    explainer = FakeExplainer()
    explainer.fit(model, *fake_dataset)

    with pytest.raises(ValueError):
        _ = explainer.graph_local_explanation(pd.DataFrame([[10, 30], [1, 2]], columns=['var_1', 'var_2']))

    graph = explainer.graph_local_explanation(pd.DataFrame([[10, 30]], columns=['var_1', 'var_2']))

    assert len(graph.data) == 1
    assert isinstance(graph.data[0], go.Waterfall)
    waterfall = graph.data[0]
    assert waterfall.x == ('start_value', 'var_2', 'var_1', 'rest', 'output_value')
    assert waterfall.y == (0., .75, -0.25, 0., 0.5)

    graph = explainer.graph_local_explanation(pd.DataFrame([[10, 30]], columns=['var_1', 'var_2']), n_cols=1)

    assert len(graph.data) == 1
    assert isinstance(graph.data[0], go.Waterfall)
    waterfall = graph.data[0]
    assert waterfall.x == ('start_value', 'var_2', 'rest', 'output_value')
    assert waterfall.y == (0., .75, -0.25, 0.5)


def test_local_graph_hover_text(FakeClassifier, fake_dataset):

    model = FakeClassifier()
    explainer = FakeExplainer()
    explainer.fit(model, *fake_dataset)

    graph = explainer.graph_local_explanation(pd.DataFrame([[10, 30]], columns=['var_1', 'var_2']),
                                              info_values=pd.DataFrame([['foo', 3]], columns=['var_1', 'var_2']))

    assert len(graph.data) == 1
    assert isinstance(graph.data[0], go.Waterfall)
    waterfall = graph.data[0]
    assert waterfall.hovertext == ('start value', 'var_2 = 3', 'var_1 = foo', 'rest', 'output_value = 0.5')

    graph = explainer.graph_local_explanation(pd.DataFrame([[10, 30]], columns=['var_1', 'var_2']), n_cols=1)


def test_feature_importance_graph(FakeClassifier, fake_dataset):
    model = FakeClassifier()
    explainer = FakeExplainer()
    explainer.fit(model, *fake_dataset)
    graph = explainer.graph_feature_importance(
        pd.DataFrame([[10, 0, 4], [0, -5, 3]], columns=['var_1', 'var_2', 'var_3']), n_cols=1, cols=['var_1', 'var_3']
    )
    assert len(graph.data) == 1
    assert isinstance(graph.data[0], go.Bar)
    bar_graph = graph.data[0]
    assert bar_graph.x == ('var_1', 'rest')
    assert abs(bar_graph.y[0] - 10 / 22) < 0.0001
    assert abs(bar_graph.y[1] + 2 / 22) < 0.0001


def test_mixed_imputs():
    explainer = FakeExplainer()

    assert all(explainer._get_dataframe_from_mixed_input(
        pd.DataFrame([[1, 2, 3]], columns=['foo', 'bar', 'baz'])
    ) == pd.DataFrame([[1, 2, 3]], columns=['foo', 'bar', 'baz']))

    assert all(explainer._get_dataframe_from_mixed_input(
        pd.Series([1, 2, 3], index=['foo', 'bar', 'baz'])
    ) == pd.DataFrame([[1, 2, 3]], columns=['foo', 'bar', 'baz']))

    assert all(explainer._get_dataframe_from_mixed_input(
        np.array([1, 2, 3])
    ) == pd.DataFrame([[1, 2, 3]], columns=['feature_0', 'feature_1', 'feature_2']))

    assert all(explainer._get_dataframe_from_mixed_input(
        np.array([[1, 2, 3]])
    ) == pd.DataFrame([[1, 2, 3]], columns=['feature_0', 'feature_1', 'feature_2']))
