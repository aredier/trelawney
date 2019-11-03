import pytest
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from trelawney.shap_explainer import ShapExplainer


def _do_explainer_test(explainer, data_to_test=None, col_real='real', col_fake='fake'):
    data_to_test = data_to_test if data_to_test is not None else pd.DataFrame([[5, 0.1], [95, -0.5]], columns=[col_real, col_fake])
    data_to_test = ShapExplainer._get_dataframe_from_mixed_input(data_to_test)
    explanation = explainer.explain_local(data_to_test)
    assert len(explanation) == data_to_test.shape[0] if len(data_to_test.shape) == 2 else 1
    for single_explanation, data_row in zip(explanation, data_to_test.iterrows()):
        assert abs(single_explanation[col_real]) > abs(single_explanation[col_fake])
        assert (single_explanation[col_real] > 0) == (data_row[1][col_real] > 50)


def test_shap_explainer_single(fake_dataset, fitted_logistic_regression):
    explainer = ShapExplainer()
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    explanation = explainer.explain_local(pd.DataFrame([[30, 0.1]], columns=['real', 'fake']))
    assert len(explanation) == 1
    single_explanation = explanation[0]
    assert abs(single_explanation['real']) > abs(single_explanation['fake'])


def test_shap_explainer_multiple(fake_dataset, fitted_logistic_regression):
    explainer = ShapExplainer()
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    _do_explainer_test(explainer)


def test_shap_explainer_series(fake_dataset, fitted_logistic_regression):
    explainer = ShapExplainer()
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    _do_explainer_test(explainer, pd.Series([5, 0.1], index=['real', 'fake']))


def test_shap_explainer_array(fake_dataset, fitted_logistic_regression):
    explainer = ShapExplainer()
    explainer.fit(fitted_logistic_regression, fake_dataset[0].values, fake_dataset[1])
    _do_explainer_test(explainer, np.array([[5, 0.1], [95, -0.5]]), col_real='feature_0', col_fake='feature_1')


def test_shap_xgb(fake_dataset):
    model = XGBClassifier()
    x, y = fake_dataset
    model.fit(x.values, y)

    explainer = ShapExplainer()
    explainer.fit(model, *fake_dataset)
    _do_explainer_test(explainer)


def test_shap_nn(fake_dataset, fitted_neural_network):

    explainer = ShapExplainer()
    explainer.fit(fitted_neural_network, *fake_dataset)
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1], [95, -0.5]], columns=['real', 'fake']))
    assert len(explanation) == 2


def test_shap_global_multiple(fake_dataset, fitted_logistic_regression):
    explainer = ShapExplainer()
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    explanation = explainer.feature_importance(pd.DataFrame([[5, 0.1], [95, -0.5]], columns=['real', 'fake']))
    assert abs(explanation['real']) > abs(explanation['fake'])
