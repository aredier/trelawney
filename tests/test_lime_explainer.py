import numpy as np
import pandas as pd
import pytest

from xgboost import XGBClassifier

from trelawney.lime_explainer import LimeExplainer


def _do_explainer_test(explainer, data_to_test=None, col_real='real', col_fake='fake'):
    data_to_test = data_to_test if data_to_test is not None else pd.DataFrame([[5, 0.1], [95, -0.5]], columns=[col_real, col_fake])
    explanation = explainer.explain_local(data_to_test)
    assert len(explanation) == data_to_test.shape[0] if len(data_to_test.shape) == 2 else 1
    for single_explanation in explanation:
        assert abs(single_explanation[col_real]) > abs(single_explanation[col_fake])


def test_lime_explainer_single(fake_dataset, fitted_logistic_regression):
    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1]]))
    assert len(explanation) == 1
    single_explanation = explanation[0]
    assert abs(single_explanation['real']) > abs(single_explanation['fake'])


def test_lime_explainer_multiple(fake_dataset, fitted_logistic_regression):
    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    _do_explainer_test(explainer)


def test_lime_explainer_array(fake_dataset, fitted_logistic_regression):
    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_logistic_regression, fake_dataset[0].values, fake_dataset[1])
    _do_explainer_test(explainer, np.array([[5, 0.1], [95, -0.5]]), col_real='feature_0', col_fake='feature_1')


def test_lime_explainer_series(fake_dataset, fitted_logistic_regression):
    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_logistic_regression, *fake_dataset)
    _do_explainer_test(explainer, pd.Series([5, 0.1], index=['real', 'fake']))


def test_lime_xgb(fake_dataset):
    model = XGBClassifier()
    x, y = fake_dataset
    model.fit(x.values, y)

    explainer = LimeExplainer()
    explainer.fit(model, *fake_dataset)
    _do_explainer_test(explainer)


def test_lime_nn(fake_dataset, fitted_neural_network):

    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_neural_network, *fake_dataset)
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1], [95, -0.5]]))
    assert len(explanation) == 2
