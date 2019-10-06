import pandas as pd
import pytest

from xgboost import XGBClassifier

from trelawney.lime_explainer import LimeExplainer


def _do_explainer_test(explainer):
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1], [95, -0.5]]))
    assert len(explanation) == 2
    for single_explanation in explanation:
        assert abs(single_explanation['real']) > abs(single_explanation['fake'])


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


def test_lime_xgb(fake_dataset):
    model = XGBClassifier()
    x, y = fake_dataset
    model.fit(x.values, y)

    explainer = LimeExplainer()
    explainer.fit(model, *fake_dataset)
    with pytest.raises(TypeError):
        explainer.explain_local(x.values)
    _do_explainer_test(explainer)


def test_lime_nn(fake_dataset, fitted_neural_network):

    explainer = LimeExplainer(class_names=['false', 'true'])
    explainer.fit(fitted_neural_network, *fake_dataset)
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1], [95, -0.5]]))
    assert len(explanation) == 2
    for single_explanation in explanation:
        assert abs(single_explanation['real']) > abs(single_explanation['fake'])
