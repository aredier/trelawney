import pytest
import pandas as pd
from xgboost import XGBClassifier

from trelawney.shap_explainer import ShapExplainer


def _do_explainer_test(explainer):
    explanation = explainer.explain_local(pd.DataFrame([[5, 0.1], [95, -0.5]], columns=['real', 'fake']))
    assert len(explanation) == 2
    for single_explanation in explanation:
        assert abs(single_explanation['real']) > abs(single_explanation['fake'])

        # we need positive values as this feature is globally positivly corelated with the target and SHAP claims
        # global consistency of it's explanations
        # assert single_explanation['real'] > 0


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


def test_shap_xgb(fake_dataset):
    model = XGBClassifier()
    x, y = fake_dataset
    model.fit(x.values, y)

    explainer = ShapExplainer()
    explainer.fit(model, *fake_dataset)
    with pytest.raises(TypeError):
        explainer.explain_local(x.values)
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
