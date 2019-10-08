import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from trelawney.surrogate_explainer import SurrogateExplainer


def test_surogate_explainer_single(fake_dataset, fitted_decision_tree):
    explainer = SurrogateExplainer(DecisionTreeClassifier(max_depth=3))
    explainer.fit(fitted_decision_tree, *fake_dataset)
    explanation = explainer.feature_importance(pd.DataFrame([[30, 0.1]], columns=['real', 'fake']))
    assert abs(explanation['real']) > abs(explanation['fake'])


def test_surogate_explainer_multiple(fake_dataset, fitted_decision_tree):
    explainer = SurrogateExplainer(DecisionTreeClassifier(max_depth=3))
    explainer.fit(fitted_decision_tree, *fake_dataset)
    explanation = explainer.feature_importance(pd.DataFrame([[5, 0.1], [95, -0.5]], columns=['real', 'fake']))
    assert abs(explanation['real']) > abs(explanation['fake'])
