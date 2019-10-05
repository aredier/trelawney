from typing import List, Optional, Dict

import pandas as pd
import sklearn
from lime import lime_tabular
from tqdm import tqdm

from trelawney.base_explainer import BaseExplainer


class SurrogateExplainer(BaseExplainer):
    """
    A surrogate model is a substitution model used to explain the initial model. Therefore, substitution models are
    generally simpler than the initial ones. Here, we use single trees and logistic regressions as surrogates.
    """

    def __init__(self, surrogate_type: str = 'single_tree', max_depth: int = 5, ):
        self._explainer = None
        self.max_depth = max_depth
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, ):
        self._model_to_explain = model
        self._explainer = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)
        self._explainer.fit(x_train, self._model_to_explain.predict(x_train))
