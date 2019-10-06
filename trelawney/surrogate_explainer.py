import operator
from typing import List, Optional, Dict

import pandas as pd
import sklearn
from lime import lime_tabular
from tqdm import tqdm

from trelawney.base_explainer import BaseExplainer
import trelawney.tree_explainer
import trelawney.logreg_explainer


class SurrogateExplainer(BaseExplainer):
    """
    A surrogate model is a substitution model used to explain the initial model. Therefore, substitution models are
    generally simpler than the initial ones. Here, we use single trees and logistic regressions as surrogates.
    """

    def __init__(self, surrogate_type: Optional[str] = 'single_tree', max_depth: Optional[int] = 5, ):
        if surrogate_type in ['single_tree', 'logistic_regression']:
            self._explainer = None
            self._x_train = None
            self.surrogate_type = surrogate_type
            self.max_depth = max_depth
            self._model_to_explain = None
        else:
            raise NotImplementedError('SurrogateExplainer is only available for single trees (single_tree) and logistic'
                                      'regression (logistic_regression) at the time being.')

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, ):
        self._model_to_explain = model
        self._x_train = x_train
        if self.surrogate_type == 'single_tree':
            self._explainer = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)
        else:
            self._explainer = sklearn.linear_model.LinearRegression()
        self._explainer.fit(x_train, self._model_to_explain.predict(x_train))

    def adequation_score(self):
        """
        returns an adequation score between the output of the surrogate and the output of the initial model based on
        the x_train set given.
        """
        if self.surrogate_type == 'single_tree':
            return sklearn.metrics.accuracy_score(self._model_to_explain.predict(self._x_train),
                                                  self._explainer(self._x_train))
        else:
            return sklearn.metrics.mean_squared_error(self._model_to_explain.predict(self._x_train),
                                                      self._explainer(self._x_train))

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns a relative importance of each feature globally as a dict.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        if self.surrogate_type == 'single_tree':
            return (trelawney.tree_explainer
                    .TreeExplainer()
                    .fit(self._explainer)
                    .feature_importance(x_explain, n_cols))
        else:
            return (trelawney.logreg_explainer
                    .LogRegExplainer()
                    .fit(self._explainer)
                    .feature_importance(x_explain, n_cols))

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        returns local relative importance of features for a specific observation.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        if self.surrogate_type == 'single_tree':
            raise NotImplementedError('no consensus on which values can explain the path followed by an observation')
        else:
            raise NotImplementedError('waiting for feature to be implemented in LogRegExplainer')

    def plot_tree(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None, out_file: str = 'tree_viz') -> Image:
        """
        returns the colored plot of the decision tree and saves an Image in the wd.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        :param out_file: name of the generated plot
        """
        if self.surrogate_type == 'single_tree':
            return (trelawney.tree_explainer
                    .TreeExplainer()
                    .fit(self._explainer)
                    .plot_tree(x_explain, n_cols=n_cols, out_file=out_file))
        else:
            raise UnappropriateSurrogateType('plot_tree is only available for single tree surrogate')
