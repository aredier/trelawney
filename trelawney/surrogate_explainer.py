import operator
from typing import List, Optional, Dict, Callable, Tuple, Union

import pandas as pd
import numpy as np
import sklearn

from trelawney.base_explainer import BaseExplainer
import trelawney.tree_explainer
# import trelawney.logreg_explainer


class SurrogateExplainer(BaseExplainer):
    """
    A surrogate model is a substitution model used to explain the initial model. Therefore, substitution models are
    generally simpler than the initial ones. Here, we use single trees and logistic regressions as surrogates.
    """

    def __init__(self, surrogate_model: sklearn.base.BaseEstimator, class_names: Optional[List[str]] = None):
        if type(surrogate_model) not in [sklearn.tree.tree.DecisionTreeClassifier,
                                         sklearn.linear_model.base.LinearRegression]:
            raise NotImplementedError('SurrogateExplainer is only available for single trees (single_tree) and logistic'
                                      'regression (logistic_regression) at the time being.')
        self._surrogate = surrogate_model
        self._explainer = None
        self._x_train = None
        self._model_to_explain = None
        self._adequation_metric = None
        self._class_names = class_names

    def fit(self, model: sklearn.base.BaseEstimator, x_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            y_train: pd.DataFrame):
        x_train = self._get_dataframe_from_mixed_input(x_train)
        self._model_to_explain = model
        self._x_train = x_train.values
        if type(self._surrogate) == sklearn.tree.tree.DecisionTreeClassifier:
            self._surrogate.fit(self._x_train, self._model_to_explain.predict(self._x_train))
            self._adequation_metric = sklearn.metrics.accuracy_score
            self._explainer = trelawney.tree_explainer.TreeExplainer(self._class_names).fit(self._surrogate, x_train, y_train)
        else:
            raise NotImplementedError
            # self._surrogate.fit(x_train, self._model_to_explain.predict_probas(x_train))
            # self._adequation_metric = sklearn.metrics.mean_squared_error
            # self._explainer = trelawney.logreg_explainer.LogRegExplainer().fit(self._surrogate)
        return self

    def adequation_score(self, metric: Union[Callable[[np.ndarray, np.ndarray], float], str] = 'auto', ):
        """
        returns an adequation score between the output of the surrogate and the output of the initial model based on
        the x_train set given.
        """
        if metric != 'auto':
            self._adequation_metric = metric
        return self._adequation_metric(self._model_to_explain.predict(self._x_train),
                                       self._surrogate.predict(self._x_train))

    def feature_importance(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                           n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns a relative importance of each feature globally as a dict.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        return self._explainer.feature_importance(x_explain, n_cols)

    def explain_local(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray], n_cols: Optional[int] = None, ) -> List[Dict[str, float]]:
        """
        returns local relative importance of features for a specific observation.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        return self._explainer.explain_local(x_explain, n_cols)

    def plot_tree(self, out_path: str = './tree_viz'):
        """returns the colored plot of the decision tree and saves an Image in the wd."""
        if type(self._surrogate) != sklearn.tree.tree.DecisionTreeClassifier:
            raise TypeError('plot_tree is only available for single tree surrogate')
        return self._explainer.plot_tree(out_path=out_path)

