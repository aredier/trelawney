"""
Module that provides the TreeExplainer class base on the Baseexplainer class
"""
import os
import tempfile
from typing import Optional, List, Dict

import pandas as pd
import sklearn
from subprocess import call
from sklearn import tree

from trelawney.base_explainer import BaseExplainer


class TreeExplainer(BaseExplainer):
    """
    The TreeExplainer class is composed of 4 methods:
    - fit: get the right model
    - feature_importance (global interpretation)
    - explain_local (local interpretation, WIP)
    - plot_tree (full tree visualisation)
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        initialize class_names, categorical_features and model_to_explain
        """
        self.class_names = class_names
        self._model_to_explain = None
        self._feature_names = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):

        self._model_to_explain = model
        self._feature_names = x_train.columns

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns a relative importance of each feature globally as a dict.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        res = dict(zip(x_explain.columns, self._model_to_explain.feature_importances_))
        return res

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        returns local relative importance of features for a specific observation.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        raise NotImplementedError('no consensus on which values can explain the path followed by an observation')

    def plot_tree(self, out_path: str = './tree_viz.png'):
        """
        creates a png file of the tree saved in out_path

        :param out_path: the path to save the png representation of the tree to
        """
        with tempfile.TemporaryDirectory() as dir_path:
            dot_path = os.path.join(dir_path, 'tree.dot')

            tree.export_graphviz(self._model_to_explain, out_file=dot_path + '.dot', filled=True, rounded=True,
                                 special_characters=True, feature_names=self._feature_names,
                                 class_names=self.class_names)
            call(['dot', '-Tpng', dot_path, '-o', out_path, '-Gdpi=600'])
