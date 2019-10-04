"""
Module that provides the TreeExplainer class base on the Baseexplainer class
"""
import pandas as pd
from subprocess import call
from IPython.display import Image
import sklearn
from sklearn import tree

class TreeExplainer(BaseExplainer):
    """
    The TreeExplainer class is composed of 4 methods:
    - fit: get the right model
    - feature_importance (global interpretation)
    - explain_local (local interpretation, WIP)
    - plot_tree (full tree visualisation)
    """

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        """
        initialize class_names, categorical_features and model_to_explain
        """
        self.class_names = class_names
        self.categorical_features = categorical_features
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator):
        """
        _model_to_explain attribute gets the right model to explain
        """
        self._model_to_explain = model

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
    
    def plot_tree(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None, out_file: str = 'tree_viz') -> Image:
        """
        returns the colored plot of the decision tree and saves an Image in the wd.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        tree.export_graphviz(self._model_to_explain, out_file = out_file + '.dot', filled=True, rounded=True,
                special_characters=True, feature_names = x_explain.columns, class_names= self.class_names)
        call(['dot', '-Tpng', out_file + '.dot', '-o', out_file + '.png', '-Gdpi=600'])
        return Image(filename = out_file + '.png')
