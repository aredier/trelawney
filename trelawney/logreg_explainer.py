"""
Module that provides the LogRegExplainer class base on the BaseExplainer class
"""
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

class LogRegExplainer(BaseExplainer):
    """
    The LogRegExplainer class is composed of 3 methods:
    - fit: get the right model
    - feature_importance (global interpretation)
    - graph_odds_ratio (visualisation of the ranking of the features, based on their odds ratio)
    """

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        """
        initialize class_names, categorical_features and model_to_explain
        """
        self.class_names = class_names
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, , x_train: pd.DataFrame, y_train: pd.DataFrame): 
        self._model_to_explain = model
        self._feature_names = x_train.columns

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        """
        returns the absolute value (i.e. magnitude) of the coefficient of each feature as a dict.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        res = dict(zip(x_explain.columns, np.abs(self._model_to_explain.coef_[0])))
        return res

    def _compute_odds_ratio(self);
        res = pd.DataFrame('variables': self._feature_names, 
                           'odds_ratio': np.exp(self._model_to_explain.coef_[0])).set_index('variables')
        return res

    def graph_odds_ratio(self, top: int = 10, ascending: bool = False):
        """
        returns a plot of the top k features, based on the magnitude of their odds ratio.
        :top: number of features to plot
        :ascending: order of the ranking of the magnitude of the coefficients
        """
        dataset_to_plot = self._compute_odds_ratio().sort_values(['odds_ratio'], ascending=ascending).head(top)

        fig, ax = plt.subplots(figsize=(10,10))

        sns.barplot(y=dataset_to_plot.index, x=dataset_to_plot['odds_ratio'], ax=ax, palette=sns.color_palette("RdBu", n_colors=7))
        ax.set_title('Top %i variables, based on their odds ratio'%(top))
        plt.show()
        return ax

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        """
        returns local relative importance of features for a specific observation.
        :param x_explain: the dataset to explain on
        :param n_cols: the maximum number of features to return
        """
        raise NotImplementedError('no consensus on which values can explain the path followed by an observation')
