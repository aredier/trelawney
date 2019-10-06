"""
Module that provides the LogRegExplainer class base on the BaseExplainer class
"""
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

class TreeExplainer(BaseExplainer):
    """
    The LogRegExplainer class is composed of 4 methods:
    - fit: get the right model and preprocess its output
    - show_feature_importance (global interpretation)
    - plot_feature_coefficient (visualisation of the ranking of the features, based on their coefficients)
    - plot_odds_ratio (visualisation of the ranking of the features, based on their odds ratio)
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
        preprocessing results of the model
        """
        self._model_to_explain = model
        self.coefficients = pd.DataFrame({'variable': X.columns,
                                          'coefficients': final_model.coef_[0].round(3),
                                          'abs_coefficients': np.abs(final_model.coef_[0]).round(3),
                                          'odds_ratio': np.exp(final_model.coef_[0]).round(3)}).set_index(['variable'])

    def show_feature_importance(self) -> pd.DataFrame():
        """
        returns a dataframe showing the coefficient, absolute coefficient and odds ratio of every variable.
        :param n_cols: the maximum number of features to return
        """
        return self.coefficients

    def plot_feature_coefficient(self, top: int = 10, ascending: bool = False):
        """
        returns a plot of the top k features, based on the magnitude of their coefficients.
        :top: number of features to plot
        :ascending: order of the ranking of the magnitude of the coefficients
        """
        dataset_to_plot = self.coefficients.sort_values(['abs_coefficients'], ascending=ascending).head(top)

        fig, ax = plt.subplots(figsize=(10,10))

        sns.barplot(y=dataset_to_plot.index, x=dataset_to_plot['coefficients'], ax=ax, palette=sns.color_palette("RdBu", n_colors=7))
        ax.set_title('Top %i variables, based on the magnitude of their coefficients'%(top))
        plt.show()
        return ax

    def plot_odds_ratio(self, top: int = 10, ascending: bool = False):
        """
        returns a plot of the top k features, based on the magnitude of their odds ratio.
        :top: number of features to plot
        :ascending: order of the ranking of the magnitude of the coefficients
        """
        dataset_to_plot = self.coefficients.sort_values(['odds_ratio'], ascending=ascending).head(top)

        fig, ax = plt.subplots(figsize=(10,10))

        sns.barplot(y=dataset_to_plot.index, x=dataset_to_plot['odds_ratio'], ax=ax, palette=sns.color_palette("RdBu", n_colors=7))
        ax.set_title('Top %i variables, based on their odds ratio'%(top))
        plt.show()
        return ax