from typing import List, Optional, Dict

import pandas as pd
import sklearn
from lime import lime_tabular
from tqdm import tqdm

from trelawney.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Lime stands for local interpretable model-agnostic explanations and is a package based on
    `this article <https://www.arxiv.org/abs/1602.04938>`_. Lime will explain a single prediction of you model
    by creating a local approximation of your model around said prediction.'sphinx.ext.autodoc', 'sphinx.ext.viewcode']
    """

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        self._explainer = None
        if len(class_names) != 2:
            raise NotImplementedError('Trelawney only handles binary classification case for now. PR welcome ;)')
        self.class_names = class_names
        self._output_len = None
        self.categorical_features = categorical_features
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame, ):
        self._model_to_explain = model
        self._explainer = lime_tabular.LimeTabularExplainer(x_train.values, feature_names=x_train.columns,
                                                            class_names=self.class_names,
                                                            categorical_features=self.categorical_features,
                                                            discretize_continuous=True)

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        raise NotImplementedError('we are not sure global explaination is mathematically sound for LIME, it is still'
                                  ' debated, refer tp https://github.com/skanderkam/trelawney/issues/10')

    @staticmethod
    def _extract_col_from_explanation(col_explanation):
        is_left_term = len([x for x in col_explanation if x in ['<', '>']]) < 2
        if is_left_term:
            return col_explanation.split()[0]
        return col_explanation.split()[2]

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        n_cols = n_cols or len(x_explain.columns)
        res = []
        for individual_sample in tqdm(x_explain.iterrows()):
            individual_explanation = self._explainer.explain_instance(individual_sample[1],
                                                                      self._model_to_explain.predict_proba,
                                                                      num_features=n_cols,
                                                                      top_labels=2)
            res.append({self._extract_col_from_explanation(col_explanation): col_value
                        for col_explanation, col_value in individual_explanation.as_list()})
        return res
