from typing import List, Optional, Dict

import pandas as pd
import sklearn
from lime import lime_tabular
from tqdm import tqdm

from trelawney.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        self._explainer = None
        self.class_names = class_names
        self.categorical_features = categorical_features
        self._model_to_explain = None

    def fit(self, model: sklearn.base.BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame, ):

        self._model_to_explain = model
        self._explainer = lime_tabular.LimeTabularExplainer(x_train, feature_names=x_train.columns,
                                                            class_names=self.class_names,
                                                            categorical_features=self.categorical_features,
                                                            discretize_continuous=True)

    def feature_importance(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> Dict[str, float]:
        raise NotImplementedError('we are not sure global explaination is mathematically sound for LIME, it is still'
                                  ' debated, refer tp https://github.com/skanderkam/trelawney/issues/10')

    def explain_local(self, x_explain: pd.DataFrame, n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        n_cols = n_cols or len(x_explain.columns)
        res = []
        for individual_sample in tqdm(x_explain.iterrows()):
            individual_explanation = self._explainer.explain_instance(individual_sample,
                                                                      self._model_to_explain.predict_proba,
                                                                      num_features=n_cols,
                                                                      top_labels=len(self.categorical_features))
            res.append(dict(individual_explanation.as_list()))
        return res
