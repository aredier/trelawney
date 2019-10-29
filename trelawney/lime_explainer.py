import operator
from typing import List, Optional, Dict, Union

import pandas as pd
import numpy as np
import sklearn
from lime import lime_tabular
from tqdm import tqdm

from trelawney.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Lime stands for local interpretable model-agnostic explanations and is a package based on
    `this article <https://www.arxiv.org/abs/1602.04938>`_. Lime will explain a single prediction of you model
    by crechariotsating a local approximation of your model around said prediction.'sphinx.ext.autodoc', 'sphinx.ext.viewcode']

    .. testsetup::

        >>> import pandas as pd
        >>> import numpy as np
        >>> from trelawney.lime_explainer import LimeExplainer
        >>> from sklearn.linear_model import LogisticRegression

    .. doctest::

        >>> X = pd.DataFrame([np.array(range(100)), np.random.normal(size=100).tolist()], index=['real', 'fake']).T
        >>> y = np.array(range(100)) > 50
        >>> # training the base model
        >>> model = LogisticRegression().fit(X, y)
        >>> # creating and fiting the explainer
        >>> explainer = LimeExplainer()
        >>> explainer.fit(model, X, y)
        <trelawney.lime_explainer.LimeExplainer object at ...>
        >>> # explaining observation
        >>> explanation =  explainer.explain_local(pd.DataFrame([[5, 0.1]]))[0]
        >>> abs(explanation['real']) > abs(explanation['fake'])
        True


    """

    def __init__(self, class_names: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None, ):
        super().__init__()
        self._explainer = None
        if class_names is not None and len(class_names) != 2:
            raise NotImplementedError('Trelawney only handles binary classification case for now. PR welcome ;)')
        self.class_names = class_names
        self._output_len = None
        self.categorical_features = categorical_features

    def fit(self, model: sklearn.base.BaseEstimator, x_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            y_train: pd.DataFrame, ):
        x_train = self._get_dataframe_from_mixed_input(x_train)
        super().fit(model, x_train, y_train)
        self._explainer = lime_tabular.LimeTabularExplainer(x_train.values, feature_names=x_train.columns,
                                                            class_names=self.class_names,
                                                            categorical_features=self.categorical_features,
                                                            discretize_continuous=True)
        return self

    def feature_importance(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                           n_cols: Optional[int] = None) -> Dict[str, float]:
        raise NotImplementedError('we are not sure global explaination is mathematically sound for LIME, it is still'
                                  ' debated, refer tp https://github.com/skanderkam/trelawney/issues/10')

    @staticmethod
    def _extract_col_from_explanation(col_explanation):
        is_left_term = len([x for x in col_explanation if x in ['<', '>']]) < 2
        if is_left_term:
            return col_explanation.split()[0]
        return col_explanation.split()[2]

    def explain_local(self, x_explain: Union[pd.Series, pd.DataFrame, np.ndarray],
                      n_cols: Optional[int] = None) -> List[Dict[str, float]]:
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        super().explain_local(x_explain)
        x_explain = self._get_dataframe_from_mixed_input(x_explain)
        n_cols = n_cols or len(x_explain.columns)
        res = []
        for individual_sample in tqdm(x_explain.iterrows()):
            individual_explanation = self._explainer.explain_instance(individual_sample[1],
                                                                      self._model_to_explain.predict_proba,
                                                                      num_features=x_explain.shape[1],
                                                                      top_labels=2)
            individual_explanation = sorted(individual_explanation.as_list(), key=operator.itemgetter(1), reverse=True)
            skewed_individual_explanation = {self._extract_col_from_explanation(col_name): col_importance
                                             for col_name, col_importance in individual_explanation[:n_cols]}
            rest = sum(map(operator.itemgetter(1), individual_explanation)) - sum(skewed_individual_explanation.values())
            skewed_individual_explanation['rest'] = rest
            res.append(skewed_individual_explanation)
        return res
