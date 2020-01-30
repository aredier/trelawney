=========
trelawney
=========


.. image:: https://img.shields.io/pypi/v/trelawney.svg
        :target: https://pypi.python.org/pypi/trelawney

.. image:: https://img.shields.io/travis/aredier/trelawney.svg
        :target: https://travis-ci.org/aredier/trelawney

.. image:: https://readthedocs.org/projects/trelawney/badge/?version=latest
        :target: https://trelawney.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/skanderkam/trelawney
        :alt: MIT License



Trelawney is a general interpretability package that aims at providing a common api to use most of the modern
interpretability methods to shed light on sklearn compatible models (support for Keras and XGBoost are tested).

Trelawney will try to provide you with two kind of explanation when possible:

- global explanation of the model that highlights the most importance features the model uses to make its
  predictions globally
- local explanation of the model that will try to shed light on why a specific model made a specific prediction

The Trelawney package is build around:

- some model specific explainers that use the inner workings of some types of models to explain them:
   - `LogRegExplainer` that uses the weights of the your logistic regression to produce global and local explanations of
     your model
   - `TreeExplainer` that uses the path of your tree (single tree model only) to produce explanations of the model

- Some model agnostic explainers that should work with all models:
   - `LimeExplainer` that uses the Lime_ package to create local explanations only (the local nature of Lime prohibits
     it from generating global explanations of a model
   - `ShapExplainer` that uses the SHAP_ package to create local and global explanations of your model
   - `SurrogateExplainer` that creates a general surogate of your model (fitted on the output of your model) using an
     explainable model (`DecisionTreeClassifier`,`LogisticRegression` for now). The explainer will then use the
     internals of the surrogate model to explain your black box model as well as informing you on how well the surrogate
     model explains the black box one

Quick Tutorial (30s to Trelawney):
----------------------------------

Here is an example of how to use a Trelawney explainer

>>> model = LogisticRegression().fit(X, y)
>>> # creating and fiting the explainer
>>> explainer = ShapExplainer()
>>> explainer.fit(model, X, y)
>>> # explaining observation
>>> explanation =  explainer.explain_local(X_expain)
[
    {'var_1': 0.1, 'var_2': -0.07, ...},
    ...
    {'var_1': 0.23, 'var_2': -0.15, ...} ,
]
>>> explanation =  explainer.graph_local_explanation(X_expain.iloc[:1, :])

.. image:: http://drive.google.com/uc?export=view&id=1a1kdH8mjGdKiiF_JHR56L2-JeaRStwr2
   :width: 400
   :alt: Local Explanation Graph

>>> explanation =  explainer.feature_importance(X_expain)
{'var_1': 0.5, 'var_2': 0.2, ...} ,
>>> explanation =  explainer.graph_feature_importance(X_expain)


.. image:: http://drive.google.com/uc?export=view&id=1R2NFEU0bcZYpeiFsLZDKYfPkjHz-cHJ_
   :width: 400
   :alt: Local Explanation Graph

FAQ
---

   Why should I use Trelawney rather than Lime_ and SHAP_

while you can definitally use the Lime and SHAP packages directly (they will give you more control over how to use their
packages), they are very specialized packages with different APIs, graphs and vocabulary. Trelawnaey offers you a
unified API, representation and vocabulary for all state of the art explanation methods so that you don't lose time
adapting to each new method but just change a class and Trelawney will adapt to you.

Comming Soon
------------

* Regressor Support (PR welcome)
* Image and text Support (PR welcome)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _SHAP: https://github.com/slundberg/shap
.. _Lime: https://github.com/marcotcr/lime
