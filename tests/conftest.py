import pandas as pd
import numpy as np
import pytest
from keras import layers, models
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import base
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def fake_dataset():
    return (pd.DataFrame([list(range(100)), np.random.normal(size=100).tolist()], index=['real', 'fake']).T,
            (np.array(range(100)) > 50).astype(np.int16))


@pytest.fixture
def fitted_logistic_regression(fake_dataset):
    model = LogisticRegression()
    return model.fit(*fake_dataset)


@pytest.fixture
def fitted_neural_network(fake_dataset):

    def make_neural_network():
        model = models.Sequential([
            layers.Dense(2, input_shape=(2,), activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    model = KerasClassifier(make_neural_network, epochs=100, batch_size=100)
    model.fit(*fake_dataset)
    return model


@pytest.fixture
def FakeClassifier():

    class FakeClassifierInner(base.BaseEstimator):

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.array([[0.5, 0.5] for _ in  range(X.shape[0])])

        def predict(self, X):
            return self.predict_proba(X)[:, 1] >= 0.5

    return FakeClassifierInner


@pytest.fixture
def fitted_decision_tree(fake_dataset):

    model = DecisionTreeClassifier()
    model.fit(*fake_dataset)
    return model
