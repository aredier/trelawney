import pandas as pd
import numpy as np
import pytest
from keras import layers, models
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def fake_dataset():
    return (pd.DataFrame([list(range(100)), np.random.normal(size=100).tolist()], index=['real', 'fake']).T,
            np.array(range(100)) > 50)


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


    model = KerasClassifier(make_neural_network, epochs=5)
    model.fit(*fake_dataset)
    return model
