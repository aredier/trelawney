import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def fake_dataset():
    return (pd.DataFrame([list(range(100)), np.random.normal(size=100).tolist()], index=['real', 'fake']).T,
            np.array(range(100)) > 50)


@pytest.fixture
def fitted_logistic_regression(fake_dataset):
    model = LogisticRegression()
    return model.fit(*fake_dataset)
