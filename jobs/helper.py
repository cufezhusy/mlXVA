# ===================================================================================
# Model helper function
# ==================================================================================

import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(1)



def divide_data(X, Y):
    return train_test_split(X, Y, test_size=0.02, random_state=42)


def norm_ts(Z):
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    invert_func = lambda X: (Z.max() - Z.min()) * X + Z.min()
    return Z_norm, invert_func


def one_step_predict(model, train_features, train_labels):
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    return model