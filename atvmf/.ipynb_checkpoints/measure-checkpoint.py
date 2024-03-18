import numpy as np
from sklearn.metrics import adjusted_rand_score

def rand_index_dataset(y_true, y_pred) -> float:
    """Compute the individual rand index for each sample in the dataset and then average it"""
    individual_rand_index = []
    for row_index in range(y_true.shape[0]):
        labels = y_true[row_index]
        preds = y_pred[row_index]
        individual_rand_index.append(adjusted_rand_score(labels, preds))
    return np.mean(individual_rand_index)