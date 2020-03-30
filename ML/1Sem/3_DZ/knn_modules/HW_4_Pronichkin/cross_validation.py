import numpy as np
from sklearn.model_selection import KFold


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is None:
        cv = []
        kfold = StratifiedKFold()
        for i in kfold.split(X, y):
            cv.append(i)
    classes = np.max(y) + 1
    dict_of_answers = {}
    for k in k_list:
        dict_of_answers[k] = []
    knn = KNNClassifier(k_list[-1], **kwargs)
    for ar in cv:
        if score == 'accuracy':
            if kwargs['weights'] is True:
                knn.fit(X[ar[0]], y[ar[0]])
                nearest_dist, nearest_neighbors = knn.find_kneighbors(
                    X[ar[1]],
                    return_distance=True)
                nearest_dist = np.vectorize(
                    lambda x: 1. /
                    (x + EPS))(nearest_dist)
                nearest_neighbors = \
                    y[ar[0]][nearest_neighbors]
                for k in k_list:
                    answers = np.zeros(X[ar[1]].shape[0], dtype=int)
                    answers = np.argmax(
                        np.sum(
                            (nearest_neighbors[:, :k][:, :, np.newaxis] ==
                             np.arange(classes)[
                                 np.newaxis,
                                 np.newaxis,
                                 :]).astype(int) *
                            nearest_dist[:, :k][:, :, np.newaxis], axis=1),
                        axis=1)
                    scor = np.sum(answers == y[ar[1]]) / len(answers)
                    dict_of_answers[k].append(scor)
            else:
                knn.fit(X[ar[0]], y[ar[0]])
                nearest_neighbors = knn.find_kneighbors(
                    X[ar[1]],
                    return_distance=False)
                nearest_neighbors = \
                    y[ar[0]][nearest_neighbors]
                for k in k_list:
                    answers = np.zeros(X[ar[1]].shape[0], dtype=int)
                    answers = np.argmax(
                        np.sum(
                            (nearest_neighbors[:, :k][:, :, np.newaxis] ==
                             np.arange(classes)[
                                 np.newaxis,
                                 np.newaxis,
                                 :]).astype(int),
                            axis=1),
                        axis=1)
                    scor = np.sum(answers == y[ar[1]]) / len(answers)
                    dict_of_answers[k].append(scor)
        else:
            return None
    return dict_of_answers
