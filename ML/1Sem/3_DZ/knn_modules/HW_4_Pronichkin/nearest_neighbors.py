import numpy as np
from sklearn.neighbors import NearestNeighbors
EPS = 1e-5

def cos_vector(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class KNNClassifier:
    def __init__(
            self,
            k,
            strategy='brute',
            metric='euclidean',
            weights=False,
            test_block_size=20):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy != 'my_own':
            if metric == 'euclidean':
                self.NearestNeighborsClass = NearestNeighbors(
                    k, algorithm=strategy, metric='euclidean')
            else:
                self.NearestNeighborsClass = NearestNeighbors(
                    k, algorithm=strategy, metric='cosine')

    def fit(self, X, y):
        if self.strategy == 'my_own':
            self.Neibors_coord = X
            self.Neibors_label = y
            if self.metric == 'euclidean':
                self.Neibors_norm = np.sum(X ** 2, axis=1)
            else:
                self.Neibors_norm = np.linalg.norm(X, axis=1)
        else:
            self.NearestNeighborsClass.fit(X, y)
            self.Neibors_label = y

    def find_kneighbors(self, X, return_distance):
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                if return_distance is True:
                    rows = np.zeros((self.test_block_size, self.k), dtype=int)
                    rows[:, :] = \
                        np.array(
                            np.arange(self.test_block_size)
                        ).reshape(self.test_block_size, 1)
                    distance = np.zeros((X.shape[0], self.k))
                    positions = np.zeros((X.shape[0], self.k), dtype=int)
                    for i in range(0, X.shape[0], self.test_block_size):
                        Distance_matrix = euclidean_distance(
                            X[i:i + self.test_block_size],
                            self.Neibors_coord,
                            self.Neibors_norm)
                        positions[i:i + self.test_block_size] = \
                            np.argsort(Distance_matrix)[:, :self.k]
                        distance[i:i + self.test_block_size] = \
                            Distance_matrix[
                                rows[:Distance_matrix.shape[0]],
                                positions[i:i + self.test_block_size]
                                ]
                    return (distance, positions)
                else:
                    positions = np.zeros((X.shape[0], self.k), dtype=int)
                    for i in range(0, X.shape[0], self.test_block_size):
                        Distance_matrix = euclidean_distance(
                            X[i:i + self.test_block_size],
                            self.Neibors_coord,
                            self.Neibors_norm)
                        positions[i:i + self.test_block_size] = \
                            np.argsort(Distance_matrix)[:, :self.k]
                    return positions
            if self.metric == 'cosine':
                if return_distance is True:
                    rows = np.zeros((self.test_block_size, self.k), dtype=int)
                    rows[:, :] = \
                        np.array(
                            np.arange(self.test_block_size)
                        ).reshape(self.test_block_size, 1)
                    distance = np.zeros((X.shape[0], self.k))
                    positions = np.zeros((X.shape[0], self.k), dtype=int)
                    for i in range(0, X.shape[0], self.test_block_size):
                        Distance_matrix = cosine_distance(
                            X[i:i + self.test_block_size],
                            self.Neibors_coord)
                        positions[i:i + self.test_block_size] = \
                            np.argsort(Distance_matrix)[:, :self.k]
                        distance[i:i + self.test_block_size] = \
                            Distance_matrix[
                                rows[:Distance_matrix.shape[0]],
                                positions[i:i + self.test_block_size]
                                ]
                    return (distance, positions)
                else:
                    positions = np.zeros((X.shape[0], self.k), dtype=int)
                    for i in range(0, X.shape[0], self.test_block_size):
                        Distance_matrix = cosine_distance(
                            X[i:i + self.test_block_size],
                            self.Neibors_coord)
                        positions[i:i + self.test_block_size] = \
                            np.argsort(Distance_matrix)[:, :self.k]
                    return positions
        else:
            distance = np.zeros((X.shape[0], self.k))
            positions = np.zeros((X.shape[0], self.k), dtype=int)
            if return_distance is True:
                for i in range(0, X.shape[0], self.test_block_size):
                    distance[i:i + self.test_block_size], \
                        positions[i:i + self.test_block_size] = \
                        self.NearestNeighborsClass.kneighbors(
                        X[i:i + self.test_block_size])
                return (distance, positions)
            else:
                for i in range(0, X.shape[0], self.test_block_size):
                    positions[i:i + self.test_block_size] = \
                        self.NearestNeighborsClass.kneighbors(
                            X[i:i + self.test_block_size],
                            return_distance=False)
                return positions

    def predict(self, X):
        classes = np.max(self.Neibors_label) + 1
        if self.weights is False:
            nearest_neighbors = self.find_kneighbors(X, return_distance=False)
            nearest_neighbors = self.Neibors_label[nearest_neighbors]
            answers = np.zeros(X.shape[0], dtype=int)
            answers = np.argmax(np.sum(
                (nearest_neighbors[:, :, np.newaxis] ==
                    np.arange(classes)[
                        np.newaxis,
                        np.newaxis, :]).astype(int), axis=1), axis=1)
            return answers
        else:
            answers = np.zeros(X.shape[0], dtype=int)
            nearest_dist, nearest_neighbors = self.find_kneighbors(
                X,
                return_distance=True)
            nearest_neighbors = self.Neibors_label[nearest_neighbors]
            nearest_dist = np.vectorize(
                lambda x: 1. /
                (x + EPS))(nearest_dist)
            answers = np.argmax(
                np.sum(
                    (nearest_neighbors[:, :, np.newaxis] ==
                        np.arange(classes)[
                            np.newaxis,
                            np.newaxis,
                            :]).astype(int) *
                    nearest_dist[:, :, np.newaxis], axis=1),
                axis=1)
            return answers
