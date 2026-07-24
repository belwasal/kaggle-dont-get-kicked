# Реализация KNN

import numpy as np

class MyKNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict_proba(self, X):
        X_test = np.array(X)
        all_probabilities = []

        for row_to_predict in X_test:
            # Считаем Евклидово расстояние
            distances = np.sqrt(np.sum((self.X_train - row_to_predict)**2, axis=1))
            
            # Ближайшие соседи k и их mean
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            probability = np.mean(nearest_labels)
            
            all_probabilities.append(probability)
            
        return np.array(all_probabilities)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)