# Реализация LogisticRegression

import numpy as np

class MyLogisticRegression:
    def __init__(self, lr, epochs, batch_size, random_state):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Прогноз
                z = X_batch @ self.w + self.b
                y_pred = self._sigmoid(z)
                
                # Градиенты
                error = y_pred - y_batch
                grad_w = (1 / len(batch_idx)) * (X_batch.T @ error)
                grad_b = error.mean()

                # Обновление весов
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
        
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)