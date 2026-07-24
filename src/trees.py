# Реализация деревьев 
import numpy as np

# Узел
class TreeNode:
    def __init__(self, depth):
        self.depth = depth
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold_value = None
        self.is_leaf = False
        self.leaf_output = None

# DecisionTree
class BaseDecisionTree:
    def __init__(self, max_depth=None, split_mode="normal", random_state=None):
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.random_state = random_state
        self.root = None

    def impurity(self, y):
        raise NotImplementedError

    def leaf_value(self, y):
        raise NotImplementedError

    def find_best_split(self, X, y):
        if self.split_mode == "extra":
            feature_index = np.random.randint(0, X.shape[1])
            values = X[:, feature_index]
            if len(np.unique(values)) <= 1: return None, None
            return feature_index, np.random.uniform(values.min(), values.max())
        
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_score = float('inf')

        for feature_index in range(n_features):
            values = X[:, feature_index]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1: continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 # Перебор thresholds
            
            for threshold in thresholds:
                left_mask = values <= threshold
                y_left, y_right = y[left_mask], y[~left_mask]
                
                if len(y_left) == 0 or len(y_right) == 0: continue

                current_score = len(y_left) * self.impurity(y_left) + len(y_right) * self.impurity(y_right) # impurity 

                if current_score < best_score:
                    best_score = current_score
                    best_feature = feature_index
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        node = TreeNode(depth)
        
        # Не останавливаемся, пока есть куда расти (в отличие от sklearn)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) <= 1 or len(y) < 2:
            node.is_leaf = True
            node.leaf_output = self.leaf_value(y)
            return node

        feature_index, threshold = self.find_best_split(X, y)
        if feature_index is None:
            node.is_leaf = True
            node.leaf_output = self.leaf_value(y)
            return node

        node.feature_index = feature_index
        node.threshold_value = threshold
        
        left_mask = X[:, feature_index] <= threshold
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        if self.random_state is not None: np.random.seed(self.random_state)
        self.root = self.build_tree(X, y, 0)

    def _predict_one(self, x, node):
        if node.is_leaf: return node.leaf_output
        if x[node.feature_index] <= node.threshold_value:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(sample, self.root) for sample in X])

# DecisionTreeClassifier
class MyDecisionTreeClassifier(BaseDecisionTree):
    def impurity(self, y):
        p = np.mean(y)
        return p * (1 - p) # Gini
    def leaf_value(self, y):
        return np.mean(y)
    def predict_proba(self, X):
        positive_probs = super().predict(X)
        return np.column_stack([1 - positive_probs, positive_probs])
    def predict(self, X):
        return (super().predict(X) >= 0.5).astype(int)

class MyDecisionTreeRegressor(BaseDecisionTree):
    def impurity(self, y):
        return np.var(y) if len(y) > 0 else 0
    def leaf_value(self, y):
        return np.mean(y)

# RandomForestClassifier
class MyRandomForestClassifier:
    def __init__(self, n_estimators=30, max_depth=7, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        if self.random_state is not None: np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        m_feat = int(np.sqrt(n_features)) if self.max_features == "sqrt" else n_features
        
        self.trees = []
        for tree_index in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            feat_idx = np.random.choice(n_features, m_feat, replace=False)
            tree = MyDecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append((tree, feat_idx))

    def predict_proba(self, X):
        probs = [tree.predict_proba(X[:, f_idx])[:, 1] for tree, f_idx in self.trees]
        avg_prob = np.mean(probs, axis=0)
        return np.column_stack([1 - avg_prob, avg_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# GBDT
class MyGBDTClassifier:
    def __init__(self, n_estimators=30, max_depth=3, learning_rate=0.1, n_bins=32, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_bins = n_bins
        self.random_state = random_state
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def fit(self, X, y):
        if self.random_state is not None: np.random.seed(self.random_state)
        n_features = X.shape[1]
        
        # Делаем биннинг только (как в sclearn для скорости обучения ансамбля, иначе очень долго по всем строкам)
        self.bin_edges = [np.linspace(X[:, i].min(), X[:, i].max(), self.n_bins + 1) for i in range(n_features)]
        X_binned = np.array([np.digitize(X[:, i], self.bin_edges[i]) - 1 for i in range(n_features)]).T
        
        logits = np.zeros(len(y))
        for step_index in range(self.n_estimators):
            gradients = y - self._sigmoid(logits)
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            
            m_feat = int(np.sqrt(n_features))
            feat_idx = np.random.choice(n_features, m_feat, replace=False)
            
            tree.fit(X_binned[:, feat_idx], gradients)
            self.trees.append((tree, feat_idx))
            logits += self.learning_rate * tree.predict(X_binned[:, feat_idx])

    def predict_proba(self, X):
        X_b = np.array([np.digitize(X[:, i], self.bin_edges[i]) - 1 for i in range(X.shape[1])]).T
        logits = np.zeros(len(X))
        for tree, feat_idx in self.trees:
            logits += self.learning_rate * tree.predict(X_b[:, feat_idx])
        p = self._sigmoid(logits)
        return np.column_stack([1 - p, p])

# ExtraTreesClassifier
class MyExtraTreesClassifier(MyRandomForestClassifier):
    def fit(self, X, y):
        if self.random_state is not None: np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        m_feat = int(np.sqrt(n_features)) if self.max_features == "sqrt" else n_features
        for tree_index in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            feat_idx = np.random.choice(n_features, m_feat, replace=False)
            tree = MyDecisionTreeClassifier(max_depth=self.max_depth, split_mode="extra")
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append((tree, feat_idx))