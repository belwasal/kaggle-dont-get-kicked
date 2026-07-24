# Реализация NaiveBayes

import numpy as np

class MyNaiveBayes:
    def fit(self, X, y):
        # Перевод в numpy (быстрее чем строки)
        X, y = np.array(X), np.array(y)
        self.all_classes = np.unique(y)
        self.class_stats = []

        for current_class in self.all_classes:
            class_data = X[y == current_class]
            
            # Среднее, разброс и априорная вероятность
            stats = {
                'mean': class_data.mean(axis=0),
                'variance': class_data.var(axis=0),
                'prior': len(class_data) / len(X)
            }
            self.class_stats.append(stats)

    def predict_proba(self, X):
        X = np.array(X)
        final_predictions = []

        for row in X:
            class_chances = []
            
            for stats in self.class_stats:
                total_chance = stats['prior'] # Априорная вероятность
                
                for i in range(len(row)):
                    current_variance = stats['variance'][i]
                    current_mean = stats['mean'][i]
                    feature_value = row[i]

                    if current_variance > 0:
                        # Разница между значением и нормой (средним)
                        diff = feature_value - current_mean
                        
                        # Гаусс (MLE)
                        exponent = np.exp(- (diff**2) / (2 * current_variance))
                        normalization = np.sqrt(2 * np.pi * current_variance)
                        
                        total_chance *= (exponent / normalization)
                
                class_chances.append(total_chance)

           
            total_sum = sum(class_chances) # Шансы -> итоговая вероятность
            
            if total_sum == 0:
                final_predictions.append(0.5)
            else:
                prob_of_class_1 = class_chances[1] / total_sum
                final_predictions.append(prob_of_class_1)
                
        return np.array(final_predictions)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)