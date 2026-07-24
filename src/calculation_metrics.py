# Реализация расчета метрик Recall, Precision, F1-меры и AUC PR

import numpy as np

def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC PR
    desc_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[desc_indices]
    
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    
    # Precision и Recall 
    recall_array = tp_cum / np.sum(y_true)
    precision_array = tp_cum / (tp_cum + fp_cum)

    recall_curve = np.concatenate([[0], recall_array])
    precision_curve = np.concatenate([[1], precision_array]) 
    
    auc_pr = np.trapz(precision_curve, recall_curve)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC PR": auc_pr
    }