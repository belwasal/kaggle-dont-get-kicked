# # Реализация Gini

import numpy as np

def custom_gini(y_true, y_probs):
    data = sorted(zip(y_probs, y_true), key=lambda x: x[0], reverse=True)
    
    total_bad = sum(y_true)
    total_good = len(y_true) - total_bad
    
    if total_bad == 0 or total_good == 0:
        auc_final = 0.5
    else:
        auc_final = 0.0
        current_height = 0.0
        
        step_up = 1 / total_bad
        step_right = 1 / total_good
        
        i = 0
        n = len(data)
        while i < n:
            prob_group = data[i][0]
            group_bads = 0
            group_goods = 0
            
            while i < n and data[i][0] == prob_group:
                if data[i][1] == 1:
                    group_bads += 1
                else:
                    group_goods += 1
                i += 1
                
            # Трапеция для группы
            old_height = current_height
            current_height += group_bads * step_up
            auc_final += (old_height + current_height) / 2 * (group_goods * step_right)
            
    gini = 2 * auc_final - 1
    return gini