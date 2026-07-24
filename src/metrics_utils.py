import pandas as pd

results_df = pd.DataFrame(columns=['model', 'gini_score'])

def add_metric(model_name, gini):
    global results_df  

    if model_name in results_df['model'].values:
        idx = results_df.index[results_df['model'] == model_name][0]
        results_df.loc[idx, 'gini_score'] = round(float(gini), 4)
    else:
        new_row = pd.DataFrame({
            'model': [str(model_name)],
            'gini_score': [round(float(gini), 4)]
        })
        
        if results_df.empty:
            results_df = new_row
        else:
            results_df = pd.concat([results_df, new_row], ignore_index=True)