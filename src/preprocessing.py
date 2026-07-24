# Препроцессинг из ML-4

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def split_by_time(train_df):

    train_df = train_df.copy()
    train_df.drop(columns=['PRIMEUNIT', 'AUCGUART', 'WheelTypeID', 'RefId'], inplace=True)

    train_df['Transmission'] = train_df['Transmission'].str.upper().fillna('UNKNOWN')
    train_df['PurchDate'] = pd.to_datetime(train_df['PurchDate'])
    train_df = train_df.sort_values('PurchDate').reset_index(drop=True)

    dates = train_df['PurchDate'].drop_duplicates()
    bins = pd.qcut(dates, q=3, duplicates='drop')

    train_date = bins.cat.categories[0].right
    val_date   = bins.cat.categories[1].right

    train_df_split = train_df[train_df['PurchDate'] < train_date].copy()
    val_df_split   = train_df[(train_df['PurchDate'] >= train_date) & (train_df['PurchDate'] < val_date)].copy()
    test_df_split  = train_df[train_df['PurchDate'] >= val_date].copy()

    return train_df_split, val_df_split, test_df_split

# Обработка пропусков
def preprocess_data(train_df_split, val_df_split, test_df_split):

    for df in (val_df_split, test_df_split):
        df['Nationality'] = df['Nationality'].fillna(train_df_split['Nationality'].mode()[0])
        df['Size'] = df['Size'].fillna(train_df_split['Size'].mode()[0])

    for df in (train_df_split, val_df_split, test_df_split):
        df['WheelType'] = df['WheelType'].fillna('Unknown')
        df['Trim'] = df['Trim'].fillna('Unknown')

    for df in (train_df_split, val_df_split, test_df_split):
        df['Color'] = df['Color'].fillna(train_df_split['Color'].mode()[0])
        df['SubModel'] = df['SubModel'].fillna(train_df_split['SubModel'].mode()[0])

    median_cost = train_df_split['VehBCost'].median()
    for df in (train_df_split, val_df_split, test_df_split):
        df.loc[df['VehBCost'] < 1000, 'VehBCost'] = median_cost

# Buyer_Risk/Model_Mean_Price/Is_New_Car
    buyer_risk_map = train_df_split.groupby('BYRNO')['IsBadBuy'].mean()

    train_df_split['Buyer_Risk'] = train_df_split['BYRNO'].map(buyer_risk_map)
    val_df_split['Buyer_Risk'] = val_df_split['BYRNO'].map(buyer_risk_map)
    test_df_split['Buyer_Risk'] = test_df_split['BYRNO'].map(buyer_risk_map)

    val_df_split['Buyer_Risk'] = val_df_split['Buyer_Risk'].fillna(train_df_split['IsBadBuy'].mean())
    test_df_split['Buyer_Risk'] = test_df_split['Buyer_Risk'].fillna(train_df_split['IsBadBuy'].mean())

    for df in (train_df_split, val_df_split, test_df_split):
        df.drop(columns=['BYRNO'], inplace=True)

    model_mean = train_df_split.groupby('Model')['VehBCost'].mean()
    overall_mean = train_df_split['VehBCost'].mean()

    for df in (train_df_split, val_df_split, test_df_split):
        df['Model_Mean_Price'] = df['Model'].map(model_mean).fillna(overall_mean)

    for df in (train_df_split, val_df_split, test_df_split):
        df['Is_New_Car'] = (df['VehOdo'] < 15000).astype(int)

# MMR
    mmr_cols = train_df_split.filter(like='MMR').columns.tolist()
    mmr_medians = train_df_split[mmr_cols].replace(0, np.nan).median()
    
    for df in (train_df_split, val_df_split, test_df_split):
        df[mmr_cols] = df[mmr_cols].replace(0, np.nan).fillna(mmr_medians)
    
    for df in (train_df_split, val_df_split, test_df_split):
        df['MMR_Final'] = df[mmr_cols].median(axis=1)

# OdoPerYear
    for df in (train_df_split, val_df_split, test_df_split):
        df['OdoPerYear'] = df['VehOdo'] / (df['VehicleAge'] + 1)

# Преобразование log 
    log_cols = ['VehOdo', 'VehBCost', 'WarrantyCost', 'MMR_Final', 'Model_Mean_Price']

    for df in (train_df_split, val_df_split, test_df_split):
        df[log_cols] = np.log1p(df[log_cols])
        df[mmr_cols] = np.log1p(df[mmr_cols])

# Price_Diff/Model_Price_Diff
    for df in (train_df_split, val_df_split, test_df_split):
        df['Price_Diff'] = df['MMR_Final'] - df['VehBCost']
        df['Model_Price_Diff'] = df['Model_Mean_Price'] - df['VehBCost']

# one_hot
    ohe_features = ['Size', 'TopThreeAmericanName', 'Nationality', 'Transmission', 'WheelType', 'Auction', 'Color']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df_split[ohe_features])
    ohe_cols = encoder.get_feature_names_out(ohe_features)

    for df in (train_df_split, val_df_split, test_df_split):
        df[ohe_cols] = encoder.transform(df[ohe_features])
        df.drop(columns=ohe_features, inplace=True)

# count
    count_cols = ['Make', 'Model', 'SubModel', 'Trim', 'VNST', 'VNZIP1']

    for col in count_cols:
        counts = train_df_split[col].value_counts()
        for df in (train_df_split, val_df_split, test_df_split):
            df[col] = df[col].map(counts).fillna(0)

    return train_df_split, val_df_split, test_df_split
