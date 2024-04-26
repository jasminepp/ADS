from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
import ast
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pathlib import Path

datasets = []
for i in range(1, 11):
    df = pd.read_csv(f'train/{i:05d}/train_stage0.csv')
    df.fillna(0, inplace=True)
    df['location_idx'] = df['location_idx'].apply(ast.literal_eval)    
    df['target_vector'] = df['target_vector'].apply(ast.literal_eval) 

    def set_max_to_one(target_list):
        max_value = max(target_list)
        return [1 if value == max_value else 0 for value in target_list]
    
    def set_list(target_list):
        return [value for value in target_list]
    
    target_df = pd.DataFrame(df['target_vector'].apply(set_max_to_one).tolist(), index=df.index)
    location_df = pd.DataFrame(df['location_idx'].apply(set_list).tolist(), index=df.index)
 
    features_df = df.drop(columns=['start', 'end', 'location_idx','target_vector'])
    for i in range(0, 9):
        features_df[str(i)] = location_df[i]

    datasets.append((features_df, target_df))
kf = KFold(n_splits=5, shuffle=True, random_state=43)


brier_scores = []
model_importance = []

for train_index, test_index in kf.split(datasets):

    X_train = pd.concat([datasets[i][0] for i in train_index], ignore_index=True)
    y_train = pd.concat([datasets[i][1] for i in train_index], ignore_index=True)
    

    X_test = pd.concat([datasets[i][0] for i in test_index], ignore_index=True)
    y_test = pd.concat([datasets[i][1] for i in test_index], ignore_index=True)
    

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    model_importance.append(estimator.feature_importances_ for estimator in model.estimators_)
    

    y_prob = model.predict_proba(X_test)
    brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
    print(f'brier score = {brier_score}.')
    brier_scores.append(brier_score)

print(f'Stage0 Average score across all folds: {np.mean(brier_scores)}')
pd.DataFrame(model_importance).to_csv('rf_stage0_feature_importance.csv')