import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import ast
from sklearn.metrics import brier_score_loss

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
import ast
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

datasets = []
for i in range(1, 11):
    df = pd.read_csv(f'train/{i:05d}/train_stage2.csv')
    df['target_vector'] = df['target_vector'].apply(ast.literal_eval) 
    
    # target_vector max to 1 other to 0
    def set_max_to_one(target_list):
        max_value = max(target_list)
        return [1 if value == max_value else 0 for value in target_list]
    
    target_df = pd.DataFrame(df['target_vector'].apply(set_max_to_one).tolist(), index=df.index)
    features_df = df.drop(columns=['target_vector'])
    datasets.append((features_df, target_df))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

brier_scores = []

for train_index, test_index in kf.split(datasets):
    X_train = pd.concat([datasets[i][0] for i in train_index], ignore_index=True)
    y_train = pd.concat([datasets[i][1] for i in train_index], ignore_index=True)
    
    X_test = pd.concat([datasets[i][0] for i in test_index], ignore_index=True)
    y_test = pd.concat([datasets[i][1] for i in test_index], ignore_index=True)
    
 
    model = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    model.fit(X_train, y_train)
    
 
    y_prob = model.predict_proba(X_test)
   
    brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
    # print(f'brier_score: {brier_score}')
    brier_scores.append(brier_score)

print(f'Stage 2 Average score across all folds: {np.mean(brier_scores)}')



# check nan and delete it
# def load_dataset(dataset_id):
#     file_path = f'train/{dataset_id:05d}/train_stage2.csv'
#     df = pd.read_csv(file_path)
#     df = df.dropna()
#     return df

# datasets = [load_dataset(id) for id in range(1, 11)]
# for i, dataset in enumerate(datasets, start=1):
#     for index, row in dataset.iterrows():
#         try:
       
#             _ = ast.literal_eval(row.iloc[-1])
#         except Exception as e:
     
#             print(f"Dataset {i}, Row {index}: {e}")

#             dataset = dataset.drop(index)

# import re

# def load_dataset(dataset_id):
#     file_path = f'train/{dataset_id:05d}/train_stage3.csv'
#     df = pd.read_csv(file_path)
#     df = df[~df.iloc[:, -1].str.contains(r'\bnan\b', na=False)]
#     return df

# datasets = [load_dataset(id) for id in range(1, 11)]
# for i, dataset in enumerate(datasets, start=1):
#     file_path = f'train/{i:05d}/train_stage3.csv'
#     dataset.to_csv(file_path, index=False)
# # 