from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
import ast
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pathlib import Path

folder_list = []
train_dir = Path('train')
for folder in train_dir.iterdir():
    if folder.is_dir():
        folder_list.append(folder)

stage0_list = []
for folder in folder_list:
    stage0_list.append(folder/'train_stage0.csv')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# def is_valid_literal(s):
#     try:
#         ast.literal_eval(s)
#         return True
#     except ValueError:
#         return False

dataset = []
for file in stage0_list:
    df = pd.read_csv(file)
    target = ast.literal_eval(df['target_vector'])
    # df['target_vector'] = df['target_vector'].apply(ast.literal_eval)
    feature_df = df.drop(['start', 'end', 'target_vector'], axis=1)
    target_df = pd.DataFrame(np.array(df['target_vector']))
    # print(np.array(df['target_vector']))
    # print(target_df)
    dataset.append((feature_df, target_df))



brier_scores = []
for train_idx, test_idx in kf.split(dataset):
    
    X_train = pd.concat([dataset[i][0] for i in train_idx], ignore_index=True)
    y_train = pd.concat([dataset[i][1] for i in train_idx], ignore_index=True)

    X_test = pd.concat([dataset[i][0] for i in test_idx], ignore_index=True)
    y_test = pd.concat([dataset[i][1] for i in test_idx], ignore_index=True)

    # print(y_test)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)
    brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
    brier_scores.append(brier_score)
    # print(f'brier score = {brier_score}.')

# print(f'Stage0 Average score: {np.mean(brier_scores)}')


# datasets = []
# for i in range(1, 11):
#     df = pd.read_csv(f'train/{i:05d}/train_stage0.csv')
#     # print(type(df['target_vector']))
#     df.fillna(0, inplace=True)
    # df['location_vector'] = df['location_vector'].apply(ast.literal_eval)
    # def safe_literal_eval(s):
    #     try:
    #         return ast.literal_eval(s)
    #     except ValueError:
    #         # 当解析失败时，返回原始字符串或您认为合适的值
    #         return s  # 或者 return None

    # 安全地应用ast.literal_eval
    # df['target_vector'] = df['target_vector'].apply(safe_literal_eval)
    
#     df['target_vector'] = df['target_vector'].apply(ast.literal_eval) 
    
#     def set_max_to_one(target_list):
#         max_value = max(target_list)
#         return [1 if value == max_value else 0 for value in target_list]
    
#     target_df = pd.DataFrame(df['target_vector'].apply(set_max_to_one).tolist(), index=df.index)
#     features_df = df.drop(columns=['start', 'end','target_vector'])
#     datasets.append((features_df, target_df))
# kf = KFold(n_splits=5, shuffle=True, random_state=43)


# brier_scores = []
# model_importance = []

# for train_index, test_index in kf.split(datasets):

#     X_train = pd.concat([datasets[i][0] for i in train_index], ignore_index=True)
#     y_train = pd.concat([datasets[i][1] for i in train_index], ignore_index=True)
    

#     X_test = pd.concat([datasets[i][0] for i in test_index], ignore_index=True)
#     y_test = pd.concat([datasets[i][1] for i in test_index], ignore_index=True)
    

#     model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
#     model.fit(X_train, y_train)

#     model_importance.append(estimator.feature_importances_ for estimator in model.estimators_)
    

#     y_prob = model.predict_proba(X_test)
#     brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
#     print(f'brier score = {brier_score}.')
#     brier_scores.append(brier_score)

# print(f'Stage1 Average score across all folds: {np.mean(brier_scores)}')