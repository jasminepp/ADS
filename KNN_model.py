import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

datasets = []
for i in range(1, 11):
    df = pd.read_csv(f'train/{i:05d}/train_stage3.csv')
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
    
    # 使用KNeighborsClassifier作为基础分类器
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=21))
    model.fit(X_train, y_train)
    
    # KNN的predict_proba方法直接返回每个类的概率
    y_prob = model.predict_proba(X_test)
    # 计算布里尔分数
    brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
    print(f'brier_score: {brier_score}')
    brier_scores.append(brier_score)

print(f'Average score across all folds: {np.mean(brier_scores)}')