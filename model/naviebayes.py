import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB 
from sklearn.multioutput import MultiOutputClassifier
import ast
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer

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
    # X_train = X_train.fillna(0)
    # X_test = X_test.fillna(0)


    imputer = SimpleImputer(strategy='mean')  # 可以选择'mean', 'median', 'most_frequent'等作为填充策略
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = MultiOutputClassifier(GaussianNB())
    model.fit(X_train, y_train)
    

    y_prob = model.predict_proba(X_test)

    brier_score = np.mean([brier_score_loss(y_test.iloc[:, c], y_prob[c][:, 1], pos_label=1) for c in range(y_test.shape[1])])
    brier_scores.append(brier_score)

print(f'Stage 1 Average score across all folds: {np.mean(brier_scores)}')
