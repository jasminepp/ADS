from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler

datasets = []
for i in range(1, 11):
    df = pd.read_csv(f'train/{i:05d}/train_stage2.csv')
    df.fillna(0, inplace=True)
    df['target_vector'] = df['target_vector'].apply(ast.literal_eval) 
    
    # target_vector max to 1 other to 0
    def set_max_to_one(target_list):
        max_value = max(target_list)
        return [1 if value == max_value else 0 for value in target_list]
    
    target_df = pd.DataFrame(df['target_vector'].apply(set_max_to_one).tolist(), index=df.index)
    features_df = df.drop(columns=['start', 'end', 'location_idx', 'target_vector'])
    datasets.append((features_df, target_df))

X = pd.concat([datasets[i][0] for i in range(len(datasets))], ignore_index=True)
# y = pd.concat([datasets[i][1] for i in range(len(datasets))], ignore_index=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA().fit(X_scaled)

# plt.figure()
# plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Different Principal Components')
# plt.grid(True)
# plt.show()

n_comp = list(range(1, len(pca.explained_variance_ratio_)+1))
ratio = pca.explained_variance_ratio_.cumsum()
comp_analysis = pd.DataFrame({'Number of Components': n_comp, 'Explained Variance Ratio': ratio})
comp_analysis.to_csv('pca/pca_components_analysis.csv', index=False)

n_components = next(i for i, cumsum in enumerate(pca.explained_variance_ratio_.cumsum(), 1) if cumsum > 0.99)
print(f'Number of components to keep for 99% variance: {n_components}')


