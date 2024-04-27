from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler

datasets = []
for i in range(1, 11):
    directory = f'{i:05}' 
    output_file = f'train/{directory}/train_stage2_pca.csv'

    df = pd.read_csv(f'train/{i:05d}/train_stage2.csv')
    df.fillna(0, inplace=True)
    
    pca = PCA(n_components=30)
    X = df.drop(['start', 'end', 'target_vector'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_X = pca.fit_transform(X_scaled)
    new_X_df = pd.DataFrame(new_X)
    feature_df = pd.concat([df[['start', 'end']], new_X_df], axis=1)
    # feature_df.columns = ['start', 'end', 'feature1', 'feature2']

    df_merged = feature_df.merge(df[['start', 'end', 'target_vector']], how='outer')

    df_merged.to_csv(output_file, index=False)
    print(f'Merged file saved as {output_file}')

# datasets = []
# for i in range(1, 11):
#     directory = f'{i:05}' 
#     output_file = f'train/{directory}/train_stage3_pca.csv'

#     df = pd.read_csv(f'train/{i:05d}/train_stage3.csv')
#     df.fillna(0, inplace=True)

#     pca = PCA(n_components=24)
#     X = df.drop(['start', 'end', 'target_vector'], axis=1)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     new_X = pca.fit_transform(X_scaled)
#     new_X_df = pd.DataFrame(new_X)
#     feature_df = pd.concat([df[['start', 'end']], new_X_df], axis=1)
#     # feature_df.columns = ['start', 'end', 'feature1', 'feature2']

#     df_merged = feature_df.merge(df[['start', 'end', 'target_vector']], how='outer')

#     df_merged.to_csv(output_file, index=False)
#     print(f'Merged file saved as {output_file}')


