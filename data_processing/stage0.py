import pandas as pd
from pathlib import Path

folder_list = []
train_dir = Path('train')

for folder in train_dir.iterdir():
    folder_list.append(folder)


for folder in folder_list:
    df_xyz = pd.read_csv(folder/'acceleration_clean.csv')
    df_pir = pd.read_csv(folder/'pir_clean.csv')
    df_targets = pd.read_csv(folder/'targets.csv')
    output_file = folder/'train_stage0.csv'

    df_pir['location_vector'] = df_pir.drop(['start', 'end'], axis=1).apply(lambda x: x.tolist(), axis=1)
    df_pir = df_pir[['start', 'end', 'location_vector']]

    
    df_targets['target_vector'] = df_targets.drop(['start', 'end'], axis=1).apply(lambda x: x.tolist(), axis=1)
    df_targets = df_targets[['start', 'end', 'target_vector']]

    df_merged = df_pir.merge(df_xyz, on=['start', 'end'], how='outer') \
                      .merge(df_targets, on=['start', 'end'], how='outer')
    

    df_merged.to_csv(output_file, index=False)
    print(f'Merged file saved as {output_file}')
    