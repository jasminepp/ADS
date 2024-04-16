import pandas as pd

for i in range(10, 11):  
    directory = f'{i:05}' 

    df_xyz = pd.read_csv(f'train/{directory}/acceleration_clean.csv')
    df_pir = pd.read_csv(f'train/{directory}/pir_clean.csv')
    df_targets = pd.read_csv(f'train/{directory}/targets.csv')
    output_file = f'train/{directory}/train_stage0.csv'

    def row_to_vector(row):
      index_of_one = row.idxmax()
      return row.index.get_loc(index_of_one)

    # process pir
    pir_columns = df_pir.columns.difference(['start', 'end'])
    df_pir['location_idx'] = df_pir[pir_columns].apply(lambda x: x.tolist(), axis=1)

    df_pir = df_pir[['start', 'end', 'location_idx']]

    # transfer target to vector
    target_columns = df_targets.columns.difference(['start', 'end'])
    df_targets['target_vector'] = df_targets[target_columns].apply(lambda x: x.tolist(), axis=1)

    df_targets = df_targets[['start', 'end', 'target_vector']]
 
    df_merged = df_pir.merge(df_xyz, on=['start', 'end'], how='outer') \
                      .merge(df_targets, on=['start', 'end'], how='outer')

    df_merged.to_csv(output_file, index=False)
    print(f'Merged file saved as {output_file}')