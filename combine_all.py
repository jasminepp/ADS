
import pandas as pd


for i in range(1, 11):  
    directory = f'{i:05}' 
    # df_pir = pd.read_csv(f'train/{directory}/pir_clean_filled_update.csv')
    # df_video = pd.read_csv(f'train/{directory}/video_feature_scaled.csv')
    # df_xyz = pd.read_csv(f'train/{directory}/xyz_otherfeature_scaled.csv')
    # df_AP = pd.read_csv(f'train/{directory}/AP_value_scaled.csv')
    # df_targets = pd.read_csv(f'train/{directory}/targets.csv')
    # output_file = f'train/{directory}/train_stage3.csv'

    # data for stage1
    # df_AP = pd.read_csv(f'train/{directory}/AP_value.csv')
    df_xyz = pd.read_csv(f'train/{directory}/acceleration_clean.csv')
    df_pir = pd.read_csv(f'train/{directory}/pir_clean_filled_update.csv')
    df_video = pd.read_csv(f'train/{directory}/video_feature.csv')
    df_targets = pd.read_csv(f'train/{directory}/targets.csv')
    output_file = f'train/{directory}/train_stage1.csv'

    # #data for stage2
    # df_AP = pd.read_csv(f'train/{directory}/AP_value.csv')
    # df_xyz = pd.read_csv(f'train/{directory}/xyz_otherfeature.csv')
    # df_pir = pd.read_csv(f'train/{directory}/pir_clean_filled_update.csv')
    # df_video = pd.read_csv(f'train/{directory}/video_feature.csv')
    # df_targets = pd.read_csv(f'train/{directory}/targets.csv')
    # output_file = f'train/{directory}/train_stage2.csv'

    def row_to_vector(row):

      index_of_one = row.idxmax()
      return row.index.get_loc(index_of_one)

   # process pir
    df_pir['location_idx'] = df_pir.iloc[:, 2:].apply(row_to_vector, axis=1)
    df_pir = df_pir[['start', 'end', 'location_idx']]

    # target_columns = df_AP.columns.difference(['start', 'end'])
    # df_AP['AP_vector'] = df_AP[target_columns].apply(lambda x: x.tolist(), axis=1)
    # df_AP = df_AP[['start', 'end', 'AP_vector']]

    # transfer target to vector
    target_columns = df_targets.columns.difference(['start', 'end'])
    df_targets['target_vector'] = df_targets[target_columns].apply(lambda x: x.tolist(), axis=1)

    df_targets = df_targets[['start', 'end', 'target_vector']]
 
    # .merge(df_AP, on=['start', 'end'], how='outer')\
    df_merged = df_pir.merge(df_video, on=['start', 'end'], how='outer') \
                      .merge(df_xyz, on=['start', 'end'], how='outer') \
                      .merge(df_targets, on=['start', 'end'], how='outer')

    df_merged.to_csv(output_file, index=False)
    print(f'Merged file saved as {output_file}')