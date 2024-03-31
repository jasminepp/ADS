import pandas as pd

pir = pd.read_csv('train/00001/pir_clean_filled.csv')
targets = pd.read_csv('train/00001/targets.csv')
merged_df = pd.merge(pir, targets, on=['start', 'end'], how='outer')
merged_df.to_csv('train/00001/targets_combined.csv', index=False)