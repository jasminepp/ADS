import pandas as pd

df = pd.read_csv('train/00001/targets_combined.csv')

# get targets, use the highest probability
def convert_prob_to_label(row):
    probabilities = row[11:]
    max_prob_index = probabilities.idxmax()
    return max_prob_index

df['label'] = df.apply(convert_prob_to_label, axis=1)

df.to_csv('train/00001/train_data.csv', index=False)