import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
import torch.nn.functional as F
def weighted_brier_score(predictions, true_labels, class_weights):
    N, C = predictions.shape
    bs = 0.0
    for c in range(C):
        term = (predictions[:, c] - true_labels[:, c]) ** 2
        bs += class_weights[c] * torch.sum(term)
    bs /= N
    return bs
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.fc(lstm_out[:, -1, :])
        y_pred = F.softmax(y_pred, dim=1)  
        return y_pred

def create_sliding_windows(data, sequence_length, step=10):
    X, y = [], []
    for i in range(0, len(data[0]) - sequence_length, step):  
        X.append(data[0][i:i+sequence_length])
        y.append(data[1][i+sequence_length-1])
    return np.array(X), np.array(y)


sequence_length = 5 
step =1
datasets = []
for i in range(1, 11):
    df = pd.read_csv(f'./train/{i:05d}/train_stage2.csv')
    df['target_vector'] = df['target_vector'].apply(ast.literal_eval) 
    
    def set_max_to_one(target_list):
        max_value = max(target_list)
        return [1 if value == max_value else 0 for value in target_list]
    
    target_df = pd.DataFrame(df['target_vector'].apply(set_max_to_one).tolist(), index=df.index)
    # target_df = pd.DataFrame(df['target_vector'], index=df.index)
    features_df = df.drop(columns=['target_vector'])
    sliding_X, sliding_y = create_sliding_windows((features_df.values, target_df.values), sequence_length, step= step)
    datasets.append((sliding_X, sliding_y))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

brier_scores = []

for train_index, test_index in kf.split(datasets):
    X_train = np.concatenate([datasets[i][0] for i in train_index])
    y_train = np.concatenate([datasets[i][1] for i in train_index])
    X_test = np.concatenate([datasets[i][0] for i in test_index])
    y_test = np.concatenate([datasets[i][1] for i in test_index])
    

    X_train = X_train[:, :, 2:]  # 假设数据的形状是 [samples, timesteps, features]
    X_test = X_test[:, :, 2:]


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # print(f'X_train_tensor.shape{X_train_tensor.shape}')
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=1, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1, shuffle=False)

    model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=100, output_dim=y_train.shape[1], num_layers=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
# 训练模型
model.train()
for epoch in range(10):  
    print(f'current_epoch{epoch}')
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
with torch.no_grad():
    total_loss = 0
    class_weights = torch.tensor([1.0] * labels.shape[1], dtype=torch.float32)  
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = weighted_brier_score(outputs, labels, class_weights)
        total_loss += loss.item()
    average_loss = total_loss / len(test_loader)

    brier_scores.append(average_loss)

print(f'Average Brier score across all folds: {np.mean(brier_scores)}')
