import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy import fftpack
from scipy.signal import resample

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

for i in range(1, 11):  # 从1到10
    directory = f'{i:05}'  # 格式化为5位数，前面填充0
    file_path = f'train/{directory}/AP_value.csv'
    df = pd.read_csv(file_path)

    # 将空值替换为np.nan，便于计算
    df.replace('', np.nan, inplace=True)

    # 将所有AP值转换为数值类型，并取反（因为原始数据是负数）
    for column in ['Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP']:
        df[column] = pd.to_numeric(df[column]) * -1

    # 找到最小和最大的AP值（现在是正数）
    min_value = df[['Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP']].min().min()
    max_value = df[['Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP']].max().max()

    # 应用缩放公式
    for column in ['Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP']:
        df[column] = (df[column] - min_value) / (max_value - min_value)

    # 将空值设为0
    df.fillna(0, inplace=True)

    # 保留小数点后五位
    df = df.round(5)

    # 保存处理后的数据
    output_file_path = f'train/{directory}/AP_value_scaled.csv'
    df.to_csv(output_file_path, index=False)
# def approximate_entropy(U, m, r):
#     """计算近似熵，简化版本"""
#     def _phi(m):
#         x = np.array([U[i:i + m] for i in range(N - m + 1)])
#         C = np.sum(np.max(np.abs(x[:, np.newaxis] - x[np.newaxis, :]), axis=2) <= r, axis=0) / (N - m + 1)
#         return np.sum(np.log(C)) / (N - m + 1)
#     N = len(U)
#     return np.abs(_phi(m + 1) - _phi(m))

# def extract_features(segment):
#     features = {}
#     features['x_mean'] = np.mean(segment[:, 0])
#     features['y_mean'] = np.mean(segment[:, 1])
#     features['z_mean'] = np.mean(segment[:, 2])
#     # # 峰度
#     # features['kurtosis'] = kurtosis(segment, axis=0)
#     # # 近似熵
#     # features['approx_entropy'] = np.array([approximate_entropy(segment[:, i], 2, 0.2 * np.std(segment[:, i])) for i in range(segment.shape[1])])
#     # # FFT前10频率
#     # fft_vals = fftpack.fft(segment, axis=0)
#     # fft_freq = fftpack.fftfreq(segment.shape[0])
#     # top_10_idx = np.argsort(np.abs(fft_vals), axis=0)[-10:]
#     # features['top_10_freq_by_fft'] = fft_freq[top_10_idx]
#     # # FFT分布峰度
#     # features['fft_distribution_kurtosis'] = kurtosis(np.abs(fft_vals), axis=0)
#     # # 平均急动度
#     # jerk = np.diff(segment, axis=0)
#     # features['average_jerk'] = np.mean(np.abs(jerk), axis=0)
#     # # 平均绝对值
#     # features['average_absolute_value'] = np.mean(np.abs(segment), axis=0)
#     # # 平均值
#     # features['average_value'] = np.mean(segment, axis=0)
#     # # 中位数
#     # features['median'] = np.median(segment, axis=0)
#     # # 标准差
#     # features['std_dev'] = np.std(segment, axis=0)
#     # # 最大值
#     # features['max_value'] = np.max(segment, axis=0)
#     # # 最小值
#     # features['min_value'] = np.min(segment, axis=0)
#     # # 最大绝对值
#     # features['max_absolute_value'] = np.max(np.abs(segment), axis=0)
    
#     return features

# # # 初始化一个空的DataFrame来存储特征
# # features_df = pd.DataFrame()
# # # # 假设有一个目录列表
# # # directories = ['train/00001', 'train/00002','train/00003', 'train/00004','train/00005', 'train/00006','train/00007', 'train/00008','train/00009', 'train/00010']  # 示例目录列表
# directory = 'train/00010'
# import pandas as pd

# # 加载数据
# pir_data = pd.read_csv(f'{directory}/pir_clean.csv')
# video_data = pd.read_csv(f'{directory}/video_combined_filtered.csv')

# # 假设已经有一个房间到传感器的映射字典
# room_to_sensor = {
#     'living_room': 'living',
#     'kitchen':    'kitchen',
#     'hallway':    'hallway'
#     # 添加其他房间和传感器的映射
# }

# # 时间对齐，这里简化处理，假设两个文件的时间戳已经可以直接对齐
# # 实际操作中可能需要更复杂的处理

# # 遍历PIR数据
# for index, row in pir_data.iterrows():
#     # 如果这一行所有传感器都没有数据
#     if row[2:].sum() == 0:  # 假设从第三列开始是传感器数据
#         # 找到对应时间段内的视频数据
#         video_row = video_data[(video_data['t'] >= row['start']) & (video_data['t'] <= row['end'])]
#         if not video_row.empty:
#             # 获取房间名称
#             room_name = video_row.iloc[0]['room']
#             # 根据房间名称找到对应的传感器
#             sensor_name = room_to_sensor.get(room_name)
#             if sensor_name:
#                 # 标记该传感器为激活状态
#                 pir_data.at[index, sensor_name] = 1

# # 保存补全后的数据
# pir_data.to_csv(f'{directory}/pir_clean_filled.csv', index=False)

# for directory in directories:
#     accel_path = f'{directory}/video_kitchen.csv'
#     targets_path = f'{directory}/targets.csv'
    
#     # 读取数据
#     accel_data = pd.read_csv(accel_path)
#     targets_data = pd.read_csv(targets_path)
    
#     for _, row in targets_data.iterrows():
#         start, end = row['start'], row['end']
#         # 分割数据
#         segment = accel_data[(accel_data['t'] >= start) & (accel_data['t'] <= end)][['x', 'y', 'z']].values
        
#         features = {
#         'start': start,
#         'end': end,
#         'x_mean': round(np.mean(segment[:, 0]), 3),  # 使用索引0访问x列
#          'y_mean': round(np.mean(segment[:, 1]), 3),  # 使用索引1访问y列
#         'z_mean': round(np.mean(segment[:, 2]), 3),  # 使用索引2访问z列
#      }
#     #         'Kitchen_AP_mean': round(np.mean(segment[:, 3]), 2),  # 使用索引0访问x列
#     # 'Lounge_AP_mean': round(np.mean(segment[:, 4]), 2),  # 使用索引1访问y列
#     # 'Upstairs_AP_mean': round(np.mean(segment[:, 5]), 2),  # 使用索引2访问z列
#     # 'Study_AP_mean': round(np.mean(segment[:, 6]), 2),  # 使用索引2访问z列
#         features_df = features_df._append(features, ignore_index=True)

#         # 保存到CSV文件
#     features_df.to_csv( f'{directory}/acceleration_clean.csv', index=False)
# import pandas as pd

# # 读取视频数据文件
# directory = 'train/00010'

# # 加载数据
# pir_data = pd.read_csv(f'{directory}/pir.csv')
# target_data = pd.read_csv(f'{directory}/targets.csv')

# # 定义传感器名称到特征向量索引的映射
# sensor_to_index = {
#     'bath': 0,
#     'bed1': 1,
#     'bed2': 2,
#     'hallway': 3,
#     'kitchen': 4,
#     'living': 5,
#     'stairs': 6,
#     'study': 7,
#     'toilet': 8
# }

# # 初始化一个空的DataFrame来存储结果
# aligned_data = pd.DataFrame(columns=['start', 'end'] + list(sensor_to_index.keys()))

# # 遍历target_data中的每个时间段
# for _, target_row in target_data.iterrows():
#     start, end = target_row['start'], target_row['end']
    
#     # 在pir_data中找到在这个时间段内激活的传感器
#     active_sensors = pir_data[(pir_data['start'] <= end) & (pir_data['end'] >= start)]
    
#     # 初始化特征向量
#     features = [0] * len(sensor_to_index)
    
#     # 标记激活的传感器
#     for _, sensor_row in active_sensors.iterrows():
#         sensor_name = sensor_row['name']
#         if sensor_name in sensor_to_index:
#             features[sensor_to_index[sensor_name]] = 1
    
#     # 将这个时间段的数据添加到aligned_data中
#     aligned_data = aligned_data._append(pd.DataFrame([[start, end] + features], columns=aligned_data.columns), ignore_index=True)

# # 保存到CSV文件
# aligned_data.to_csv(f'{directory}/pir_clean.csv', index=False)

# df_living_room = pd.read_csv(f'{directory}/video_living_room.csv')
# df_living_room['room'] = 'living_room'
# df_hallway = pd.read_csv(f'{directory}/video_hallway.csv')
# df_hallway['room'] = 'hallway'
# df_kitchen = pd.read_csv(f'{directory}/video_kitchen.csv')
# df_kitchen['room'] = 'kitchen'

# # 合并三个DataFrame
# df_combined = pd.concat([df_hallway, df_living_room, df_kitchen])

# # 按时间t排序
# df_combined.sort_values(by='t', inplace=True)

# # 读取目标时间段文件
# df_targets = pd.read_csv(f'{directory}/targets.csv')

# # 初始化一个空的DataFrame来存储筛选后的数据
# df_filtered = pd.DataFrame()

# # 根据targets.csv中的start和end筛选数据
# for index, row in df_targets.iterrows():
#     start, end = row['start'], row['end']
#     # 筛选在start和end时间段内的数据
#     df_temp = df_combined[(df_combined['t'] >= start) & (df_combined['t'] <= end)]
#     df_filtered = pd.concat([df_filtered, df_temp])

# # 重置索引
# df_filtered.reset_index(drop=True, inplace=True)

# # 保存筛选后的数据到新的CSV文件
# df_filtered.to_csv(f'{directory}/video_combined_filtered.csv', index=False)
