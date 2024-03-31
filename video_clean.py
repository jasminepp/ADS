import pandas as pd
import numpy as np


df = pd.read_csv('train/00010/video_combined_filtered.csv')


features_per_second = {}


for index, row in df.iterrows():
 
    t = int(row['t'])
    room = row['room']
    
    if t not in features_per_second:
        features_per_second[t] = {
            '2d_movement': [],
            '3d_movement': [],
            '2d_shape': [],
            '3d_shape': [],
            'room': room
        }
    

    centre_2d = np.array([row['centre_2d_x'], row['centre_2d_y']])
    

    centre_3d = np.array([row['centre_3d_x'], row['centre_3d_y'], row['centre_3d_z']])
    

    width_2d = row['bb_2d_br_x'] - row['bb_2d_tl_x']
    height_2d = row['bb_2d_br_y'] - row['bb_2d_tl_y']

    width_3d = row['bb_3d_brb_x'] - row['bb_3d_flt_x']
    height_3d = row['bb_3d_brb_y'] - row['bb_3d_flt_y']
    depth_3d = row['bb_3d_brb_z'] - row['bb_3d_flt_z']

    features_per_second[t]['2d_movement'].append(centre_2d)
    features_per_second[t]['3d_movement'].append(centre_3d)
    
   
    features_per_second[t]['2d_shape'].append((width_2d, height_2d))
    features_per_second[t]['3d_shape'].append((width_3d, height_3d, depth_3d))

# 计算每秒的平均特征
for t, features in features_per_second.items():
    # 计算2D运动特征的平均值和标准差
    features['2d_movement_mean'] = np.mean(features['2d_movement'], axis=0)
    features['2d_movement_std'] = np.std(features['2d_movement'], axis=0)
    
    # 计算3D运动特征的平均值和标准差
    features['3d_movement_mean'] = np.mean(features['3d_movement'], axis=0)
    features['3d_movement_std'] = np.std(features['3d_movement'], axis=0)
    
    # 计算2D形状特征的平均值和标准差
    features['2d_shape_mean'] = np.mean(features['2d_shape'], axis=0)
    features['2d_shape_std'] = np.std(features['2d_shape'], axis=0)
    
    # 计算3D形状特征的平均值和标准差
    features['3d_shape_mean'] = np.mean(features['3d_shape'], axis=0)
    features['3d_shape_std'] = np.std(features['3d_shape'], axis=0)


features_list = []
for t, features in features_per_second.items():
    features_list.append({
        'start': f"{t}.0", 
        'end': f"{t + 1}.0",
        '2d_movement_mean_x': round(features['2d_movement_mean'][0], 3),
        '2d_movement_mean_y': round(features['2d_movement_mean'][1], 3),
        '2d_movement_std_x': round(features['2d_movement_std'][0], 3),
        '2d_movement_std_y': round(features['2d_movement_std'][1], 3),
        '3d_movement_mean_x': round(features['3d_movement_mean'][0], 3),
        '3d_movement_mean_y': round(features['3d_movement_mean'][1], 3),
        '3d_movement_mean_z': round(features['3d_movement_mean'][2], 3),
        '3d_movement_std_x': round(features['3d_movement_std'][0], 3),
        '3d_movement_std_y': round(features['3d_movement_std'][1], 3),
        '3d_movement_std_z': round(features['3d_movement_std'][2], 3),
        '2d_shape_mean_width': round(features['2d_shape_mean'][0], 3),
        '2d_shape_mean_height': round(features['2d_shape_mean'][1], 3),
        '2d_shape_std_width': round(features['2d_shape_std'][0], 3),
        '2d_shape_std_height': round(features['2d_shape_std'][1], 3),
        '3d_shape_mean_width': round(features['3d_shape_mean'][0], 3),
        '3d_shape_mean_height': round(features['3d_shape_mean'][1], 3),
        '3d_shape_mean_depth': round(features['3d_shape_mean'][2], 3),
        '3d_shape_std_width': round(features['3d_shape_std'][0], 3),
        '3d_shape_std_height': round(features['3d_shape_std'][1], 3),
        'room': features['room']  
    })


features_df = pd.DataFrame(features_list)


features_df.to_csv('train/00010/video_feature.csv', index=False)