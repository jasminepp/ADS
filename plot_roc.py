# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# # 加载数据
# with open('all_y_true.pkl', 'rb') as f:
#     all_y_true = pickle.load(f)

# with open('all_y_scores.pkl', 'rb') as f:
#     all_y_scores = pickle.load(f)

# class_names = [
#     'a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk',
#     'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat',
#     'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie',
#     't_sit_stand', 't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn'
# ]

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))

# ax_index = 0

# # 每五个类别绘制一个图
# for i in range(0, len(all_y_true), 5):
#     ax = axes[ax_index // 2, ax_index % 2]  
#     end = min(i + 5, len(all_y_true))  
#     for c in range(i, end):
#         fpr, tpr, _ = roc_curve(all_y_true[c], all_y_scores[c])
#         roc_auc = auc(fpr, tpr)
#         ax.plot(fpr, tpr, label=f'{class_names[c]} (area = {roc_auc:.2f})')

#     ax.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.set_title(f'ROC Curves for Classes {i} to {end-1}')
#     ax.legend(loc="lower right")
    
#     ax_index += 1  


# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix

with open('all_y_true.pkl', 'rb') as f:
    all_y_true = pickle.load(f)

with open('all_y_scores.pkl', 'rb') as f:
    all_y_scores = pickle.load(f)

# print(all_y_scores)
class_names = [
    'a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk',
    'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat',
    'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie',
    't_sit_stand', 't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn'
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))

ax_index = 0

# 每五个类别绘制一个图
for i in range(0, len(all_y_true), 5):
    ax = axes[ax_index // 2, ax_index % 2]
    end = min(i + 5, len(all_y_true))
    for c in range(i, end):

        fpr, tpr, thresholds = roc_curve(all_y_true[c], all_y_scores[c])

        specificity = 1 - fpr
        ax.plot(specificity, tpr, label=f'{class_names[c]} (area = {auc(fpr, tpr):.2f})')

    ax.plot([0, 1], [1, 0], 'k--') 
    ax.set_xlim([1.0, 0.0])  
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    # ax.set_title(f'Sensitivity vs Specificity for Classes {i} to {end-1}')
    ax.legend(loc="lower right")
    
    ax_index += 1

plt.tight_layout()
plt.show()