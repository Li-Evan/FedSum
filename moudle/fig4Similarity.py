import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def transpose(matrix):
    transposed = np.array(matrix).T
    return transposed


log_list = [
    '''bert_similarity_list = [tensor(1.), tensor(1.0000), tensor(0.9995), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.)] ;clf_similarity_list = [tensor(0.9993)''',
    '''bert_similarity_list = [tensor(1.), tensor(1.0000), tensor(0.9994), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.)] ;clf_similarity_list = [tensor(0.9993)''',
    '''bert_similarity_list = [tensor(1.), tensor(1.0000), tensor(0.9992), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.)] ;clf_similarity_list = [tensor(0.9992)''',
    '''bert_similarity_list = [tensor(1.0000), tensor(1.0000), tensor(0.9993), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.)] ;clf_similarity_list = [tensor(0.9992)''',
    '''[tensor(1.), tensor(1.0000), tensor(0.9993), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.), tensor(1.), tensor(1.), tensor(1.), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.0000), tensor(1.)] ;clf_similarity_list = [tensor(0.9992)'''
]
data, new_data = [], []
for i in range(len(log_list)):
    log_list[i] = log_list[i].replace(" ;clf_similarity_list = ", ",").replace("bert_similarity_list = [", "").replace(
        "tensor(", "").replace("[", "").replace("]", "").replace(")", "")
    new_list = log_list[i].split(",")
    for j in range(len(new_list)):
        new_list[j] = float(new_list[j])
    data.append(new_list)

cosine_similarity_matrix = transpose(data)

# 将矩阵拆分成8个25x5的子矩阵
submatrices = np.split(cosine_similarity_matrix, 8)

# 计算整个矩阵的最大值和最小值，用于归一化
vmin = cosine_similarity_matrix.min()
vmax = cosine_similarity_matrix.max()

# 创建一个新的图形窗口，并设置其大小
plt.figure(figsize=(10, 10))

# 遍历每个子矩阵，并绘制对应的热图
for i, submatrix in enumerate(submatrices):
    # 计算子图的位置
    row = i // 2
    col = i % 2

    # 创建一个子图
    ax = plt.subplot(2, 4, i + 1)

    # 绘制热图，使用相同的vmin和vmax进行归一化
    sns.heatmap(submatrix, cmap='coolwarm', annot=False, vmin=vmin, vmax=vmax)

    # 设置子图的标题
    ax.set_title(f'Submatrix {i + 1}')

    # 设置x轴标签（仅在第一行的子图）
    if row == 0:
        ax.set_xlabel('Feature')
    else:
        ax.set_xlabel('')

    # 设置y轴标签（显示正确的索引）
    start_index = i * 25 + 1
    end_index = start_index + submatrix.shape[0]
    ax.set_yticks(range(1, submatrix.shape[0] + 1))
    ax.set_yticklabels(range(start_index, end_index), rotation=0)

    # 如果不是第一列的子图，移除y轴标签
    if col != 0:
        ax.yaxis.set_label_position('right')
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelright=True)

    # 显示颜色条
# plt.colorbar(ax=plt.gca(), label='Cosine Similarity')

# 显示整个图形
plt.tight_layout()  # 调整子图间距
plt.show()