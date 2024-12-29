import numpy as np
import matplotlib.pyplot as plt
import torch


def get_class_ave(samples, start, end):
    """
    计算某一类的均值
    :param samples: 所有输入数据
    :param start: 某一类的开始索引（包含）
    :param end: 某一类的结束索引（不包含）
    :return:
    """
    class_ave = 0.0
    for i in range(start, end):
        class_ave += samples[i]
    class_ave = class_ave / (end - start)
    return class_ave


def get_class_diameter(samples, start, end):
    """
    计算某一类的直径（类内各样本的差异）
    :param samples: 所有输入数据
    :param start: 某一类的开始索引（包含）
    :param end: 某一类的结束索引（不包含）
    :return:
    """
    class_diameter = 0.0
    class_ave = get_class_ave(samples, start, end)
    for i in range(start, end):
        class_diameter += (samples[i] - class_ave) ** 2
    return class_diameter


def get_split_loss(samples, sample_num, split_class_num):
    """
    计算得到不同样本划分为不同分类的loss矩阵
    :param samples: 样本
    :param sample_num: 样本数
    :param split_class_num: 最大分类数
    :return: 不同样本划分为不同分类的loss矩阵
    """
    # 记录不同样本数（1~sample_num）分成不同类（1~split_class_num）的loss值
    split_loss_result = np.zeros((sample_num + 1, split_class_num + 1))

    # 对于第一列k=1
    for n in range(1, sample_num + 1):
        # 将所有样本分成1类，直接调用函数get_class_diameter计算
        # 递推公式L(P(n,1))=D(1,n)，其中P(n,1)表示将n个样本分成1类的最优划分，D(1,n)表示所有样本的差异
        # 该式表示将n个样本分成1类的损失函数值L=该类的直径D（类内各样本的差异）
        split_loss_result[n, 1] = get_class_diameter(samples, 0, n)

    # 使用递推公式计算k>1时采取不同划分的损失函数值
    for k in range(2, split_class_num + 1):
        # n不能小于k
        for n in range(k, sample_num + 1):
            # 递推公式：L(P(n,k))=min{L(P(j-1,k-1))+D(j,n)}
            # 其中，k<=j<=n，要保证前面每一类都至少有一个样本（j>=k），最后一类至少有一个样本（j<=n）
            loss = []
            for j in range(k - 1, n):
                loss.append(split_loss_result[j, k - 1] + get_class_diameter(samples, j, n))
            split_loss_result[n, k] = min(loss)

    return split_loss_result


def get_split_info(samples, split_loss_result):
    """
    确定最佳分类数、分类点、各个样本的类别
    :param samples: 样本
    :param split_loss_result: loss矩阵
    :return: 最佳分类数、分类点、各个样本的类别
    """
    # 首先确定最优分类数k_best
    # 以增加一个分类后loss下降不足30%（需多次尝试不同值比较分类效果）作为阈值确定k_best
    loss_n_k = split_loss_result[-1, 1:]
    k_best = 1
    for i in range(len(loss_n_k) - 1):
        # 对于loss为0的情况，直接确定k_best
        if loss_n_k[i] == 0:
            k_best = i + 1
            break
        else:
            desc_rate = (loss_n_k[i] - loss_n_k[i + 1]) / loss_n_k[i]
            if desc_rate > 0.05:
                k_best = i + 2

    # 确定各个样本所属类别
    split_point = []
    k = k_best - 1
    n = len(samples)
    split_loss = split_loss_result[n][k_best]
    while k > 0:

        while n > 0:
            # 寻找类别划分点
            if split_loss_result[n][k] < split_loss:
                split_point.insert(0, n)
                split_loss = split_loss_result[n][k]
                break
            n -= 1
        k -= 1

    # 确定样本类别
    sample_class = []
    class_index = 1
    point_index = 0

    # split_point中补充完整
    # 举例：split_point = [6]，有9个样本，则补充完整的split_point = [1, 6, 9]
    split_point.insert(0, 1)
    split_point.append(len(samples))
    for i in range(len(samples)):
        if i < split_point[point_index + 1]:
            sample_class.append(class_index)
        else:
            point_index += 1
            class_index += 1
            sample_class.append(class_index)

    return k_best, split_point, sample_class

# RESNET56  CIFAR10: CKA_matrix_for_visualization_resnet56_cifar10.pth, CKA_matrix_for_visualization_cvgg16_bn_cifar10.pth
matrix = torch.load('save/CKA_matrix_for_visualization_tdanet.pth')
avg_matrix = torch.zeros((matrix[0].shape[0], matrix[0].shape[1]))

for i in range(len(matrix)):
    avg_matrix += matrix[i] 

avg_matrix = avg_matrix / len(matrix)
samples = torch.sum(avg_matrix, dim=0)
print(samples)

# 设置分类数k
k = 6
split_loss_result = get_split_loss(samples, len(samples), k)
# print(split_loss_result[-1, -1])

_, split_point, sample_class = get_split_info(samples, split_loss_result)
print(split_point)
print(sample_class)

# # 绘图（散点图）
# plt.scatter(list(range(len(samples))), samples, c=sample_class)
# plt.xlabel('K')
# plt.ylabel('Loss')
# plt.show()



