import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ==========================
# 0. 设置 Excel 文件路径
# ==========================
# 请在此处修改 Excel 文件路径
filepath = "data.xlsx"

# ==========================
# 1. 导入 Excel 数据
# ==========================
def load_excel_data(filepath, x_col=0, y_col=1):
    """
    从 Excel 文件中加载二维数据
    :param filepath: Excel 文件路径 (.xlsx)
    :param x_col: X列索引或列名
    :param y_col: Y列索引或列名
    :return: numpy.ndarray, shape (n_samples, 2)
    """
    df = pd.read_excel(filepath)
    if isinstance(x_col, int) and isinstance(y_col, int):
        data = df.iloc[:, [x_col, y_col]].values
    else:
        data = df[[x_col, y_col]].values
    print(f"数据加载完成，共 {data.shape[0]} 个样本")
    return data

# ==========================
# 2. 执行 Fuzzy C-Means 聚类
# ==========================
def run_fcm(data, n_clusters=3, m=2.0, error=1e-5, maxiter=1000, random_state=42):
    """
    对二维数据执行模糊C均值聚类 (FCM)
    :param data: 输入数据 (n_samples, 2)
    :param n_clusters: 聚类数量
    :param m: 模糊系数 (>1, 通常取2)
    :param error: 迭代终止条件
    :param maxiter: 最大迭代次数
    :return: 聚类标签, 聚类中心, 隶属度矩阵
    """
    np.random.seed(random_state)
    data_T = data.T  # skfuzzy要求输入为 (features, samples)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=random_state
    )
    labels = np.argmax(u, axis=0)  # 隶属度最大者为所属类别
    return labels, cntr, u

# ==========================
# 3. 聚类效果评价
# ==========================
def evaluate_clustering(data, labels):
    """
    计算轮廓系数(SC)与Davies-Bouldin指数(DBI)
    """
    sc = silhouette_score(data, labels)
    dbi = davies_bouldin_score(data, labels)
    print(f" Silhouette Coefficient (SC): {sc:.4f}")
    print(f" Davies-Bouldin Index (DBI): {dbi:.4f}")
    return sc, dbi

# ==========================
# 4. 可视化结果
# ==========================
def plot_clusters(data, labels, centers, title="Fuzzy C-Means 2D Clustering"):
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centers')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================
# 运行聚类分析
# ==========================
data = load_excel_data(filepath, x_col=0, y_col=1)
n_clusters = 4  # 可以修改聚类数
labels, centers, u = run_fcm(data, n_clusters=n_clusters)
evaluate_clustering(data, labels)
plot_clusters(data, labels, centers, title=f"Fuzzy C-Means Clustering (k={n_clusters})")
