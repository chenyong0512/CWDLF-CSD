import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ==========================
# 0. 设置 Excel 文件路径
# ==========================
# 请在此处修改你的文件路径（例如 "data.xlsx"）
filepath = "data.xlsx"
# 如果需要指定列名，也可以写成：
# data = load_excel_data(filepath, x_col="Feature1", y_col="Feature2")

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
# 2. 执行 K-Means 聚类
# ==========================
def run_kmeans(data, n_clusters=3, random_state=42):
    """
    对二维数据执行K-Means聚类
    :param data: 输入数据 (n_samples, 2)
    :param n_clusters: 聚类数量
    :return: 聚类标签, 聚类中心
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return labels, centers

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
def plot_clusters(data, labels, centers, title="K-Means 2D Clustering"):
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
n_clusters = 4  # 你可以改成需要的聚类数
labels, centers = run_kmeans(data, n_clusters=n_clusters)
evaluate_clustering(data, labels)
plot_clusters(data, labels, centers, title=f"K-Means Clustering (k={n_clusters})")
