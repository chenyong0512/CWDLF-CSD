import pandas as pd
import numpy as np
from minepy import MINE
from scipy.stats import pearsonr
import itertools

# ===============================
# 0. Excel 文件路径和目标列
# ===============================
filepath = "data.xlsx"        #  修改为你的 Excel 文件路径
target_col = "IPR"            #  修改为目标列，例如 "IPR" 或 "TMP"
output_path = "data_mic_pcc_selected.xlsx"  #  保存路径

# ===============================
# 1. 导入 Excel 数据
# ===============================
df = pd.read_excel(filepath)
print(f" 数据加载完成：{filepath} | 形状 = {df.shape}")

# ===============================
# 2. 第一阶段：MIC >= 0.2
# ===============================
mine = MINE()
mic_scores = {}

for col in df.columns:
    if col == target_col:
        continue
    mine.compute_score(df[col], df[target_col])
    mic_scores[col] = mine.mic()

mic_df = pd.DataFrame(list(mic_scores.items()), columns=["Feature", "MIC"])
mic_selected = mic_df[mic_df["MIC"] >= 0.2]["Feature"].tolist()
print(f" 阶段 1 (MIC >= 0.2) 保留 {len(mic_selected)} 个特征: {mic_selected}")

# ===============================
# 3. 第二阶段：PCC 去除高度线性相关特征
# ===============================
remove_features = set()
for f1, f2 in itertools.combinations(mic_selected, 2):
    pcc, _ = pearsonr(df[f1], df[f2])
    if abs(pcc) >= 0.8:
        remove_features.add(f2)  # 默认删除组合中的第二个

final_features = [f for f in mic_selected if f not in remove_features]
print(f" 阶段 2 (PCC < 0.8) 最终保留 {len(final_features)} 个特征: {final_features}")

# ===============================
# 4. 保存筛选后的数据
# ===============================
df_selected = df[final_features + [target_col]]
df_selected.to_excel(output_path, index=False)
print(f" 筛选后的数据已保存到：{output_path}")
