import pandas as pd
from minepy import MINE

# ===============================
# 0. Excel 文件路径和目标列
# ===============================
filepath = "data.xlsx"        #  修改为你的 Excel 文件路径
target_col = "IPR"            #  修改为目标列，例如 "IPR" 或 "TMP"
output_path = "data_mic_selected.xlsx"  #  保存路径

# ===============================
# 1. 导入 Excel 数据
# ===============================
df = pd.read_excel(filepath)
print(f" 数据加载完成：{filepath} | 形状 = {df.shape}")

# ===============================
# 2. MIC 特征选择
# ===============================
mine = MINE()
mic_scores = {}

for col in df.columns:
    if col == target_col:
        continue
    mine.compute_score(df[col], df[target_col])
    mic_scores[col] = mine.mic()

# 转为 DataFrame 并筛选 MIC >= 0.2
mic_df = pd.DataFrame(list(mic_scores.items()), columns=["Feature", "MIC"])
selected_features = mic_df[mic_df["MIC"] >= 0.2]["Feature"].tolist()
print(f" {len(selected_features)} 个特征被保留 (MIC >= 0.2):")
print(selected_features)

# ===============================
# 3. 保存筛选后的数据
# ===============================
df_selected = df[selected_features + [target_col]]
df_selected.to_excel(output_path, index=False)
print(f" 筛选后的数据已保存到：{output_path}")
