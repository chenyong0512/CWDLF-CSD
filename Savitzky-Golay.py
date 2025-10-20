import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ===============================
# 0. Excel 文件路径
# ===============================
filepath = "data.xlsx"          #  修改为你的 Excel 文件路径
output_path = "data_sg_smoothed.xlsx"  #  保存路径

# ===============================
# 1. 导入 Excel 数据
# ===============================
df = pd.read_excel(filepath)
print(f" 数据加载完成：{filepath} | 形状 = {df.shape}")

# ===============================
# 2. S-G 滤波平滑处理
# ===============================
# 参数设置
window = 11      # 窗口长度，必须为奇数
polyorder = 2    # 多项式阶数，通常 < window

df_smooth = df.copy()
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        # 如果数据点少于窗口，跳过平滑
        if len(df[col]) < window:
            print(f"⚠️ 列 {col} 样本少于窗口长度，跳过平滑")
            continue
        df_smooth[col] = savgol_filter(df[col], window_length=window, polyorder=polyorder)

print(f" S-G 滤波处理完成 (window={window}, polyorder={polyorder})")

# ===============================
# 3. 保存平滑后的数据
# ===============================
df_smooth.to_excel(output_path, index=False)
print(f" 平滑后的数据已保存到：{output_path}")
