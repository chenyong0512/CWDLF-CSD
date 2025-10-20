import pandas as pd
import numpy as np

# ===============================
# 0. Excel 文件路径
# ===============================
filepath = "data.xlsx"  # 修改为你的 Excel 文件路径
output_path = "data_cleaned_boxplot.xlsx"  # 保存路径

# ===============================
# 1. 导入 Excel 数据
# ===============================
df = pd.read_excel(filepath)
print(f" 数据加载完成：{filepath} | 形状 = {df.shape}")

# ===============================
# 2. 箱型图法异常值处理
# ===============================
df_cleaned = df.copy()
for col in df.columns:
    if not np.issubdtype(df[col].dtype, np.number):
        continue  # 跳过非数值列

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # 标记异常值为 NaN
    df_cleaned[col] = np.where(df[col].between(lower, upper), df[col], np.nan)
    # 线性插值补全
    df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')

    print(f" {col}: 异常值区间 <{lower:.2f}, {upper:.2f}>")

print(" 异常值处理完成（箱型图法 + 插值）")

# ===============================
# 3. 保存清洗后的数据
# ===============================
df_cleaned.to_excel(output_path, index=False)
print(f" 清洗后的数据已保存到：{output_path}")
