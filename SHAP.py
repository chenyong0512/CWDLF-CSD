import pandas as pd
import shap
import matplotlib.pyplot as plt
import torch

# ===============================
# 0. 配置
# ===============================
excel_path = "data_preprocessed.xlsx"  # 预处理后的 Excel 数据
target_col = "IPR"                      # 目标列
model_path = "dpstimesnet_model.pth"    # 已训练模型文件（PyTorch）

# ===============================
# 1. 导入数据
# ===============================
df = pd.read_excel(excel_path)
X = df.drop(columns=[target_col])
y = df[target_col].values

# ===============================
# 2. 加载训练好的模型
# ===============================
# 假设你的 DPSTimesNet 是 PyTorch 模型
from models.dpstimesnet import DPSTimesNet

input_dim = X.shape[1]
output_dim = 1  # 单输出，如果多输出可调整
model = DPSTimesNet(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# ===============================
# 3. 使用 SHAP 解释
# ===============================
# 使用 KernelExplainer 适合任何模型（慢）
explainer = shap.KernelExplainer(
    lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
    X.sample(50, random_state=42).values  # 背景样本
)

shap_values = explainer.shap_values(X.values)

# ===============================
# 4. 可视化特征重要性
# ===============================
# 特征条形图
shap.summary_plot(shap_values, X, plot_type="bar")

# 特征分布图
shap.summary_plot(shap_values, X)

# ===============================
# 5. 可选：保存 SHAP 值
# ===============================
shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df.to_excel("shap_values.xlsx", index=False)
print(" SHAP 值已保存到 shap_values.xlsx")
