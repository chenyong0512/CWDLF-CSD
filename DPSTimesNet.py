import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------
# 1. 使用示例 & Excel 数据导入
# ------------------------------
excel_path = "your_data.xlsx"  # 替换为你的 Excel 文件路径
seq_len = 24

# 构建 Dataset
class ExcelTimeSeriesDataset(Dataset):
    def __init__(self, excel_path, seq_len=24):
        """
        读取 Excel 文件，前8列作为输入，后2列作为输出
        """
        df = pd.read_excel(excel_path)
        self.data = df.values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len, :8]  # 输入特征
        y = self.data[idx+self.seq_len, 8:10]   # 输出特征
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 创建 Dataset
dataset = ExcelTimeSeriesDataset(excel_path, seq_len=seq_len)

# ------------------------------
# 2. ProbSparse 自注意力
# ------------------------------
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, top_k_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.top_k_ratio = top_k_ratio
        self.qkv = nn.Linear(dim, dim*3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        attn_scores = torch.einsum('blhd,bshd->bhls', q, k) * self.scale

        top_k = max(1, int(self.top_k_ratio * L))
        top_scores, top_idx = torch.topk(attn_scores, top_k, dim=-1)
        mask = torch.zeros_like(attn_scores).scatter_(-1, top_idx, 1)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bhls,bshd->blhd', attn_probs, v)
        out = out.reshape(B, L, C)
        return self.out(out)

# ------------------------------
# 3. FFT 注意力调制
# ------------------------------
class FFTAttentionModulation(nn.Module):
    def __init__(self, dim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.modulation = nn.Parameter(torch.ones(dim) * alpha)

    def forward(self, x):
        fft_x = torch.fft.fft(x, dim=1).real
        return x * (1 + self.modulation * fft_x)

# ------------------------------
# 4. DPSTimesNet 模型
# ------------------------------
class DPSTimesNet(nn.Module):
    def __init__(self, input_dim=8, output_dim=2, d_model=64, d_fcn=32, n_heads=4, n_layers=2, alpha=0.5, top_k_ratio=0.25):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                FFTAttentionModulation(d_model, alpha=alpha),
                ProbSparseSelfAttention(d_model, n_heads=n_heads, top_k_ratio=top_k_ratio)
            )
            for _ in range(n_layers)
        ])
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_fcn),
            nn.ReLU(),
            nn.Linear(d_fcn, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x) + x
        return self.fcn(x[:, -1, :])

# ------------------------------
# 5. KOA 超参数优化
# ------------------------------
def kepler_optimize(dataset, n_agents=5, n_iterations=3, seq_len=24):
    # 搜索空间
    d_model_space = [32, 48, 64, 80, 96, 112, 128]
    d_fcn_space = [16, 32, 48, 64, 80, 96, 112, 128]
    n_layers_space = [1,2,3,4]
    alpha_space = [0.1,0.3,0.5,0.7,0.9,1.0]
    top_k_ratio_space = [0.1,0.2,0.3,0.4,0.5]
    lr_space = [1e-4,5e-4,1e-3,5e-3,1e-2]

    # 数据集划分
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化 agent
    agents = []
    for _ in range(n_agents):
        agent = {
            'd_model': random.choice(d_model_space),
            'd_fcn': random.choice(d_fcn_space),
            'n_layers': random.choice(n_layers_space),
            'alpha': random.choice(alpha_space),
            'top_k_ratio': random.choice(top_k_ratio_space),
            'lr': random.choice(lr_space),
            'fitness': float('inf')
        }
        agents.append(agent)

    for it in range(n_iterations):
        print(f"\nKOA Iteration {it+1}/{n_iterations}")
        for agent in agents:
            model = DPSTimesNet(
                d_model=agent['d_model'],
                d_fcn=agent['d_fcn'],
                n_layers=agent['n_layers'],
                alpha=agent['alpha'],
                top_k_ratio=agent['top_k_ratio']
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=agent['lr'])
            criterion = nn.MSELoss()

            # 训练 1 epoch 快速评估
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

            # 验证集评估
            model.eval()
            y_true, y_pred_list = [], []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    y_pred_val = model(x_val)
                    y_true.append(y_val.numpy())
                    y_pred_list.append(y_pred_val.numpy())
            y_true = np.vstack(y_true)
            y_pred_list = np.vstack(y_pred_list)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_list))
            agent['fitness'] = rmse
            print(f"Agent: {agent}, RMSE={rmse:.4f}")

        # 排序并变异
        agents.sort(key=lambda a: a['fitness'])
        top_agent = agents[0]
        for agent in agents[1:]:
            agent['d_model'] = random.choice(d_model_space)
            agent['d_fcn'] = random.choice(d_fcn_space)
            agent['n_layers'] = random.choice(n_layers_space)
            agent['alpha'] = random.choice(alpha_space)
            agent['top_k_ratio'] = random.choice(top_k_ratio_space)
            agent['lr'] = random.choice(lr_space)

    best_agent = agents[0]
    print("\nBest Hyperparameters found by KOA:")
    print(best_agent)

    # 最终训练
    final_model = DPSTimesNet(
        d_model=best_agent['d_model'],
        d_fcn=best_agent['d_fcn'],
        n_layers=best_agent['n_layers'],
        alpha=best_agent['alpha'],
        top_k_ratio=best_agent['top_k_ratio']
    )
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_agent['lr'])
    criterion = nn.MSELoss()
    for epoch in range(10):
        final_model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = final_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Final Training Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 最终评估指标
    final_model.eval()
    y_true, y_pred_list = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            y_pred_val = final_model(x_val)
            y_true.append(y_val.numpy())
            y_pred_list.append(y_pred_val.numpy())
    y_true = np.vstack(y_true)
    y_pred_list = np.vstack(y_pred_list)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_list))
    mae = mean_absolute_error(y_true, y_pred_list)
    mape = np.mean(np.abs((y_true - y_pred_list)/y_true)) * 100
    r2 = r2_score(y_true, y_pred_list)
    print("\nEvaluation Metrics on Validation Set:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

    return final_model

