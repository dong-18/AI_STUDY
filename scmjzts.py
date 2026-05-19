"""
用前馈神经网络拟合主题赛机器人的积分
人员下标改为 Embedding，并加入归一化
改造内容：
1. 从 txt 文件读取数据
2. 使用 DataLoader 分批次训练
3. 加入训练/验证集划分
4. 加入早停机制
5. 保存最优模型
6. 将 BatchNorm 改为 LayerNorm
"""

import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import Dataset, DataLoader

# 判断当前机器是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 固定随机数种子
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证 cuDNN 尽量使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# -----------------------------
# 0) 归一化工具
# -----------------------------
class FeatureNormalizer:
    def __init__(self, max_difficulty=3.0, max_time_ms=30 * 24 * 60 * 60 * 1000):
        """
        max_difficulty: 难度最大值，按业务可调整
        max_time_ms: 时间最大值，默认 30 天的毫秒数
        """
        self.max_difficulty = float(max_difficulty)
        self.max_time_ms = float(max_time_ms)
        self.y_max = None

    def normalize_numeric_x(self, raw_x):
        """
        raw_x: shape = [batch_size, 3]
               每行是 [难度, 人员下标, 时间毫秒]

        返回:
        numeric_x: shape = [batch_size, 2]
                   对应 [归一化难度, 归一化时间]
        person_idx: shape = [batch_size]
        """
        difficulty = raw_x[:, 0:1] / self.max_difficulty
        time_ms = raw_x[:, 2:3] / self.max_time_ms

        numeric_x = torch.cat([difficulty, time_ms], dim=1)
        person_idx = raw_x[:, 1].long()

        return numeric_x, person_idx

    def fit_target(self, y):
        """
        只用训练集来拟合目标值归一化比例，避免数据泄漏
        """
        self.y_max = float(y.max().item())
        if self.y_max <= 0:
            self.y_max = 1.0

    def normalize_target(self, y):
        if self.y_max is None:
            raise ValueError("请先调用 fit_target(y_train) 再进行目标归一化")
        return y / self.y_max

    def denormalize_target(self, y_norm):
        if self.y_max is None:
            raise ValueError("normalizer.y_max 尚未设置")
        return y_norm * self.y_max


# -----------------------------
# 1) 从 txt 读取原始数据
# -----------------------------
def load_txt_data(file_path, sep=r"\s+"):
    """
    读取 txt 文件
    默认按空白符分隔：空格 / tab 都可以

    文件格式示例：
    difficulty person_idx time_ms score
    1 0 10000 100
    2 1 500000 180
    ...
    """
    df = pd.read_csv(file_path, sep=sep, engine="python")

    required_cols = ["difficulty", "person_idx", "time_ms", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"txt 文件缺少字段: {col}")

    raw_x = torch.tensor(
        df[["difficulty", "person_idx", "time_ms"]].values,
        dtype=torch.float32
    )
    y = torch.tensor(
        df[["score"]].values,
        dtype=torch.float32
    )

    return raw_x, y


# -----------------------------
# 2) 自定义数据集
# -----------------------------
class RobotScoreDataset(Dataset):
    def __init__(self, numeric_x, person_idx, y_norm):
        self.numeric_x = numeric_x
        self.person_idx = person_idx
        self.y = y_norm

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.numeric_x[idx], self.person_idx[idx], self.y[idx]


# -----------------------------
# 3) 定义带 Embedding 的前馈神经网络
#    已将 BatchNorm 改为 LayerNorm
# -----------------------------
class FNNWithEmbedding(nn.Module):
    def __init__(
        self,
        num_numeric_features,
        num_persons,
        person_embed_dim,
        hidden_dim,
        output_dim,
        dropout=0.1
    ):
        super(FNNWithEmbedding, self).__init__()

        self.person_embedding = nn.Embedding(
            num_embeddings=num_persons,
            embedding_dim=person_embed_dim
        )

        input_dim = num_numeric_features + person_embed_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, numeric_x, person_idx):
        person_emb = self.person_embedding(person_idx)
        x = torch.cat([numeric_x, person_emb], dim=1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.fc3(x)


# -----------------------------
# 4) FNN 回归智能体
# -----------------------------
class FNNAgent:
    def __init__(
        self,
        num_numeric_features,
        num_persons,
        person_embed_dim,
        out_dim,
        hidden_dim=128,
        lr=0.001,
        weight_decay=1e-5,
        dropout=0.1
    ):
        self.model = FNNWithEmbedding(
            num_numeric_features=num_numeric_features,
            num_persons=num_persons,
            person_embed_dim=person_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            dropout=dropout
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = nn.MSELoss()

    def train_batch(self, numeric_x, person_idx, output_value):
        self.model.train()

        numeric_x = numeric_x.to(device)
        person_idx = person_idx.to(device)
        output_value = output_value.to(device)

        pred = self.model(numeric_x, person_idx)
        loss = self.criterion(pred, output_value)

        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，提升训练稳定性
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.optimizer.step()

        return loss.item()

    def evaluate_batch(self, numeric_x, person_idx, output_value):
        self.model.eval()

        numeric_x = numeric_x.to(device)
        person_idx = person_idx.to(device)
        output_value = output_value.to(device)

        with torch.no_grad():
            pred = self.model(numeric_x, person_idx)
            loss = self.criterion(pred, output_value)

        return loss.item()

    def predict(self, numeric_x, person_idx):
        self.model.eval()

        numeric_x = numeric_x.to(device)
        person_idx = person_idx.to(device)

        with torch.no_grad():
            return self.model(numeric_x, person_idx)

    def save_checkpoint(self, model_path, config, normalizer):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": config,
            "normalizer": {
                "max_difficulty": normalizer.max_difficulty,
                "max_time_ms": normalizer.max_time_ms,
                "y_max": normalizer.y_max
            }
        }
        torch.save(checkpoint, model_path)

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        return checkpoint


# -----------------------------
# 5) 训练与验证
# -----------------------------
def evaluate_loader(agent, dataloader):
    total_loss = 0.0

    for batch_numeric_x, batch_person_idx, batch_y in dataloader:
        loss = agent.evaluate_batch(batch_numeric_x, batch_person_idx, batch_y)
        total_loss += loss

    return total_loss / len(dataloader)


def fnn_train(
    agent,
    train_loader,
    val_loader,
    normalizer,
    config,
    episodes=1000,
    patience=100,
    model_path="best_fnn_model.pth"
):
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_state_dict = None
    wait = 0

    for episode in range(episodes):
        total_train_loss = 0.0

        for batch_numeric_x, batch_person_idx, batch_y in train_loader:
            loss = agent.train_batch(batch_numeric_x, batch_person_idx, batch_y)
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate_loader(agent, val_loader)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(agent.model.state_dict())
            agent.save_checkpoint(model_path, config=config, normalizer=normalizer)
            wait = 0
        else:
            wait += 1

        if (episode + 1) % 50 == 0:
            print(
                f"Episode [{episode + 1}/{episodes}], "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

        if wait >= patience:
            print(f"早停触发：连续 {patience} 轮验证集未提升，停止训练。")
            break

    if best_state_dict is not None:
        agent.model.load_state_dict(best_state_dict)

    return train_loss_history, val_loss_history


# -----------------------------
# 6) 构建数据集和 DataLoader
# -----------------------------
def build_dataloaders(
    file_path,
    normalizer,
    batch_size=32,
    train_ratio=0.8,
    sep=r"\s+"
):
    raw_x, y = load_txt_data(file_path, sep=sep)

    numeric_x, person_idx = normalizer.normalize_numeric_x(raw_x)

    # 先划分训练/验证索引，再只用训练集拟合 y_max，避免数据泄漏
    total_size = len(y)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    y_train = y[train_indices]
    normalizer.fit_target(y_train)

    y_norm = normalizer.normalize_target(y)

    dataset = RobotScoreDataset(numeric_x, person_idx, y_norm)

    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_indices.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    num_persons = int(person_idx.max().item()) + 1

    return train_loader, val_loader, num_persons


# -----------------------------
# 7) 主程序
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # 文件路径
    # -----------------------------
    train_file = "train_data.txt"

    # -----------------------------
    # 归一化器
    # 注意：这里 max_time_ms 要和你的 time_ms 单位一致
    # 如果 time_ms 真的是“毫秒”，那 2 天应写成：
    # 2 * 24 * 60
    # -----------------------------
    normalizer = FeatureNormalizer(
        max_difficulty=3.0,
        max_time_ms=2 * 24 * 60
    )

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_loader, val_loader, num_persons = build_dataloaders(
        file_path=train_file,
        normalizer=normalizer,
        batch_size=32,
        train_ratio=0.8,
        sep=r"\s+"
    )

    # 配置参数
    num_numeric_features = 2
    person_embed_dim = 4
    out_dim = 1
    hidden_dim = 64
    dropout = 0.1

    config = {
        "num_numeric_features": num_numeric_features,
        "num_persons": num_persons,
        "person_embed_dim": person_embed_dim,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "dropout": dropout
    }

    agent = FNNAgent(
        num_numeric_features=num_numeric_features,
        num_persons=num_persons,
        person_embed_dim=person_embed_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        lr=0.001,
        weight_decay=1e-5,
        dropout=dropout
    )

    print("开始 FNN Embedding + LayerNorm 分批训练...")

    train_loss_history, val_loss_history = fnn_train(
        agent=agent,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=config,
        episodes=1000,
        patience=150,
        model_path="best_fnn_model.pth"
    )

    print("训练完成，最优模型已保存到 best_fnn_model.pth")

    # -----------------------------
    # 加载最优模型（可选）
    # -----------------------------
    if os.path.exists("best_fnn_model.pth"):
        agent.load_checkpoint("best_fnn_model.pth")

    # -----------------------------
    # 预测测试
    # -----------------------------
    test_raw_x = torch.tensor([
        [3, 519,2880]
    ], dtype=torch.float32)

    test_numeric_x, test_person_idx = normalizer.normalize_numeric_x(test_raw_x)

    pred_norm = agent.predict(test_numeric_x, test_person_idx)
    pred = normalizer.denormalize_target(pred_norm)

    print("归一化预测积分：", pred_norm.cpu().numpy())
    print("还原后预测积分：", pred.cpu().numpy())

    # 查看学到的人员 Embedding
    print("人员 Embedding：")
    print(agent.model.person_embedding.weight.data.cpu().numpy())

    # -----------------------------
    # 绘制训练/验证损失曲线
    # -----------------------------
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()