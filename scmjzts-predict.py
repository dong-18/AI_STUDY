# 文件名：scmjzts-predict.py

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# 1) 归一化工具
# =============================
class FeatureNormalizer:
    def __init__(self, max_difficulty=3.0, max_time_ms=30 * 24 * 60 * 60 * 1000):
        self.max_difficulty = float(max_difficulty)
        self.max_time_ms = float(max_time_ms)
        self.y_max = None

    def normalize_numeric_x(self, raw_x):
        difficulty = raw_x[:, 0:1] / self.max_difficulty
        time_ms = raw_x[:, 2:3] / self.max_time_ms
        numeric_x = torch.cat([difficulty, time_ms], dim=1)
        person_idx = raw_x[:, 1].long()
        return numeric_x, person_idx

    def denormalize_target(self, y_norm):
        if self.y_max is None:
            raise ValueError("normalizer.y_max 尚未设置")
        return y_norm * self.y_max

    def load_state_dict(self, state):
        self.max_difficulty = float(state["max_difficulty"])
        self.max_time_ms = float(state["max_time_ms"])
        self.y_max = float(state["y_max"]) if state["y_max"] is not None else None


# =============================
# 2) 模型定义
# =============================
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
        super().__init__()

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

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


# =============================
# 3) 预测器
# =============================
class Predictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        if "config" not in checkpoint:
            raise ValueError("模型文件中缺少 config，无法自动构建模型")
        if "normalizer" not in checkpoint:
            raise ValueError("模型文件中缺少 normalizer，无法自动还原预测值")

        config = checkpoint["config"]

        self.model = FNNWithEmbedding(
            num_numeric_features=config["num_numeric_features"],
            num_persons=config["num_persons"],
            person_embed_dim=config["person_embed_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["out_dim"],
            dropout=config["dropout"]
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.normalizer = FeatureNormalizer()
        self.normalizer.load_state_dict(checkpoint["normalizer"])

        self.num_persons = config["num_persons"]

    def predict_one(self, difficulty, person_idx, time_ms):
        if person_idx < 0 or person_idx >= self.num_persons:
            raise ValueError(
                f"person_idx={person_idx} 超出范围，合法范围是 [0, {self.num_persons - 1}]"
            )

        raw_x = torch.tensor(
            [[difficulty, person_idx, time_ms]],
            dtype=torch.float32
        )

        numeric_x, person_idx_tensor = self.normalizer.normalize_numeric_x(raw_x)
        numeric_x = numeric_x.to(device)
        person_idx_tensor = person_idx_tensor.to(device)

        with torch.no_grad():
            pred_norm = self.model(numeric_x, person_idx_tensor)
            pred = self.normalizer.denormalize_target(pred_norm)

        return pred_norm.cpu().numpy(), pred.cpu().numpy()

    def predict_batch(self, samples):
        """
        samples: list of [difficulty, person_idx, time_ms]
        """
        raw_x = torch.tensor(samples, dtype=torch.float32)

        person_ids = raw_x[:, 1].long()
        if torch.any(person_ids < 0) or torch.any(person_ids >= self.num_persons):
            raise ValueError(
                f"存在超范围的 person_idx，合法范围是 [0, {self.num_persons - 1}]"
            )

        numeric_x, person_idx_tensor = self.normalizer.normalize_numeric_x(raw_x)
        numeric_x = numeric_x.to(device)
        person_idx_tensor = person_idx_tensor.to(device)

        with torch.no_grad():
            pred_norm = self.model(numeric_x, person_idx_tensor)
            pred = self.normalizer.denormalize_target(pred_norm)

        return pred_norm.cpu().numpy(), pred.cpu().numpy()


# =============================
# 4) 主函数
# =============================
def main():
    parser = argparse.ArgumentParser(description="FNN + Embedding 积分预测脚本")
    parser.add_argument("--model", type=str, default="best_fnn_model.pth", help="模型文件路径")
    parser.add_argument("--difficulty", type=float, required=True, help="难度")
    parser.add_argument("--person_idx", type=int, required=True, help="人员下标")
    parser.add_argument("--time_ms", type=float, required=True, help="时间分值")

    args = parser.parse_args()

    predictor = Predictor(args.model)

    pred_norm, pred = predictor.predict_one(
        difficulty=args.difficulty,
        person_idx=args.person_idx,
        time_ms=args.time_ms
    )

    print("模型文件：", args.model)
    print("输入：")
    print(f"  difficulty = {args.difficulty}")
    print(f"  person_idx = {args.person_idx}")
    print(f"  time_ms    = {args.time_ms}")
    print("输出：")
    print("  归一化预测积分 =", pred_norm)
    print("  还原后预测积分 =", pred)


if __name__ == "__main__":
    main()