import random  # 导入随机数模块，用于可能的随机操作
from dataclasses import dataclass  # 导入 dataclass，用于定义简单的数据结构

import numpy as np  # 导入 numpy，用于棋盘和数值计算
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.distributions import Categorical  # 导入离散分布，用于策略采样动作


class GomokuEnv:  # 定义五子棋环境类
    def __init__(self, board_size=9, win_length=5):  # 初始化环境，默认棋盘 9x9，5 子连线获胜
        self.board_size = board_size  # 保存棋盘边长
        self.win_length = win_length  # 保存获胜所需连子数
        self.action_size = board_size * board_size  # 动作总数，等于棋盘格子总数
        self.reset()  # 初始化棋盘状态

    def reset(self):  # 重置环境
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # 创建空棋盘，0 表示空位
        self.current_player = 1  # 当前玩家，1 表示先手，-1 表示后手
        self.done = False  # 对局是否结束
        self.winner = 0  # 胜者，1/-1/0 分别表示先手/后手/平局或未结束
        return self.get_obs()  # 返回初始观察

    def get_obs(self):  # 获取当前观察状态
        cur = (self.board == self.current_player).astype(np.float32)  # 当前玩家自己的棋子位置，转为 float32
        opp = (self.board == -self.current_player).astype(np.float32)  # 对手棋子位置，转为 float32
        player_plane = np.ones_like(cur, dtype=np.float32)  # 当前玩家平面，全 1，用于提示当前视角
        obs = np.stack([cur, opp, player_plane], axis=0)  # 按通道堆叠成 3 x H x W 的输入
        return obs  # 返回观察

    def legal_actions_mask(self):  # 获取合法动作掩码
        mask = (self.board.reshape(-1) == 0).astype(np.float32)  # 展平棋盘，空位为 1，非空位为 0
        return mask  # 返回合法动作掩码

    def step(self, action):  # 执行动作
        if self.done:  # 如果对局已经结束
            raise ValueError("Game is already done.")  # 抛出异常，禁止继续走子

        r, c = divmod(action, self.board_size)  # 将一维动作索引映射为二维坐标

        if self.board[r, c] != 0:  # 如果该位置已经有子，属于非法动作
            self.done = True  # 直接终局
            self.winner = -self.current_player  # 当前玩家非法落子，判对手获胜
            reward = -1.0  # 给当前执行动作的一方负奖励
            return self.get_obs(), reward, self.done, {"illegal": True}  # 返回结果

        self.board[r, c] = self.current_player  # 在棋盘对应位置落子

        if self.check_win(r, c, self.current_player):  # 检查当前落子后是否获胜
            self.done = True  # 标记终局
            self.winner = self.current_player  # 记录胜者
            reward = 1.0  # 给当前落子方正奖励
            return self.get_obs(), reward, self.done, {"illegal": False}  # 返回结果

        if np.all(self.board != 0):  # 如果棋盘已满且没人获胜
            self.done = True  # 标记终局
            self.winner = 0  # 平局
            reward = 0.0  # 平局奖励为 0
            return self.get_obs(), reward, self.done, {"illegal": False}  # 返回结果

        self.current_player *= -1  # 切换玩家
        reward = 0.0  # 非终局时即时奖励为 0
        return self.get_obs(), reward, self.done, {"illegal": False}  # 返回下一状态

    def check_win(self, r, c, player):  # 检查 player 在位置 (r, c) 落子后是否形成连线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向：竖、横、主对角、副对角
        for dr, dc in directions:  # 遍历四个方向
            count = 1  # 当前落子本身算 1 个
            count += self.count_direction(r, c, dr, dc, player)  # 正方向统计连续棋子数
            count += self.count_direction(r, c, -dr, -dc, player)  # 反方向统计连续棋子数
            if count >= self.win_length:  # 如果连续数达到获胜要求
                return True  # 返回获胜
        return False  # 所有方向都不满足则未获胜

    def count_direction(self, r, c, dr, dc, player):  # 统计从 (r, c) 沿某方向连续的 player 棋子数
        cnt = 0  # 初始化计数器
        rr, cc = r + dr, c + dc  # 先走一步到目标方向
        while 0 <= rr < self.board_size and 0 <= cc < self.board_size and self.board[rr, cc] == player:  # 只要不越界且棋子属于当前玩家
            cnt += 1  # 连续数加一
            rr += dr  # 继续沿行方向前进
            cc += dc  # 继续沿列方向前进
        return cnt  # 返回该方向连续数量


class PolicyValueNet(nn.Module):  # 定义策略-价值网络
    def __init__(self, board_size=9):  # 初始化网络
        super().__init__()  # 调用父类初始化
        self.board_size = board_size  # 保存棋盘尺寸
        self.action_size = board_size * board_size  # 动作维度

        self.encoder = nn.Sequential(  # 定义共享特征提取网络
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 第一层卷积，输入 3 通道，输出 64 通道
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第二层卷积
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第三层卷积
            nn.ReLU(),  # 激活函数
        )  # 编码器结束

        self.policy_head = nn.Sequential(  # 定义策略头，输出每个动作的 logits
            nn.Conv2d(64, 2, kernel_size=1),  # 用 1x1 卷积降通道到 2
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(2 * board_size * board_size, self.action_size),  # 映射到动作空间大小
        )  # 策略头结束

        self.value_head = nn.Sequential(  # 定义价值头，输出状态价值
            nn.Conv2d(64, 1, kernel_size=1),  # 用 1x1 卷积降通道到 1
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 展平
            nn.Linear(board_size * board_size, 64),  # 全连接到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 1),  # 输出单个标量价值
        )  # 价值头结束

    def forward(self, x):  # 前向传播
        feat = self.encoder(x)  # 提取共享特征
        logits = self.policy_head(feat)  # 计算策略 logits
        value = self.value_head(feat).squeeze(-1)  # 计算状态价值并去掉最后一维
        return logits, value  # 返回动作 logits 和状态价值

    def get_action_and_value(self, obs, legal_mask, action=None):  # 同时获取动作、对数概率、熵和价值
        logits, value = self(obs)  # 先前向计算网络输出

        masked_logits = logits.masked_fill(legal_mask < 0.5, -1e9)  # 将非法动作的 logits 置成极小值
        dist = Categorical(logits=masked_logits)  # 用 masked logits 构建离散动作分布

        if action is None:  # 如果没有给定动作
            action = dist.sample()  # 按策略采样一个动作

        logprob = dist.log_prob(action)  # 计算该动作的对数概率
        entropy = dist.entropy()  # 计算策略熵，用于鼓励探索
        return action, logprob, entropy, value  # 返回动作、logprob、熵、价值


@dataclass  # 使用 dataclass 简化数据结构定义
class Transition:  # 定义单步样本结构
    obs: np.ndarray  # 状态观察
    action: int  # 执行动作
    logprob: float  # 旧策略下该动作的对数概率
    reward: float  # 奖励
    done: bool  # 该样本是否为轨迹终点
    value: float  # 当时网络预测的状态价值
    legal_mask: np.ndarray  # 合法动作掩码
    player: int  # 当前样本所属玩家


def compute_gae(transitions, gamma=0.99, gae_lambda=0.95):  # 计算 GAE 优势和 returns
    rewards = np.array([t.reward for t in transitions], dtype=np.float32)  # 提取奖励数组
    dones = np.array([t.done for t in transitions], dtype=np.float32)  # 提取终止标记数组
    values = np.array([t.value for t in transitions], dtype=np.float32)  # 提取状态价值数组

    advantages = np.zeros_like(rewards, dtype=np.float32)  # 初始化优势数组
    lastgaelam = 0.0  # 记录递推的上一步 GAE 值

    next_values = np.append(values[1:], 0.0)  # 下一个状态价值，最后一个补 0

    for t in reversed(range(len(transitions))):  # 从后往前递推
        nonterminal = 1.0 - dones[t]  # 如果当前为终点，则 nonterminal 为 0
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]  # TD 误差
        lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam  # GAE 递推公式
        advantages[t] = lastgaelam  # 保存当前时刻优势

    returns = advantages + values  # 回报 = 优势 + 价值估计
    return advantages, returns  # 返回优势和目标价值


def self_play_episode(env, model, device):  # 进行一局自对弈并收集数据
    trajectories = {1: [], -1: []}  # 分别保存先手和后手的轨迹

    obs = env.reset()  # 重置环境并拿到初始观察

    while not env.done:  # 只要对局未结束就继续下
        player = env.current_player  # 记录当前玩家
        legal_mask = env.legal_actions_mask()  # 获取当前合法动作掩码

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # 状态转 tensor 并加 batch 维
        legal_tensor = torch.tensor(legal_mask, dtype=torch.float32, device=device).unsqueeze(0)  # mask 转 tensor 并加 batch 维

        with torch.no_grad():  # 推理阶段不计算梯度
            action, logprob, _, value = model.get_action_and_value(obs_tensor, legal_tensor)  # 获取动作和价值

        action = action.item()  # 取出 Python 标量动作
        logprob = logprob.item()  # 取出 Python 标量 logprob
        value = value.item()  # 取出 Python 标量 value

        transition = Transition(  # 构造当前一步的样本
            obs=obs.copy(),  # 保存当前观察
            action=action,  # 保存动作
            logprob=logprob,  # 保存旧策略对数概率
            reward=0.0,  # 先占位，终局后再统一回填奖励
            done=False,  # 先占位，终局后再修正
            value=value,  # 保存价值估计
            legal_mask=legal_mask.copy(),  # 保存合法动作掩码
            player=player,  # 保存玩家身份
        )  # 当前样本构造结束

        trajectories[player].append(transition)  # 将样本加入当前玩家的轨迹

        next_obs, reward, done, info = env.step(action)  # 执行动作，推进环境

        obs = next_obs  # 更新当前观察

        if done:  # 如果对局结束
            winner = env.winner  # 读取胜者

            for p in [1, -1]:  # 分别处理两个玩家的轨迹
                final_reward = 0.0  # 默认奖励为 0

                if winner == p:  # 如果该玩家获胜
                    final_reward = 1.0  # 奖励为 +1
                elif winner == -p:  # 如果该玩家失败
                    final_reward = -1.0  # 奖励为 -1
                else:  # 否则为平局
                    final_reward = 0.0  # 奖励为 0

                for i in range(len(trajectories[p])):  # 遍历该玩家的所有样本
                    trajectories[p][i].reward = final_reward  # 回填整个轨迹的最终奖励
                    trajectories[p][i].done = (i == len(trajectories[p]) - 1)  # 只有最后一步标记为 done=True

    all_transitions = trajectories[1] + trajectories[-1]  # 将双方轨迹合并成训练样本
    return all_transitions, env.winner  # 返回样本和胜者


class PPOTrainer:  # 定义 PPO 训练器
    def __init__(  # 初始化训练器
        self,  # 实例自身
        board_size=9,  # 棋盘大小
        lr=3e-4,  # 学习率
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE 的 lambda
        clip_coef=0.2,  # PPO clip 系数
        ent_coef=0.01,  # 熵奖励系数
        vf_coef=0.5,  # 价值损失系数
        max_grad_norm=0.5,  # 梯度裁剪阈值
        device="cpu",  # 设备
    ):  # 初始化结束
        self.device = device  # 保存设备
        self.gamma = gamma  # 保存 gamma
        self.gae_lambda = gae_lambda  # 保存 gae lambda
        self.clip_coef = clip_coef  # 保存 clip 系数
        self.ent_coef = ent_coef  # 保存熵系数
        self.vf_coef = vf_coef  # 保存价值损失系数
        self.max_grad_norm = max_grad_norm  # 保存最大梯度范数

        self.model = PolicyValueNet(board_size=board_size).to(device)  # 创建网络并放到设备上
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # 创建 Adam 优化器?lr用在哪

    def update(self, transitions, batch_size=128, epochs=4):  # 使用一批样本更新 PPO
        advantages, returns = compute_gae(transitions, self.gamma, self.gae_lambda)  # 计算优势和回报

        obs = torch.tensor(np.array([t.obs for t in transitions]), dtype=torch.float32, device=self.device)  # 收集状态张量
        actions = torch.tensor(np.array([t.action for t in transitions]), dtype=torch.long, device=self.device)  # 收集动作张量
        old_logprobs = torch.tensor(np.array([t.logprob for t in transitions]), dtype=torch.float32, device=self.device)  # 收集旧策略 logprob
        legal_masks = torch.tensor(np.array([t.legal_mask for t in transitions]), dtype=torch.float32, device=self.device)  # 收集合合法动作掩码
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)  # returns 转 tensor
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)  # advantages 转 tensor

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 对优势做标准化，稳定训练

        n = len(transitions)  # 样本总数
        idxs = np.arange(n)  # 生成样本索引

        for _ in range(epochs):  # 多轮 PPO 更新
            np.random.shuffle(idxs)  # 每一轮打乱样本顺序

            for start in range(0, n, batch_size):  # 按 batch 划分小批次
                end = start + batch_size  # 计算当前 batch 结束位置
                mb_idx = idxs[start:end]  # 取出当前 batch 的索引

                mb_obs = obs[mb_idx]  # 当前 batch 的状态
                mb_actions = actions[mb_idx]  # 当前 batch 的动作
                mb_old_logprobs = old_logprobs[mb_idx]  # 当前 batch 的旧 logprob
                mb_legal_masks = legal_masks[mb_idx]  # 当前 batch 的合法动作掩码
                mb_returns = returns[mb_idx]  # 当前 batch 的 returns
                mb_advantages = advantages[mb_idx]  # 当前 batch 的优势

                _, new_logprob, entropy, value = self.model.get_action_and_value(  # 用当前网络重新计算 logprob、熵、价值
                    mb_obs,  # 输入状态
                    mb_legal_masks,  # 输入 mask
                    mb_actions,  # 指定动作，得到该动作在当前策略下的 logprob
                )  # 前向结束

                logratio = new_logprob - mb_old_logprobs  # 计算 log(pi_new/pi_old)
                ratio = torch.exp(logratio)  # 得到概率比值 pi_new/pi_old

                pg_loss1 = -mb_advantages * ratio  # PPO 原始策略损失
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)  # PPO clip 后策略损失
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 取两者最大值后求均值

                v_loss = ((value - mb_returns) ** 2).mean()  # 价值函数均方误差损失
                entropy_loss = entropy.mean()  # 熵均值，用于鼓励探索

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss  # 总损失

                self.optimizer.zero_grad()  # 清空旧梯度
                loss.backward()  # 反向传播计算梯度
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # 梯度裁剪，防止梯度爆炸
                self.optimizer.step()  # 优化器更新参数

    def save(self, path):  # 保存模型参数
        torch.save(self.model.state_dict(), path)  # 将 state_dict 保存到文件

    def load(self, path):  # 加载模型参数
        self.model.load_state_dict(torch.load(path, map_location=self.device))  # 从文件加载参数到当前设备


def train():  # 训练主函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择 GPU 或 CPU
    print("Use device:", device)  # 打印当前设备

    board_size = 9  # 设置棋盘大小
    env = GomokuEnv(board_size=board_size, win_length=5)  # 创建五子棋环境
    trainer = PPOTrainer(board_size=board_size, device=device)  # 创建 PPO 训练器

    total_iterations = 500  # 总训练轮数
    episodes_per_iter = 32  # 每轮采样多少局自对弈

    for iteration in range(1, total_iterations + 1):  # 开始训练循环
        all_transitions = []  # 用于存储当前轮所有样本
        win_stats = {1: 0, -1: 0, 0: 0}  # 统计先手胜、后手胜、平局数量

        for _ in range(episodes_per_iter):  # 每轮采样多局
            transitions, winner = self_play_episode(env, trainer.model, device)  # 自对弈采样一局
            all_transitions.extend(transitions)  # 将该局样本加入总样本池
            win_stats[winner] += 1  # 统计胜负情况

        trainer.update(all_transitions, batch_size=128, epochs=4)  # 用当前轮收集的样本更新网络

        if iteration % 10 == 0:  # 每 10 轮打印一次日志
            print(  # 打印训练信息
                f"Iter {iteration:4d} | "  # 当前轮数
                f"Samples {len(all_transitions):5d} | "  # 当前轮样本数
                f"P1 win {win_stats[1]:2d} | "  # 先手获胜局数
                f"P2 win {win_stats[-1]:2d} | "  # 后手获胜局数
                f"Draw {win_stats[0]:2d}"  # 平局局数
            )  # 打印结束

        if iteration % 50 == 0:  # 每 50 轮保存一次模型
            trainer.save(f"gomoku_ppo_iter_{iteration}.pth")  # 保存模型文件
            print(f"Model saved: gomoku_ppo_iter_{iteration}.pth")  # 打印保存提示


if __name__ == "__main__":  # 如果当前文件被直接执行
    train()  # 启动训练