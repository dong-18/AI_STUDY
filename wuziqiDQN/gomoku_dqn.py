# 导入随机数模块，用于 epsilon-greedy 选择随机动作、经验回放采样等
import random

# 导入 numpy，用于棋盘状态存储与数值计算
import numpy as np

# 从 collections 导入 deque，作为固定长度的经验回放队列
from collections import deque

# 导入 dataclass，方便定义 Transition 数据结构
from dataclasses import dataclass

# 导入 PyTorch 主库
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入优化器模块
import torch.optim as optim


# =========================
# 1. 五子棋环境
# =========================
class GomokuEnv:
    # 初始化环境
    def __init__(self, board_size=15, win_len=5):
        # 棋盘大小，默认 15x15
        self.board_size = board_size
        # 连成多少子算赢，五子棋就是 5
        self.win_len = win_len
        # 动作总数 = 棋盘总格子数
        self.action_size = board_size * board_size
        # 创建并重置环境
        self.reset()

    # 重置环境，开始新一局
    def reset(self):
        # 棋盘用二维数组表示，0 表示空位，1 表示黑子，-1 表示白子
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # 当前玩家设为 1
        self.current_player = 1
        # 是否结束
        self.done = False
        # 胜者，1 / -1 / 0
        self.winner = 0
        # 返回当前状态
        return self.get_state()

    # 获取当前状态
    def get_state(self):
        """
        返回“从当前玩家视角”编码的状态：
        channel 0: 当前玩家的棋子
        channel 1: 对手的棋子
        状态形状为 [2, H, W]
        """
        # 找出当前玩家棋子的位置，转换为 float32
        cur = (self.board == self.current_player).astype(np.float32)
        # 找出对手棋子的位置，转换为 float32
        opp = (self.board == -self.current_player).astype(np.float32)
        # 将两个通道堆叠起来
        state = np.stack([cur, opp], axis=0)
        # 返回状态
        return state

    # 返回当前合法动作掩码
    def valid_actions(self):
        # 将棋盘展平成一维，空位为 1，非空位为 0
        return (self.board.reshape(-1) == 0).astype(np.float32)

    # 执行动作
    def step(self, action):
        # 如果游戏已经结束，则报错
        if self.done:
            raise ValueError("Game is already over.")

        # 将一维动作编号映射回二维坐标
        x = action // self.board_size
        y = action % self.board_size

        # 如果该位置已经有棋子，则视为非法动作
        if self.board[x, y] != 0:
            # 非法动作直接结束游戏
            self.done = True
            # 另一方获胜
            self.winner = -self.current_player
            # 当前玩家给负奖励
            reward = -1.0
            # 返回当前状态、奖励、结束标记、额外信息
            return self.get_state(), reward, self.done, {"illegal_move": True}

        # 在棋盘对应位置落子
        self.board[x, y] = self.current_player

        # 检查当前这一步是否形成胜利
        if self.check_win(x, y):
            # 游戏结束
            self.done = True
            # 当前玩家获胜
            self.winner = self.current_player
            # 奖励为 +1
            reward = 1.0
            # 取得下一状态
            next_state = self.get_state()
            # 返回结果
            return next_state, reward, self.done, {}

        # 如果棋盘已满，则平局
        if np.all(self.board != 0):
            # 游戏结束
            self.done = True
            # 无胜者
            self.winner = 0
            # 平局奖励 0
            reward = 0.0
            # 获取状态
            next_state = self.get_state()
            # 返回结果
            return next_state, reward, self.done, {}

        # 切换到另一位玩家
        self.current_player *= -1
        # 非终局奖励为 0
        reward = 0.0
        # 获取切换视角后的下一状态
        next_state = self.get_state()
        # 返回结果
        return next_state, reward, self.done, {}

    # 判断最近落下的一子是否导致胜利
    def check_win(self, x, y):
        # 当前落子玩家
        player = self.board[x, y]
        # 四个方向：横向、纵向、主对角线、副对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # 枚举每个方向
        for dx, dy in directions:
            # 至少当前这个点本身算一个
            count = 1

            # 向正方向统计连续子数
            i, j = x + dx, y + dy
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i += dx
                j += dy

            # 向反方向统计连续子数
            i, j = x - dx, y - dy
            while 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == player:
                count += 1
                i -= dx
                j -= dy

            # 如果连续数量达到胜利条件，则返回 True
            if count >= self.win_len:
                return True

        # 所有方向都不满足，则未获胜
        return False

    # 文本方式打印棋盘
    def render(self):
        # 定义棋盘字符映射
        mapping = {0: ".", 1: "X", -1: "O"}
        # 逐行输出
        for i in range(self.board_size):
            print(" ".join(mapping[v] for v in self.board[i]))
        # 输出空行分隔
        print()


# =========================
# 2. 经验回放数据结构
# =========================
@dataclass
class Transition:
    # 当前状态
    state: np.ndarray
    # 动作
    action: int
    # 奖励
    reward: float
    # 下一状态
    next_state: np.ndarray
    # 是否终局
    done: float
    # 当前状态的合法动作掩码
    valid_mask: np.ndarray
    # 下一状态的合法动作掩码
    next_valid_mask: np.ndarray


# 定义经验回放池
class ReplayBuffer:
    # 初始化经验池容量
    def __init__(self, capacity=50000):
        # 使用 deque 实现固定长度队列
        self.buffer = deque(maxlen=capacity)

    # 往经验池添加一条经验
    def push(self, *args):
        # 使用 Transition 封装后加入队列
        self.buffer.append(Transition(*args))

    # 从经验池随机采样一个 batch
    def sample(self, batch_size):
        # 随机取 batch_size 个样本
        batch = random.sample(self.buffer, batch_size)
        # 返回采样结果
        return batch

    # 返回经验池当前长度
    def __len__(self):
        # 返回队列长度
        return len(self.buffer)


# =========================
# 3. DQN 网络
# =========================
class GomokuDQN(nn.Module):
    # 初始化网络
    def __init__(self, board_size=15, action_size=225):
        # 调用父类初始化
        super().__init__()
        # 保存棋盘大小
        self.board_size = board_size
        # 保存动作空间大小
        self.action_size = action_size

        # 卷积特征提取部分
        self.net = nn.Sequential(
            # 第 1 层卷积，输入 2 个通道，输出 64 个通道
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第 2 层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第 3 层卷积
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # 激活函数
            nn.ReLU(),
        )

        # 输出头，将卷积特征映射到 225 个动作的 Q 值
        self.head = nn.Sequential(
            # 拉平成一维向量
            nn.Flatten(),
            # 全连接层
            nn.Linear(128 * board_size * board_size, 512),
            # 激活函数
            nn.ReLU(),
            # 输出每个动作的 Q 值
            nn.Linear(512, action_size)
        )

    # 前向传播
    def forward(self, x):
        # 先经过卷积层
        x = self.net(x)
        # 再经过全连接输出
        x = self.head(x)
        # 返回 Q 值
        return x


# =========================
# 4. DQN Agent
# =========================
class DQNAgent:
    # 初始化智能体
    def __init__(
        self,
        board_size=15,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=50000,
        target_update_freq=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        # 棋盘大小
        self.board_size = board_size
        # 动作数量
        self.action_size = board_size * board_size
        # 折扣因子
        self.gamma = gamma
        # 当前 epsilon
        self.epsilon = epsilon_start
        # 初始 epsilon
        self.epsilon_start = epsilon_start
        # 最终 epsilon
        self.epsilon_end = epsilon_end
        # epsilon 衰减速度
        self.epsilon_decay = epsilon_decay
        # 目标网络更新频率
        self.target_update_freq = target_update_freq
        # 设备
        self.device = device
        # 学习步数
        self.learn_step = 0

        # 策略网络
        self.policy_net = GomokuDQN(board_size, self.action_size).to(device)
        # 目标网络
        self.target_net = GomokuDQN(board_size, self.action_size).to(device)
        # 初始时将目标网络参数设置成和策略网络一致
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 目标网络不参与训练
        self.target_net.eval()

        # Adam 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # MSE 损失函数
        self.criterion = nn.MSELoss()

    # 根据当前状态选择动作
    def select_action(self, state, valid_mask, train=True):
        # 如果在训练模式并且触发探索，则随机选一个合法动作
        if train and random.random() < self.epsilon:
            # 找出所有合法动作编号
            valid_actions = np.where(valid_mask > 0)[0]
            # 随机选取其中一个
            return int(np.random.choice(valid_actions))

        # 将状态转成张量并扩展 batch 维度
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 不需要梯度
        with torch.no_grad():
            # 计算当前状态下每个动作的 Q 值
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()

        # 把非法动作的 Q 值设为极小
        q_values[valid_mask == 0] = -1e9
        # 返回 Q 值最大的动作
        return int(np.argmax(q_values))

    # 更新 epsilon
    def update_epsilon(self, step):
        # 指数衰减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * step / self.epsilon_decay)

    # 执行一次训练
    def train_step(self, replay_buffer, batch_size=64):
        # 如果经验不够一个 batch，则不训练
        if len(replay_buffer) < batch_size:
            return None

        # 从经验池采样
        batch = replay_buffer.sample(batch_size)

        # 拼接状态 batch
        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        # 拼接动作 batch
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        # 拼接奖励 batch
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        # 拼接下一状态 batch
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        # 拼接 done batch
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        # 拼接下一状态合法动作掩码
        next_valid_masks = torch.tensor(
            np.array([b.next_valid_mask for b in batch]),
            dtype=torch.float32,
            device=self.device
        )

        # 计算当前 Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # 计算目标 Q 值，不需要梯度
        with torch.no_grad():
            # 目标网络输出下一状态所有动作的 Q 值
            next_q = self.target_net(next_states)

            # 非法动作位置设为极小值
            invalid = (next_valid_masks == 0)
            next_q[invalid] = -1e9

            # 取下一状态最大 Q 值
            max_next_q = next_q.max(dim=1, keepdim=True)[0]

            # DQN 目标值
            target_q = rewards + self.gamma * (1 - dones) * max_next_q

        # 计算损失
        loss = self.criterion(q_values, target_q)

        # 梯度清零
        self.optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        # 更新参数
        self.optimizer.step()

        # 学习步数加一
        self.learn_step += 1

        # 每隔固定步数同步一次目标网络
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 返回损失值
        return loss.item()


# =========================
# 5. 自我对弈训练
# =========================
def train_self_play(
    episodes=5000,
    board_size=15,
    batch_size=64,
    buffer_capacity=50000,
    save_path="gomoku_dqn.pth"
):
    # 创建五子棋环境
    env = GomokuEnv(board_size=board_size)
    # 创建智能体
    agent = DQNAgent(board_size=board_size)
    # 创建经验回放池
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # 全局步数
    global_step = 0

    # 遍历每一局
    for episode in range(episodes):
        # 重置环境并获得初始状态
        state = env.reset()
        # 当前局是否结束
        done = False

        # 用于记录这一局所有步的信息
        player_history = []

        # 一直下到终局
        while not done:
            # 获取当前合法动作掩码
            valid_mask = env.valid_actions().copy()
            # 保存当前玩家
            current_player = env.current_player

            # 智能体根据当前状态选择动作
            action = agent.select_action(state, valid_mask, train=True)
            # 环境执行动作
            next_state, reward, done, info = env.step(action)
            # 获取下一状态下合法动作掩码
            next_valid_mask = env.valid_actions().copy()

            # 记录这一手的信息
            player_history.append({
                "player": current_player,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "valid_mask": valid_mask,
                "next_valid_mask": next_valid_mask
            })

            # 更新当前状态
            state = next_state
            # 全局步数加一
            global_step += 1
            # 更新 epsilon 为什么
            agent.update_epsilon(global_step)

            # 可选：每一步都尝试训练一次
            #loss = agent.train_step(replay_buffer, batch_size=batch_size)

        # 对局结束后获取胜者
        winner = env.winner

        # 将本局的历史记录写入经验池
        for item in player_history:
            # 默认最终奖励为 0
            final_reward = 0.0

            # 如果不是平局，则胜者为 +1，败者为 -1
            if winner != 0:
                final_reward = 1.0 if item["player"] == winner else -1.0

            # 写入经验池
            replay_buffer.push(
                item["state"],
                item["action"],
                final_reward,
                item["next_state"],
                float(item["done"]),
                item["valid_mask"],
                item["next_valid_mask"]
            )

            # 每加入一条经验，就训练一步
            agent.train_step(replay_buffer, batch_size=batch_size)
        print(episode)
        # 每 50 局打印一次训练信息
        if (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1}/{episodes}, "
                f"Epsilon: {agent.epsilon:.4f}, "
                f"Buffer: {len(replay_buffer)}"
            )越远的收益和近处的-1，两个五子棋

        # 每 500 局保存一次模型
        if (episode + 1) % 500 == 0:
            torch.save(agent.policy_net.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # 训练结束后保存最终模型
    torch.save(agent.policy_net.state_dict(), save_path)
    # 打印保存信息
    print(f"Training finished. Final model saved to {save_path}")


# =========================
# 6. 人机对战
# =========================
def play_human_vs_ai(model_path="gomoku_dqn.pth", board_size=15):
    # 创建环境
    env = GomokuEnv(board_size=board_size)
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = GomokuDQN(board_size, board_size * board_size).to(device)
    # 加载训练好的参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 切换到评估模式
    model.eval()

    # 重置环境
    state = env.reset()

    # 设定人类和 AI 的执子
    human_player = -1
    ai_player = 1

    # 对局循环
    while not env.done:
        # 打印棋盘
        env.render()

        # 如果轮到人类
        if env.current_player == human_player:
            try:
                # 读取用户输入坐标
                x, y = map(int, input("请输入落子坐标 x y: ").split())
                # 转成动作编号
                action = x * board_size + y
            except Exception:
                # 输入异常则提示重新输入
                print("输入错误，请重新输入")
                continue
        else:
            # 获取合法动作掩码
            valid_mask = env.valid_actions()
            # 将状态转成张量
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # 不计算梯度
            with torch.no_grad():
                # 计算 Q 值
                q_values = model(state_tensor).squeeze(0).cpu().numpy()

            # 屏蔽非法动作
            q_values[valid_mask == 0] = -1e9
            # 选取 Q 值最大的动作
            action = int(np.argmax(q_values))
            # 还原成二维坐标
            x, y = action // board_size, action % board_size
            # 打印 AI 落子位置
            print(f"AI 落子: {x} {y}")

        # 执行动作并更新状态
        state, reward, done, info = env.step(action)

    # 最后一轮渲染棋盘
    env.render()

    # 输出胜负结果
    if env.winner == ai_player:
        print("AI 获胜")
    elif env.winner == human_player:
        print("你赢了")
    else:
        print("平局")


# =========================
# 7. 主函数入口
# =========================
if __name__ == "__main__":
    # 开始训练模型
    train_self_play(
        episodes=3000,
        board_size=15,
        batch_size=64,
        buffer_capacity=50000,
        save_path="gomoku_dqn.pth"
    )

    # 如果你已经训练好模型，可以注释掉上面的训练代码，改为运行下面这一行进行人机对战
    # play_human_vs_ai("gomoku_dqn.pth", board_size=15)