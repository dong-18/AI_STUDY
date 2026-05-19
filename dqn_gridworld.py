"""
用 DQN (Deep Q-Network) 实现 Grid-World（网格世界）：
- 环境：一个 N×N 的网格，起点 S，终点 G（到达即结束）
- 动作：上/下/左/右
- 奖励：每走一步 -1；到达终点 +10
- 目标：使用神经网络近似 Q 函数，学习最优策略

DQN 改进：
1. 使用神经网络代替 Q 表
2. 经验回放（Experience Replay）
3. 目标网络（Target Network）
"""


import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# -----------------------------
# 1) 定义 GridWorld 环境
# -----------------------------
class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.reset()

        # 动作编号到动作向量的映射：0上 1下 2左 3右
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

    def reset(self):
        """重置环境到起点，返回初始状态"""
        self.agent_pos = self.start
        return self._state()

    def _state(self):
        """把 (row, col) 转换为特征向量：[row/size, col/size]"""
        r, c = self.agent_pos
        return [float(r / self.size), float(c / self.size)]

    def step(self, action):
        """
        执行动作 action，返回:
        next_state, reward, done
        """
        dr, dc = self.actions[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # 边界处理：撞墙则停在原地（也算走了一步）
        nr = max(0, min(self.size - 1, nr))
        nc = max(0, min(self.size - 1, nc))

        self.agent_pos = (nr, nc)

        # 奖励设计：到达终点 +10，否则每步 -1
        if self.agent_pos == self.goal:
            return self._state(), 10.0, True
        else:
            return self._state(), -1.0, False


# -----------------------------
# 2) 定义 Q 神经网络
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------
# 3) 经验回放缓冲区
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return list(state), list(action), list(reward), \
               list(next_state), list(done)

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# 4) DQN 智能体
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 主网络和目标网络
        self.policy_net = DQN(state_dim, hidden_dim, action_dim)
        self.target_net = DQN(state_dim, hidden_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 经验回放
        self.memory = ReplayBuffer()

        # 超参数
        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10  # 目标网络更新频率

    def select_action(self, state):
        """使用 epsilon-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            #这部分在干嘛 将state转为二维，选择当前网络最优的选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 计算当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            #这里为什么要1-dones使得最大值为10，因为到了终点，不需要再往下走了，gamma是对于当前而言是否考虑未来的影响，以便做出选择
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------
# 5) DQN 训练函数
# -----------------------------
def dqn_train(
    agent,
    env,
    episodes=1000,
    max_steps=200,
    update_target_every=10,
    save_rewards=True
):
    """DQN 训练过程"""
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
#这里为什么总是要取最大值 为了强化最优结果
            # 执行动作
            next_state, reward, done = env.step(action)

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 训练
            agent.replay()

            # 更新状态和奖励
            state = next_state
            episode_reward += reward

            # 如果结束，跳出循环
            if done:
                break

        # 衰减探索率
        agent.decay_epsilon()

        # 定期更新目标网络
        if episode % update_target_every == 0:
            agent.update_target()

        # 记录奖励
        rewards_history.append(episode_reward)

        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / len(rewards_history[-100:]) if rewards_history else 0
            print(f"Episode {episode + 1}/{episodes}, Avg Reward (last 100): {avg_reward:.2f}")

    return rewards_history


# -----------------------------
# 6) 可视化和评估函数
# -----------------------------
def get_greedy_policy(agent, env):
    """获取贪心策略"""
    policy = [0] * (env.size * env.size)

    for i in range(env.size):
        for j in range(env.size):
            state = [i / env.size, j / env.size]
            action = agent.select_action(state)
            policy[i * env.size + j] = action

    return [int(x) for x in policy]

def render_path(env, policy, max_steps=50):
    """渲染路径"""
    env.reset()
    path = [env.agent_pos]

    for _ in range(max_steps):
        r, c = env.agent_pos
        state = env._state()
        action = policy[r * env.size + c]
        _, _, done = env.step(action)
        path.append(env.agent_pos)
        if done:
            break

    return path

def print_grid_with_path(size, start, goal, path):
    """打印网格路径"""
    grid = [["." for _ in range(size)] for _ in range(size)]
    sr, sc = start
    gr, gc = goal
    grid[sr][sc] = "S"
    grid[gr][gc] = "G"

    for (r, c) in path:
        if (r, c) != start and (r, c) != goal:
            grid[r][c] = "*"

    for r in range(size):
        print(" ".join(grid[r]))

def plot_training_rewards(rewards, window=100):
    """绘制训练奖励曲线"""
    plt.figure(figsize=(10, 6))

    # 计算移动平均
    moving_avg = [sum(rewards[i:i+window])/window for i in range(len(rewards)-window+1)]

    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(range(window-1, len(rewards)), moving_avg,
             label=f'Moving Average (window={window})', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# 7) 主程序
# -----------------------------
if __name__ == "__main__":
    # 创建环境
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4))

    # 创建 DQN 智能体
    state_dim = 2  # [row/size, col/size]
    action_dim = 4  # 4个动作
    agent = DQNAgent(state_dim, action_dim, hidden_dim=64)

    # 训练 DQN
    print("开始 DQN 训练...")
    rewards_history = dqn_train(
        agent,
        env,
        episodes=1000,
        max_steps=200,
        update_target_every=10
    )

    # 绘制训练曲线
    plot_training_rewards(rewards_history)

    # 获取贪心策略并展示路径
    policy = get_greedy_policy(agent, env)
    path = render_path(env, policy, max_steps=50)

    print("\nLearned path (row, col):")
    print(path)
    print("\nGrid with path:")
    print_grid_with_path(env.size, env.start, env.goal, path)