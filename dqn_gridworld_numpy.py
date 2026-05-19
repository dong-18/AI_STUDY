"""
用 DQN (Deep Q-Network) 实现 Grid-World（网格世界）- NumPy版本：
- 环境：一个 N×N 的网格，起点 S，终点 G（到达即结束）
- 动作：上/下/左/右
- 奖励：每走一步 -1；到达终点 +10
- 目标：使用神经网络近似 Q 函数，学习最优策略

DQN 改进：
1. 使用神经网络代替 Q 表
2. 经验回放（Experience Replay）
3. 目标网络（Target Network）
"""

import numpy as np
import random
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
        return np.array([r / self.size, c / self.size], dtype=np.float32)

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
# 2) 定义 Q 神经网络（NumPy版本）
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))

        # 存储梯度用于反向传播
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        self.dW3 = np.zeros_like(self.W3)
        self.db3 = np.zeros_like(self.b3)

        # 优化器参数
        self.momentum = 0.9
        self.learning_rate = 0.001

        # 动量
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_b3 = np.zeros_like(self.b3)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """前向传播"""
        # 第1层
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # 第2层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        # 第3层（输出层）
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def backward(self, x, y, output):
        """反向传播"""
        # 计算输出层的梯度
        delta3 = output - y
        # 第3层的梯度
        self.dW3 = np.dot(self.a2.T, delta3)
        self.db3 = np.sum(delta3, axis=0, keepdims=True)

        # 第2层的梯度
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(self.z2)
        self.dW2 = np.dot(self.a1.T, delta2)
        self.db2 = np.sum(delta2, axis=0, keepdims=True)

        # 第1层的梯度
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        self.dW1 = np.dot(x.T, delta1)
        self.db1 = np.sum(delta1, axis=0, keepdims=True)

    def update_weights(self):
        """更新权重（使用动量）"""
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * self.dW1
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * self.db1
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * self.dW2
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * self.db2
        self.v_W3 = self.momentum * self.v_W3 - self.learning_rate * self.dW3
        self.v_b3 = self.momentum * self.v_b3 - self.learning_rate * self.db3

        self.W1 += self.v_W1
        self.b1 += self.v_b1
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W3 += self.v_W3
        self.b3 += self.v_b3

    def mse_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)


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
        return np.array(state), np.array(action), np.array(reward), \
               np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# 4) DQN 智能体（NumPy版本）
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 主网络和目标网络
        self.policy_net = NeuralNetwork(state_dim, hidden_dim, action_dim)
        self.target_net = NeuralNetwork(state_dim, hidden_dim, action_dim)

        # 复制权重
        self.target_net.W1 = self.policy_net.W1.copy()
        self.target_net.b1 = self.policy_net.b1.copy()
        self.target_net.W2 = self.policy_net.W2.copy()
        self.target_net.b2 = self.policy_net.b2.copy()
        self.target_net.W3 = self.policy_net.W3.copy()
        self.target_net.b3 = self.policy_net.b3.copy()

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
            q_values = self.policy_net.forward(np.array([state]))
            return np.argmax(q_values[0])
    def select_action_1(self, state):
        """使用 epsilon-greedy 策略选择动作"""
        return random.randint(0, self.action_dim - 1)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 批量前向传播
        current_q = self.policy_net.forward(states)

        # 计算目标 Q 值
        with np.errstate(divide='ignore', invalid='ignore'):
            next_q = self.target_net.forward(next_states)
            max_next_q = np.max(next_q, axis=1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 创建目标输出
        target_output = current_q.copy()
        for i in range(self.batch_size):
            target_output[i, actions[i]] = target_q[i]

        # 计算损失
        loss = self.policy_net.mse_loss(current_q, target_output)

        # 反向传播
        self.policy_net.backward(states, target_output, current_q)

        # 更新权重
        self.policy_net.update_weights()

        return loss

    def update_target(self):
        """更新目标网络"""
        self.target_net.W1 = self.policy_net.W1.copy()
        self.target_net.b1 = self.policy_net.b1.copy()
        self.target_net.W2 = self.policy_net.W2.copy()
        self.target_net.b2 = self.policy_net.b2.copy()
        self.target_net.W3 = self.policy_net.W3.copy()
        self.target_net.b3 = self.policy_net.b3.copy()

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
    losses_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 训练
            loss = agent.replay()

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

        # 记录奖励和损失
        rewards_history.append(episode_reward)
        if loss is not None:
            losses_history.append(loss)

        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_loss = np.mean(losses_history[-100:]) if losses_history else 0
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")

    return rewards_history


# -----------------------------
# 6) 可视化和评估函数
# -----------------------------
def get_greedy_policy(agent, env):
    """获取贪心策略"""
    policy = np.zeros(env.size * env.size)

    for i in range(env.size):
        for j in range(env.size):
            state = np.array([i / env.size, j / env.size])
            action = agent.select_action(state)
            policy[i * env.size + j] = action

    return policy.astype(int)

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
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

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
    print("开始 DQN 训练（NumPy版本）...")
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