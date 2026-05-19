"""
用 Grid-World（网格世界）说明 Q-learning：
- 环境：一个 N×N 的网格，起点 S，终点 G（到达即结束）
- 动作：上/下/左/右
- 奖励：每走一步 -1；到达终点 +10
- 目标：学习动作价值函数 Q(s,a)，最终得到一条到终点的最短/近似最短路径策略

Q-learning 更新公式（离策略）：
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
"""

import numpy as np
import random

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
        """把 (row, col) 映射为单个整数状态 id，便于建 Q 表"""
        r, c = self.agent_pos
        return r * self.size + c

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
# 2) Q-learning 训练
# -----------------------------
def q_learning_train(
    env,
    episodes=2000,
    alpha=0.1,      # 学习率
    gamma=0.99,     # 折扣因子
    epsilon=1.0,    # 初始探索率
    epsilon_min=0.05,
    epsilon_decay=0.995,
    max_steps=200
):
    """
    用 epsilon-greedy 进行探索：
    - 以概率 epsilon 随机选动作（探索）
    - 否则选 argmax_a Q(s,a)（利用）
    """
    n_states = env.size * env.size
    n_actions = 4

    # Q 表：每个状态、每个动作一个价值
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    for ep in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            # 选动作：epsilon-greedy
            if random.random() < epsilon:
                a = random.randint(0, n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))

            s_next, r, done = env.step(a)

            # Q-learning 更新（核心）
            # 目标 = r + gamma * max_a' Q(s_next, a')
            td_target = r + (0.0 if done else gamma * np.max(Q[s_next]))
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error

            s = s_next
            if done:
                break

        # 逐渐降低探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q


# -----------------------------
# 3) 从 Q 表导出策略并演示一条路径
# -----------------------------
def greedy_policy_from_Q(Q):
    """给定 Q 表，返回每个状态下的贪心动作"""
    return np.argmax(Q, axis=1)

def render_path(env, policy, max_steps=50):
    """
    从起点按贪心策略走，打印路径。
    用坐标显示更直观： (row, col)
    """
    env.reset()
    path = [env.agent_pos]

    for _ in range(max_steps):
        s = env._state()
        a = int(policy[s])
        _, _, done = env.step(a)
        path.append(env.agent_pos)
        if done:
            break
    return path

def print_grid_with_path(size, start, goal, path):
    """
    把路径画在网格上：
    S = 起点, G = 终点, * = 路径, . = 空
    """
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


if __name__ == "__main__":
    # 创建 5x5 环境：左上到右下
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4))

    # 训练 Q 表
    Q = q_learning_train(
        env,
        episodes=3000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        max_steps=200
    )

    # 导出贪心策略并展示一条路径
    policy = greedy_policy_from_Q(Q)
    path = render_path(env, policy, max_steps=50)

    print("Learned path (row, col):")
    print(path)
    print("\nGrid with path:")
    print_grid_with_path(env.size, env.start, env.goal, path)