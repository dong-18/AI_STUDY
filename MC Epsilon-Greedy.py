# 用 GridWorld 演示 Monte Carlo Control + Epsilon-Greedy（on-policy）
# 目标：通过回合采样学习动作价值 Q(s,a)，并用 ε-greedy 改进策略
#
# 环境：
# - 网格 5x5，起点 S=(0,0)，终点 G=(4,4)
# - 每步奖励 -1，到达终点奖励 0（并终止）
# - 撞墙原地不动
#
# 算法（每回合）：
# 1) 用当前 ε-greedy 策略生成一条 episode: (S0,A0,R1),...,(ST)
# 2) 反向计算回报 G_t = R_{t+1} + γ G_{t+1}
# 3) First-Visit MC：每个 (s,a) 在该回合第一次出现时，用样本均值更新 Q(s,a)
# 4) 策略改进：行为策略始终是基于 Q 的 ε-greedy（on-policy MC control）

import random
from collections import defaultdict

ACTIONS = ["U", "D", "L", "R"]
A2D = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

class GridWorld:
    def __init__(self, n=5, start=(0, 0), goal=(4, 4)):
        self.n = n
        self.start = start
        self.goal = goal
        self.s = start

    def reset(self):
        self.s = self.start
        return self.s

    def step(self, a):
        if self.s == self.goal:
            return self.s, 0.0, True

        dr, dc = A2D[a]
        r, c = self.s
        nr, nc = r + dr, c + dc

        # 撞墙：留在原地
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            nr, nc = r, c

        ns = (nr, nc)
        self.s = ns

        done = (ns == self.goal)
        reward = 0.0 if done else -1.0
        return ns, reward, done

def epsilon_greedy_action(Q, s, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    # 选 Q 最大的动作（并列随机）
    qs = [Q[(s, a)] for a in ACTIONS]
    m = max(qs)
    best = [a for a, q in zip(ACTIONS, qs) if q == m]
    return random.choice(best)

def mc_control_epsilon_greedy(
    episodes=50_000,
    gamma=0.99,
    epsilon=0.1,
    max_steps=500,
    seed=0,
):
    random.seed(seed)
    env = GridWorld()

    Q = defaultdict(float)    # Q[(s,a)]
    N = defaultdict(int)      # 访问计数，用于样本均值 MC 更新

    for _ in range(episodes):
        # 1) 采样生成一条 episode
        episode = []  # list of (s,a,r)
        s = env.reset()
        for _t in range(max_steps):
            a = epsilon_greedy_action(Q, s, epsilon)
            ns, r, done = env.step(a)
            episode.append((s, a, r))
            s = ns
            if done:
                break

        # 2) 反向算回报并做 First-Visit MC 更新
        G = 0.0
        seen = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if (s, a) in seen:
                continue
            seen.add((s, a))
            N[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]  # 样本均值

    return Q

def greedy_policy_from_Q(Q):
    pi = {}
    for r in range(5):
        for c in range(5):
            s = (r, c)
            if s == (4, 4):
                pi[s] = "G"
                continue
            qs = {a: Q[(s, a)] for a in ACTIONS}
            m = max(qs.values())
            best = [a for a, q in qs.items() if q == m]
            pi[s] = random.choice(best)
    return pi

def print_policy(pi, n=5):
    # 用箭头打印策略
    arrow = {"U":"^","D":"v","L":"<","R":">","G":"G"}
    for r in range(n):
        row = []
        for c in range(n):
            row.append(arrow[pi[(r,c)]])
        print(" ".join(row))

if __name__ == "__main__":
    Q = mc_control_epsilon_greedy(episodes=80_0000, gamma=0.99, epsilon=0.1, seed=42)
    pi = greedy_policy_from_Q(Q)
    print_policy(pi)