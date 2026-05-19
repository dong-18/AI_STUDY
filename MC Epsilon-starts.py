# MC Control with Exploring Starts (ES) 用 GridWorld 说明
# 要点：每个 episode 都从“随机起点状态 + 随机第一步动作”开始（Exploring Starts）
#      然后用当前策略继续走，收集整条轨迹，用 MC 回报更新 Q(s,a)，并对每个状态做 greedy 改进。
#
# 环境：4x4 GridWorld
# - 动作：0:U 1:D 2:L 3:R
# - 终止：到达 goal
# - 奖励：每步 -1，到 goal 终止（因此等价于学最短路）
#
# 这是“每次访问 (s,a) 的回报样本均值”版本（First-Visit 也可，这里用 Every-Visit 简化）。

import random
from collections import defaultdict

ACTIONS = [0, 1, 2, 3]
DIRS = {
    0: (-1, 0),  # U
    1: (1, 0),   # D
    2: (0, -1),  # L
    3: (0, 1),   # R
}

class GridWorld:
    def __init__(self, n=4, goal=(3, 3)):
        self.n = n
        self.goal = goal

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0, True

        r, c = s
        dr, dc = DIRS[a]
        nr, nc = r + dr, c + dc

        # 撞墙则原地不动
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            ns = (r, c)
        else:
            ns = (nr, nc)

        done = self.is_terminal(ns)
        reward = 0 if done else -1
        return ns, reward, done

    def is_terminal(self, s):
        return s == self.goal

    def random_nonterminal_state(self):
        while True:
            s = (random.randrange(self.n), random.randrange(self.n))
            if not self.is_terminal(s):
                return s

def greedy_action(Q, s):
    qs = [Q[(s, a)] for a in ACTIONS]
    m = max(qs)
    best = [a for a, v in zip(ACTIONS, qs) if v == m]
    return random.choice(best)

def mc_control_exploring_starts(env, episodes=50_000, gamma=1.0, max_steps=200, seed=0):
    random.seed(seed)

    # Q(s,a) 与计数 N(s,a)
    Q = defaultdict(float)
    N = defaultdict(int)

    # 确定性策略 π(s)（terminal 不需要）
    pi = {}

    for _ in range(episodes):
        # --- Exploring Starts：随机起点 + 随机第一动作 ---
        s0 = env.random_nonterminal_state()
        a0 = random.choice(ACTIONS)

        episode = []  # [(s,a,r), ...]
        s = s0
        a = a0

        # 走完整条 episode
        for _t in range(max_steps):
            ns, r, done = env.step(s, a)
            episode.append((s, a, r))
            if done:
                break
            s = ns
            # 后续动作按当前策略（若未定义则先 greedy）
            a = pi.get(s, greedy_action(Q, s))

        # --- MC 回报计算并更新 Q（Every-Visit）---
        G = 0.0
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            key = (s, a)
            N[key] += 1
            Q[key] += (G - Q[key]) / N[key]

        # --- 策略改进：对本回合出现过的状态做 greedy 改进 ---
        visited_states = {s for (s, _a, _r) in episode}
        for s in visited_states:
            if not env.is_terminal(s):
                pi[s] = greedy_action(Q, s)

    return Q, pi

def render_policy(env, pi):
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    out = []
    for r in range(env.n):
        row = []
        for c in range(env.n):
            s = (r, c)
            if env.is_terminal(s):
                row.append("G")
            else:
                a = pi.get(s, None)
                row.append(arrows[a] if a is not None else "·")
        out.append(" ".join(row))
    return "\n".join(out)

if __name__ == "__main__":
    env = GridWorld(n=4, goal=(3, 3))
    Q, pi = mc_control_exploring_starts(env, episodes=80_000, seed=42)

    print("Learned greedy policy (G=goal):")
    print(render_policy(env, pi))