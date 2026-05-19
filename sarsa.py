import numpy as np

# ---------- GridWorld: 悬崖漫步 CliffWalking ----------
# 4x12 网格：
# 起点 start=(3,0)，终点 goal=(3,11)
# 悬崖格子为 (3,1..10)：踩到悬崖 -> 奖励 -100，并被传送回起点（episode 不结束）
# 普通走一步奖励 -1；到达终点奖励 0 且 episode 结束

class CliffWalkingEnv:
    def __init__(self, nrow=4, ncol=12):
        self.nrow = nrow
        self.ncol = ncol
        self.start = (nrow - 1, 0)
        self.goal = (nrow - 1, ncol - 1)
        self.cliff = {(nrow - 1, c) for c in range(1, ncol - 1)}
        self.reset()

    def reset(self):
        """重置环境到起点，返回状态编号 s"""
        self.pos = self.start
        return self._state_index(self.pos)

    def _state_index(self, pos):
        """把二维坐标 (r,c) 映射到一维状态编号 s"""
        r, c = pos
        return r * self.ncol + c

    def step(self, action):
        """
        执行动作，返回 (s', r, done)
        action: 0=上, 1=右, 2=下, 3=左
        """
        r, c = self.pos

        # 根据动作更新坐标
        if action == 0:   r -= 1
        elif action == 1: c += 1
        elif action == 2: r += 1
        elif action == 3: c -= 1

        # 边界裁剪：出界则停在边界
        r = int(np.clip(r, 0, self.nrow - 1))
        c = int(np.clip(c, 0, self.ncol - 1))
        next_pos = (r, c)

        # 默认：普通移动一步
        reward = -1
        done = False

        # 踩到悬崖：重罚并回到起点（不结束）
        if next_pos in self.cliff:
            reward = 0
            next_pos = self.start
            done = False

        # 到达终点：结束
        if next_pos == self.goal:
            reward = 0
            done = True

        # 更新当前位置
        self.pos = next_pos
        return self._state_index(next_pos), reward, done


# ---------- SARSA 智能体 ----------
def epsilon_greedy(Q, s, eps):
    """
    epsilon-greedy 选动作（on-policy 的关键：行为策略就是用来采样数据的策略）
    - 以 eps 概率随机探索
    - 否则选择当前 Q(s,·) 最大的动作（并在并列最大值中随机打破平局）
    """
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])

    max_q = np.max(Q[s])
    candidates = np.flatnonzero(Q[s] == max_q)
    return np.random.choice(candidates)


def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, max_steps=10_000):
    """
    SARSA (on-policy TD control)
    更新公式：
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma*Q(s',a') - Q(s,a) ]
    其中 a' 是在 s' 上用同一个 epsilon-greedy 策略选出来的动作（这就是 on-policy）
    """
    nS = env.nrow * env.ncol  # 状态数
    nA = 4                   # 动作数：上下左右
    Q = np.zeros((nS, nA), dtype=np.float64)  # 动作价值函数表
    returns = []  # 每个 episode 的累计回报，便于观察训练效果

    for ep in range(episodes):
        # 初始化一局：从起点开始
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)  # 在起始状态选择起始动作
        total_reward = 0

        for _ in range(max_steps):
            # 执行动作 a，得到下一状态与奖励
            s2, r, done = env.step(a)
            total_reward += r

            if done:
                # 若到终点：目标值就是 r（终止状态没有后续 Q）
                Q[s, a] += alpha * (r - Q[s, a])
                break

            # SARSA 的关键：在 s' 上继续用同一个行为策略选 a'
            a2 = epsilon_greedy(Q, s2, epsilon)

            # TD 目标：r + gamma * Q(s', a')
            td_target = r + gamma * Q[s2, a2]
            # TD 误差：td_target - Q(s,a)
            Q[s, a] += alpha * (td_target - Q[s, a])

            # 状态/动作推进： (s,a) <- (s',a')
            s, a = s2, a2

        returns.append(total_reward)

    return Q, np.array(returns)


def greedy_policy_from_Q(Q, nrow=4, ncol=12):
    """从 Q 表导出贪心策略（仅用于展示学习结果）"""
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    pol = []
    for r in range(nrow):
        row = []
        for c in range(ncol):
            s = r * ncol + c
            a = int(np.argmax(Q[s]))
            row.append(arrows[a])
        pol.append(row)
    return pol


def print_policy(env, policy):
    """
    打印策略：
    S 起点，G 终点，X 悬崖，其余格子打印动作箭头
    """
    for r in range(env.nrow):
        out = []
        for c in range(env.ncol):
            pos = (r, c)
            if pos == env.start:
                out.append("S")
            elif pos == env.goal:
                out.append("G")
            elif pos in env.cliff:
                out.append("X")
            else:
                out.append(policy[r][c])
        print(" ".join(out))


if __name__ == "__main__":
    # 固定随机种子便于复现
    np.random.seed(0)

    env = CliffWalkingEnv()

    # 训练 SARSA
    Q, returns = sarsa(
        env,
        episodes=8000,
        alpha=0.1,
        gamma=1.0,
        epsilon=0.1
    )

    # 展示最终贪心策略
    policy = greedy_policy_from_Q(Q, env.nrow, env.ncol)
    print_policy(env, policy)

    # 输出训练回报信息
    print("\n最后 20 局回报：", returns[-20:])
    print("最后 100 局平均回报：", returns[-100:].mean())