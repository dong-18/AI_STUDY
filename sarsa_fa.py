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

    def _pos_from_index(self, s):
        """从状态编号获取二维坐标"""
        return (s // self.ncol, s % self.ncol)

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
            reward = -100
            next_pos = self.start
            done = False

        # 到达终点：结束
        if next_pos == self.goal:
            reward = 0
            done = True

        # 更新当前位置
        self.pos = next_pos
        return self._state_index(next_pos), reward, done


# ---------- 特征工程 ----------

class StateActionFeatureVectorizer:
    """
    将 (s, a) 对映射为特征向量 φ(s, a)
    使用简单的 one-hot 编码：每个 (s, a) 对对应一个特征
    对于 4x12 网格和 4 个动作，特征维度 = 48 * 4 = 192
    """
    def __init__(self, nrow=4, ncol=12, n_actions=4):
        self.nrow = nrow
        self.ncol = ncol
        self.n_actions = n_actions
        self.n_states = nrow * ncol
        # 每个状态-动作对一个特征
        self.feature_dim = self.n_states * n_actions

    def transform(self, s, a):
        """
        将 (s, a) 转换为 one-hot 特征向量
        返回: shape (feature_dim,) 的稀疏二值向量
        """
        features = np.zeros(self.feature_dim)
        feature_idx = s * self.n_actions + a
        features[feature_idx] = 1.0
        return features


class LinearSARSAAgent:
    """
    使用线性函数近似的 SARSA 智能体
    Q(s,a) = w^T · φ(s,a)
    """
    def __init__(self, feature_dim, n_actions=4, alpha=0.01, gamma=1.0, epsilon=0.1):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # 权重向量 w：初始化为小的正值，鼓励初始探索
        self.weights = np.ones(feature_dim) * 0.1

    def Q_value(self, features):
        """计算 Q(s,a) = w^T · φ(s,a)"""
        return np.dot(self.weights, features)

    def select_action(self, s, vectorizer):
        """使用 epsilon-greedy 策略选择动作"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # 贪心：选择 Q 值最大的动作
        q_values = []
        for a in range(self.n_actions):
            features = vectorizer.transform(s, a)
            q_values.append(self.Q_value(features))

        max_q = max(q_values)
        # 在最大值中随机选择（处理平局）
        candidates = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(candidates)

    def update(self, s, a, r, s2, a2, vectorizer, done):
        """
        SARSA 更新权重：
        w <- w + alpha * [r + gamma*Q(s',a') - Q(s,a)] * φ(s,a)
        """
        features = vectorizer.transform(s, a)
        q_sa = self.Q_value(features)

        if done:
            # 终止状态，目标就是 r
            td_target = r
        else:
            features_next = vectorizer.transform(s2, a2)
            q_s2a2 = self.Q_value(features_next)
            td_target = r + self.gamma * q_s2a2

        # TD 误差
        td_error = td_target - q_sa

        # 更新权重：只更新当前 (s,a) 对应的特征
        self.weights += self.alpha * td_error * features


def sarsa_with_function_approximation(env, vectorizer, episodes=500,
                                       alpha=0.1, gamma=1.0, epsilon=0.1,
                                       max_steps=500, verbose=True):
    """
    带函数近似的 SARSA

    参数:
        alpha: 学习率（函数近似通常 0.01-0.1）
        max_steps: 单局最大步数（避免死循环）
        verbose: 是否打印进度
    """
    agent = LinearSARSAAgent(
        feature_dim=vectorizer.feature_dim,
        n_actions=4,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )

    returns = []

    for ep in range(episodes):
        s = env.reset()
        a = agent.select_action(s, vectorizer)
        total_reward = 0

        for step in range(max_steps):
            s2, r, done = env.step(a)
            total_reward += r

            if done:
                agent.update(s, a, r, s2, None, vectorizer, done=True)
                break

            # 选择下一个动作 a'
            a2 = agent.select_action(s2, vectorizer)

            # 更新权重
            agent.update(s, a, r, s2, a2, vectorizer, done=False)

            s, a = s2, a2

        returns.append(total_reward)

        # 逐渐衰减 epsilon（重要：帮助收敛）
        decay_rate = ep / episodes
        agent.epsilon = max(0.01, epsilon * (1 - decay_rate))

        # 打印进度
        if verbose and (ep == 0 or (ep + 1) % 100 == 0 or ep == episodes - 1):
            recent_avg = np.mean(returns[max(0, ep - 99):ep + 1])
            print(f"Episode {ep + 1}/{episodes}, "
                  f"回报: {total_reward:3d}, "
                  f"最近100局平均: {recent_avg:6.1f}, "
                  f"ε: {agent.epsilon:.3f}, "
                  f"步数: {step + 1}")

    return agent, np.array(returns)


def visualize_learned_policy(agent, vectorizer, env):
    """从学到的权重中提取并可视化策略"""
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("学到的策略 (S=起点, G=终点, X=悬崖):")
    for r in range(env.nrow):
        row_str = ""
        for c in range(env.ncol):
            pos = (r, c)
            s = env._state_index(pos)

            if pos == env.start:
                row_str += "S "
            elif pos == env.goal:
                row_str += "G "
            elif pos in env.cliff:
                row_str += "X "
            else:
                # 选择 Q 值最大的动作
                best_action = None
                best_q = -np.inf
                for a in range(4):
                    features = vectorizer.transform(s, a)
                    q_val = agent.Q_value(features)
                    if q_val > best_q:
                        best_q = q_val
                        best_action = a
                row_str += arrows[best_action] + " "
        print(row_str)

    # 调试：显示起点处的 Q 值
    print("\n起点 (3,0) 处的 Q 值:")
    start_s = env._state_index(env.start)
    action_names = ["向上", "向右", "向下", "向左"]
    for a in range(4):
        features = vectorizer.transform(start_s, a)
        q_val = agent.Q_value(features)
        print(f"  {action_names[a]}: {q_val:.4f}")


if __name__ == "__main__":
    # 固定随机种子
    np.random.seed(42)

    # 创建环境
    env = CliffWalkingEnv()

    # 创建特征向量化器（使用简单 one-hot 编码）
    vectorizer = StateActionFeatureVectorizer(
        nrow=4,
        ncol=12,
        n_actions=4
    )

    print(f"特征维度: {vectorizer.feature_dim}")
    print(f"状态数: {vectorizer.n_states}")
    print(f"动作数: {vectorizer.n_actions}")
    print()

    # 训练
    agent, returns = sarsa_with_function_approximation(
        env,
        vectorizer,
        episodes=1000,     # 使用 one-hot 编码，收敛更快
        alpha=0.1,         # 学习率
        gamma=1.0,
        epsilon=0.1,       # 初始探索率
        max_steps=500
    )

    # 可视化策略
    visualize_learned_policy(agent, vectorizer, env)

    # 输出训练统计
    print(f"\n最后 20 局回报: {returns[-20:]}")
    print(f"最后 100 局平均回报: {returns[-100:].mean():.2f}")
    print(f"权重统计: mean={agent.weights.mean():.4f}, std={agent.weights.std():.4f}")

    # 测试学到的策略（无探索）
    print("\n=== 测试学到的策略（无探索）===")
    agent.epsilon = 0
    test_rewards = []
    for _ in range(10):
        s = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            a = agent.select_action(s, vectorizer)
            s2, r, done = env.step(a)
            total_reward += r
            s = s2
            steps += 1
        test_rewards.append(total_reward)

    print(f"测试回报 (10局): {test_rewards}")
    print(f"平均测试回报: {np.mean(test_rewards):.2f}")
