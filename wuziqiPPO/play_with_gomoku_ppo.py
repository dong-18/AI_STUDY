import numpy as np  # 导入 numpy，用于棋盘数组处理
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class GomokuEnv:  # 定义五子棋环境
    def __init__(self, board_size=9, win_length=5):  # 初始化环境
        self.board_size = board_size  # 保存棋盘大小
        self.win_length = win_length  # 保存获胜所需连子数
        self.action_size = board_size * board_size  # 动作总数
        self.reset()  # 初始化棋盘

    def reset(self):  # 重置棋盘
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # 0 表示空位
        self.current_player = 1  # 当前玩家，1 表示先手，-1 表示后手
        self.done = False  # 对局是否结束
        self.winner = 0  # 胜者，0 表示暂无或平局
        return self.get_obs()  # 返回初始观察

    def get_obs(self):  # 获取当前观察
        cur = (self.board == self.current_player).astype(np.float32)  # 当前玩家自己的棋子
        opp = (self.board == -self.current_player).astype(np.float32)  # 对手棋子
        player_plane = np.ones_like(cur, dtype=np.float32)  # 当前玩家平面，全 1
        obs = np.stack([cur, opp, player_plane], axis=0)  # 拼成 3 通道输入
        return obs  # 返回观察

    def legal_actions_mask(self):  # 获取合法动作掩码
        return (self.board.reshape(-1) == 0).astype(np.float32)  # 空位为 1，非空位为 0

    def step(self, action):  # 执行动作
        if self.done:  # 如果对局已经结束
            raise ValueError("Game is already done.")  # 抛出异常

        r, c = divmod(action, self.board_size)  # 将一维动作转成二维坐标

        if self.board[r, c] != 0:  # 如果该位置不为空
            self.done = True  # 直接结束
            self.winner = -self.current_player  # 当前玩家非法落子，判负
            return self.get_obs(), -1.0, self.done, {"illegal": True}  # 返回结果

        self.board[r, c] = self.current_player  # 落子

        if self.check_win(r, c, self.current_player):  # 检查是否获胜
            self.done = True  # 标记结束
            self.winner = self.current_player  # 记录胜者
            return self.get_obs(), 1.0, self.done, {"illegal": False}  # 返回结果

        if np.all(self.board != 0):  # 如果棋盘已满
            self.done = True  # 标记结束
            self.winner = 0  # 平局
            return self.get_obs(), 0.0, self.done, {"illegal": False}  # 返回结果

        self.current_player *= -1  # 切换玩家
        return self.get_obs(), 0.0, self.done, {"illegal": False}  # 返回下一状态

    def check_win(self, r, c, player):  # 检查是否形成五连
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向
        for dr, dc in directions:  # 遍历四个方向
            count = 1  # 当前落子本身算一个
            count += self.count_direction(r, c, dr, dc, player)  # 正方向连续数量
            count += self.count_direction(r, c, -dr, -dc, player)  # 反方向连续数量
            if count >= self.win_length:  # 如果达到连子要求
                return True  # 获胜
        return False  # 否则未获胜

    def count_direction(self, r, c, dr, dc, player):  # 统计某一方向连续棋子数
        cnt = 0  # 初始化计数
        rr, cc = r + dr, c + dc  # 从下一格开始检查
        while 0 <= rr < self.board_size and 0 <= cc < self.board_size and self.board[rr, cc] == player:  # 沿方向连续检查
            cnt += 1  # 数量加一
            rr += dr  # 行前进
            cc += dc  # 列前进
        return cnt  # 返回统计结果


class PolicyValueNet(nn.Module):  # 定义策略价值网络
    def __init__(self, board_size=9):  # 初始化网络
        super().__init__()  # 调用父类初始化
        self.board_size = board_size  # 保存棋盘大小
        self.action_size = board_size * board_size  # 保存动作总数

        self.encoder = nn.Sequential(  # 特征提取层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 第一层卷积
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第二层卷积
            nn.ReLU(),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第三层卷积
            nn.ReLU(),  # 激活函数
        )  # 编码器结束

        self.policy_head = nn.Sequential(  # 策略头
            nn.Conv2d(64, 2, kernel_size=1),  # 1x1 卷积降维
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 展平
            nn.Linear(2 * board_size * board_size, self.action_size),  # 输出每个动作的 logits
        )  # 策略头结束

        self.value_head = nn.Sequential(  # 价值头
            nn.Conv2d(64, 1, kernel_size=1),  # 1x1 卷积
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 展平
            nn.Linear(board_size * board_size, 64),  # 全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 1),  # 输出单个价值
        )  # 价值头结束

    def forward(self, x):  # 前向传播
        feat = self.encoder(x)  # 提取特征
        logits = self.policy_head(feat)  # 策略输出
        value = self.value_head(feat).squeeze(-1)  # 价值输出
        return logits, value  # 返回 logits 和 value


def print_board(board):  # 打印棋盘
    size = board.shape[0]  # 获取棋盘大小

    print("\n   " + " ".join([f"{i:2d}" for i in range(size)]))  # 打印列号
    for r in range(size):  # 遍历每一行
        row_str = f"{r:2d} "  # 行号
        for c in range(size):  # 遍历每一列
            if board[r, c] == 1:  # 如果是先手棋子
                row_str += " X "  # 用 X 表示
            elif board[r, c] == -1:  # 如果是后手棋子
                row_str += " O "  # 用 O 表示
            else:  # 否则为空位
                row_str += " . "  # 用 . 表示空位
        print(row_str)  # 打印整行
    print()  # 换行


def select_ai_action(model, obs, legal_mask, device):  # 让 AI 选择动作
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # 转 tensor 并增加 batch 维
        legal_tensor = torch.tensor(legal_mask, dtype=torch.float32, device=device).unsqueeze(0)  # mask 转 tensor

        logits, value = model(obs_tensor)  # 前向计算
        logits = logits.masked_fill(legal_tensor < 0.5, -1e9)  # 屏蔽非法动作
        action = torch.argmax(logits, dim=-1).item()  # 选择概率最大的动作

    return action  # 返回 AI 动作


def human_action(env):  # 读取人类输入动作
    while True:  # 循环直到输入合法
        try:  # 尝试解析输入
            text = input("请输入落子坐标，格式：row col，例如 3 4：")  # 提示用户输入
            r, c = map(int, text.strip().split())  # 解析为两个整数

            if not (0 <= r < env.board_size and 0 <= c < env.board_size):  # 检查是否越界
                print("坐标越界，请重新输入。")  # 提示错误
                continue  # 继续输入

            if env.board[r, c] != 0:  # 检查是否为空位
                print("该位置已有棋子，请重新输入。")  # 提示错误
                continue  # 继续输入

            action = r * env.board_size + c  # 转成一维动作编号
            return action  # 返回动作

        except Exception:  # 如果输入格式错误
            print("输入格式错误，请按 'row col' 输入，例如：3 4")  # 提示重新输入


def play_game(model_path, board_size=9, human_first=True):  # 进行一局人机对战
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    print("使用设备：", device)  # 打印设备信息

    env = GomokuEnv(board_size=board_size, win_length=5)  # 创建环境
    model = PolicyValueNet(board_size=board_size).to(device)  # 创建网络
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
    model.eval()  # 切换评估模式

    obs = env.reset()  # 重置环境

    human_player = 1 if human_first else -1  # 设定人类执子方
    ai_player = -human_player  # AI 执子方

    print("对战开始！")  # 打印提示
    print(f"你执 {'X(先手)' if human_player == 1 else 'O(后手)'}")  # 打印人类身份
    print(f"AI执 {'X(先手)' if ai_player == 1 else 'O(后手)'}")  # 打印 AI 身份

    while not env.done:  # 只要没结束就继续
        print_board(env.board)  # 打印当前棋盘

        if env.current_player == human_player:  # 如果当前轮到人类
            action = human_action(env)  # 读取人类动作
            _, _, _, info = env.step(action)  # 执行动作
            if info.get("illegal", False):  # 理论上不会进入，因为前面已经检查过
                print("你下了非法位置，判负。")  # 提示非法动作
                break  # 结束对局
        else:  # 否则轮到 AI
            legal_mask = env.legal_actions_mask()  # 获取合法动作掩码
            action = select_ai_action(model, obs, legal_mask, device)  # AI 选择动作
            r, c = divmod(action, env.board_size)  # 转成坐标
            print(f"AI 落子：{r} {c}")  # 打印 AI 的落子位置
            _, _, _, info = env.step(action)  # 执行 AI 动作
            if info.get("illegal", False):  # 正常不会发生
                print("AI 下了非法动作，AI 判负。")  # 提示错误
                break  # 结束对局

        obs = env.get_obs()  # 更新当前观察

    print_board(env.board)  # 打印最终棋盘

    if env.winner == 0:  # 如果平局
        print("对局结束：平局")  # 输出平局
    elif env.winner == human_player:  # 如果人类赢了
        print("恭喜你，你赢了！")  # 输出人类胜利
    else:  # 否则 AI 获胜
        print("AI 赢了。")  # 输出 AI 胜利


if __name__ == "__main__":  # 主程序入口
    model_path = "gomoku_ppo_iter_500.pth"  # 你的模型文件路径
    play_game(model_path=model_path, board_size=9, human_first=True)  # 开始对战