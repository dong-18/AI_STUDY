import numpy as np

def sigmoid(z):
    # 数值稳定性：截断 z，避免 exp 溢出
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, num_iters=2000, reg_lambda=0.0, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iters = num_iters
        self.reg_lambda = reg_lambda
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        # 这里不实际添加截距列，而是用单独的 b 参数
        return X

    def fit(self, X, y):
        X = self._add_intercept(X)
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for i in range(self.num_iters):
            z = X @ self.w + self.b
            y_hat = sigmoid(z)

            # 计算梯度
            error = (y_hat - y)  # shape: (m,)
            grad_w = (X.T @ error) / m + (self.reg_lambda / m) * self.w
            grad_b = np.sum(error) / m

            # 参数更新
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if self.verbose and (i % 200 == 0 or i == self.num_iters - 1):
                # 数值稳定地计算损失
                eps = 1e-12
                loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)) \
                       + (self.reg_lambda / (2 * m)) * np.sum(self.w ** 2)
                print(f"iter {i:4d} | loss={loss:.6f}")

        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# 示例：使用合成数据测试
if __name__ == "__main__":
    np.random.seed(42)
    m, n = 800, 5
    X = np.random.randn(m, n)

    # 真实参数
    true_w = np.array([0.8, -1.2, 0.5, 0.0, 1.5])
    true_b = -0.3
    logits = X @ true_w + true_b
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(m) < probs).astype(int)

    # 划分训练/测试集
    idx = np.arange(m)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:600], idx[600:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 训练
    model = LogisticRegressionScratch(lr=0.1, num_iters=3000, reg_lambda=0.01, verbose=True)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    # 查看前10个样本的预测概率
    print("First 10 predicted probabilities:", model.predict_proba(X_test)[:10])