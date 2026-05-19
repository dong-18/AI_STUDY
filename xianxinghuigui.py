import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(42)

# 1) 生成合成数据（X: n×3，y: n×1）
n = 500
area = np.random.normal(90, 30, n)            # 面积(㎡)
age = np.random.exponential(10, n)            # 房龄(年)
dist = np.random.normal(8, 3, n)              # 距离市中心(公里)
X = np.vstack([area, age, dist]).T

# “真实”关系（线性 + 噪声）
true_w = np.array([6000, -800, -1200])        # 系数：面积正相关，房龄/距离负相关
true_b = 80_000
noise = np.random.normal(0, 30_000, n)
y = X @ true_w + true_b + noise

# 2) 划分训练/测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3) 训练线性回归
model = LinearRegression()
model.fit(X_train, y_train)

# 4) 评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Learned weights:", model.coef_)
print("Learned bias:", model.intercept_)
print(f"MAE={mae:.0f}, MSE={rmse:.0f}, R^2={r2:.3f}")

# 5) 可视化（预测 vs 真实）
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("House Price: True vs Predicted")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')  # 理想对角线
plt.show()