import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

def gradient_descent(X, y, weights, alpha, learning_rate, n_iters):
    m = len(y)
    for i in range(n_iters):
        y_pred = np.dot(X, weights)
        gradient = (1/m) * np.dot(X.T, (y_pred - y)) + (alpha/m) * weights
        gradient[0] -= (alpha/m) * weights[0]  # Don't regularize the bias term
        weights -= learning_rate * gradient
    return weights

def ridge_regression(X, y, alpha, learning_rate=0.01, n_iters=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    weights = np.zeros(X_b.shape[1])
    weights = gradient_descent(X_b, y, weights, alpha, learning_rate, n_iters)
    return weights

def predict(X, weights):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return X_b @ weights

# Plotting data points
plt.figure(figsize=(14, 10))
plt.subplot(2, 3, 1)
plt.scatter(X, y, color='blue', edgecolor='k')
plt.title('Data')

alphas = [0, 10, 20, 40, 400]


for i, alpha in enumerate(alphas):
    weights = ridge_regression(X, y, alpha)
    y_pred = predict(X, weights)
    
    plt.subplot(2, 3, i + 2)
    plt.scatter(X, y, color='blue', edgecolor='k')
    plt.plot(X, y_pred, color='red')
    plt.title(f'Ridge Regression\nλ = {alpha}')

plt.tight_layout()
plt.show()

# Ridge regression cost function 
slope_values = np.arange(-30, 30, 1)
plt.figure(figsize=(12, 6))

for alpha in alphas:
    cost_ridge = [(np.sum((y - (slope * X.squeeze() + 1)) ** 2) + alpha * slope ** 2) for slope in slope_values]
    plt.plot(slope_values, cost_ridge, label=f'λ = {alpha}')

plt.title('Ridge Regression Cost Function')
plt.xlabel('Slope Values')
plt.ylabel('Sum of Squared Residuals + λ * Slope^2')
plt.legend()
plt.grid(True)
plt.show()
