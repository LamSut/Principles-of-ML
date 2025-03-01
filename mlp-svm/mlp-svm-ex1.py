import numpy as np
import matplotlib.pyplot as plt

# Training dataset (X1, X2, class)
data = np.array([
    [0, 0, -1],
    [0, 1, +1],
    [1, 0, +1],
    [1, 1, +1]
])

X = data[:, :2]  # Features (X1, X2)
y = data[:, 2]   # Labels (-1 or +1)

# Initialize weights
w = np.array([-0.5, 0.4, 0.5])  # [w0 (bias), w1, w2]
learning_rate = 0.2

# Perceptron learning algorithm
def perceptron_train(X, y, w, learning_rate, epochs=10):
    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)  # Add bias input (x0 = 1)
            y_pred = np.sign(np.dot(w, x_i))
            if y_pred != y[i]:
                w += learning_rate * y[i] * x_i  # Weight update
    return w

# Train the perceptron
w_trained = perceptron_train(X, y, w, learning_rate)
print("Trained weights:", w_trained)

# Visualization
def plot_decision_boundary(X, y, w):
    plt.figure(figsize=(6, 6))
    for i, label in enumerate(y):
        if label == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o', label='+1' if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red', marker='x', label='-1' if i == 0 else "")
    
    # Decision boundary
    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = -(w[1] / w[2]) * x_vals - (w[0] / w[2])
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
    plt.show()

plot_decision_boundary(X, y, w_trained)
