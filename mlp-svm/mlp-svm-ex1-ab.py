import numpy as np
import matplotlib.pyplot as plt

# training dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, 1])

w = np.array([-0.5, 0.4, 0.5])  # w0, w1, w2
learning_rate = 0.2
epochs = 10

# train perceptron model
for epoch in range(epochs):
    for i in range(len(X)):
        x_i = np.insert(X[i], 0, 1)  #
        y_pred = 1 if np.dot(w, x_i) > 0 else -1 
        if y_pred != y[i]:
            w += learning_rate * (y[i] - y_pred) * x_i

print(f"Final weights: {w}")

# plot
plt.figure(figsize=(6, 6))
for i, point in enumerate(X):
    plt.scatter(point[0], point[1], color='blue' if y[i] == -1 else 'red', s=100, edgecolors='k')

x_vals = np.linspace(-0.1, 1.1, 100)
y_vals = -(w[1] * x_vals + w[0]) / w[2]
plt.plot(x_vals, y_vals, 'g-', label="Perceptron Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Perceptron Classification")
plt.grid()
plt.show()
