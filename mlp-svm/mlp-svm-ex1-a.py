import numpy as np
import matplotlib.pyplot as plt

# training dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, 1])

# w0, w1, w2
w = np.array([-0.5, 0.4, 0.5])
# learning rate
eta = 0.2
# number of epochs
epochs = 10

# training perceptron model
for epoch in range(epochs):
    for i in range(len(X)):
        x_i = np.insert(X[i], 0, 1)
        activation = np.dot(w, x_i)
        y_pred = 1 if activation > 0 else -1
        if y_pred != y[i]:
            w += eta * (y[i] - y_pred) * x_i

# plot training data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=100)

# set labels
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Classification Model")
plt.show()

# plot decision boundary
x_vals = np.linspace(-0.2, 1.2, 100)
y_vals = -(w[1] / w[2]) * x_vals - (w[0] / w[2])
plt.plot(x_vals, y_vals, 'k-')
