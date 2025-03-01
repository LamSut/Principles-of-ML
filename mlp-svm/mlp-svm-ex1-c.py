import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# training dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, 1])

# train SVM model
svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)

# plot 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=100)

xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['dashed', 'solid', 'dashed'])
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=200, edgecolors='k', facecolors='none', linewidths=2)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM Classification Model")
plt.show()
