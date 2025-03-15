import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load train & test datasets
train_data = pd.read_csv("../data/iris/iris.trn", header=None)
test_data = pd.read_csv("../data/iris/iris.tst", header=None)

# combine train & test data
data = pd.concat([train_data, test_data], ignore_index=True)

# remove label
X = data.iloc[:, :-1] if data.shape[1] > 4 else data

# normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# plot
df = pd.DataFrame(X_scaled, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
df['Cluster'] = clusters

sns.pairplot(df, hue="Cluster", palette="viridis", diag_kind="hist")
plt.suptitle("K-Means Clustering on Iris Dataset", y=1.02)
plt.show()
