import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load training and test datasets
train_data = pd.read_csv("../data/iris/iris.trn", header=None)
test_data = pd.read_csv("../data/iris/iris.tst", header=None)

# Combine train and test data (assuming they have the same format)
data = pd.concat([train_data, test_data], ignore_index=True)

# Assuming the last column is the label (removing it if present)
X = data.iloc[:, :-1] if data.shape[1] > 4 else data

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Convert to DataFrame for visualization
df = pd.DataFrame(X_scaled, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
df['Cluster'] = clusters

# Pairplot visualization
sns.pairplot(df, hue="Cluster", palette="viridis", diag_kind="hist")
plt.suptitle("K-Means Clustering on Iris Dataset", y=1.02)
plt.show()
