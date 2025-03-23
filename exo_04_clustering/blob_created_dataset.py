import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

centers = [[2, 2], [-2, -2], [2, -2]]
X, labels_true = make_blobs(
    n_samples=3000,
    centers=centers,
    cluster_std=0.7
)

print(X.shape)
print(labels_true)
print(len(centers))

fig = plt.figure(figsize=(6, 6))
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

# Visualisation initialisation dataset
ax = fig.add_subplot(1, 1, 1)

ax.set_title("3 clusters")

for k, col in zip(range(len(centers)), colors):
    my_members = labels_true == k
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="o", markersize=4, alpha = 1)
for k, col in zip(range(len(centers)), colors):
    ax.plot(
        centers[k][0],
        centers[k][1],
        "o",
        markerfacecolor='#CCC',
        markeredgecolor=col,
        markersize=9,
    )
# ax.set_title("Original")
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_linestyle('dotted')
ax.spines['left'].set_linestyle('dotted')

# Customize the appearance of grid lines (dotted and alpha=0.5)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("3_clusters_initialisation")


# K-means
k_means = KMeans( n_clusters=3, random_state=808)
k_means.fit(X)
print(k_means.cluster_centers_)
print(k_means.score(X))

# Visualisation résultats
plt.figure()
k_means_labels = k_means.predict(X)
print("silhouette_score: ", silhouette_score(X,k_means_labels ))
print("silhouette_score: ", silhouette_score(X,labels_true ))


fig = plt.figure(figsize=(12, 6))
colors = ["#FF0000", "#0000FF", "#00FF00","#111111"]

# KMeans
ax = fig.add_subplot(1, 2, 1)

# for k, col in zip(range(n_clusters), colors):
n_clusters = 3
for k in range(n_clusters):
    ax.plot(X[labels_true == k, 0], X[labels_true == k, 1], "w", markerfacecolor=colors[k], marker="o", markersize=6, alpha = 1)

ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_linestyle('dotted')
ax.spines['left'].set_linestyle('dotted')

# Customize the appearance of grid lines (dotted and alpha=0.5)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("Original")

ax = fig.add_subplot(1, 2, 2)
for k in range(n_clusters):
    cluster_center = k_means.cluster_centers_[k]
    ax.plot(X[k_means_labels == k, 0], X[k_means_labels == k, 1], "w", markerfacecolor=colors[k], marker="o", markersize=6, alpha = 1)
for k in range(n_clusters):
    cluster_center = k_means.cluster_centers_[k]
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor='#000',
        markeredgecolor=colors[k],
        markersize=9,
    )
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_linestyle('dotted')
ax.spines['left'].set_linestyle('dotted')
print('loop')

# Customize the appearance of grid lines (dotted and alpha=0.5)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("KMeans")
plt.tight_layout()
plt.savefig('k_means_results.png')


# Silouhette_score
plt.figure()
centers = [[2, 2], [-2, -2], [2, -2]]
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.9)

silhouette_scores = []
for n_clusters in range(2, 8):
    print()
    print("n_clusters", n_clusters)
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, random_state=808, n_init='auto')
    k_means.fit(X)
    print("score", k_means.score(X))

    k_means_labels = k_means.predict(X)
    print("silhouette_score: ", silhouette_score(X, k_means_labels))
    silhouette_scores.append(silhouette_score(X, k_means_labels))

fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(1, 1, 1)
plt.plot(range(2, 8), silhouette_scores)
ax.set_title('silhouette scores')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('Coefficient de silhouette')
plt.grid()
plt.savefig('siljouette.png')
