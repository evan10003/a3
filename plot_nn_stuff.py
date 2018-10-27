import matplotlib.pyplot as plt
import numpy as np

pca_kmeans_nn = [[0.506, 0.506, 0.494, 0.491, 0.506, 0.506, 0.494, 0.494, 0.494, 0.488, 0.488, 0.506, 0.494, 0.474], [0.506, 0.582, 0.578, 0.589, 0.593, 0.595, 0.602, 0.621, 0.623, 0.6, 0.622, 0.599, 0.593, 0.587], [0.484, 0.593, 0.583, 0.555, 0.575, 0.603, 0.588, 0.59, 0.593, 0.587, 0.579, 0.584, 0.591, 0.613], [0.494, 0.572, 0.564, 0.577, 0.596, 0.609, 0.602, 0.582, 0.617, 0.62, 0.617, 0.614, 0.602, 0.602], [0.494, 0.576, 0.57, 0.556, 0.584, 0.587, 0.574, 0.6, 0.582, 0.591, 0.593, 0.583, 0.612, 0.601], [0.494, 0.592, 0.578, 0.586, 0.588, 0.588, 0.583, 0.584, 0.562, 0.561, 0.581, 0.579, 0.596, 0.608]]
pca_kmeans_nn = np.array(pca_kmeans_nn)

rca_kmeans_nn = [[0.569, 0.571, 0.571, 0.555, 0.566, 0.563, 0.568, 0.566, 0.564, 0.568, 0.552, 0.57, 0.561, 0.571], [0.57, 0.54, 0.572, 0.538, 0.576, 0.579, 0.579, 0.553, 0.579, 0.566, 0.566, 0.585, 0.58, 0.585]]

dims = pca_kmeans_nn.shape[0]
cluster_nums = pca_kmeans_nn.shape[1]

plt.figure()

maxi = np.amax(pca_kmeans_nn)
mini = np.amin(pca_kmeans_nn)
for i in range(dims):
    for j in range(cluster_nums):
        pca_kmeans_nn[i,j] = ((pca_kmeans_nn[i,j]-mini)/(maxi-mini))

a = np.random.rand(dims, cluster_nums)

for i in range(dims):
    print(pca_kmeans_nn[i,:])
    plt.scatter(list(range(2,2+cluster_nums)), [1+i]*cluster_nums, s=(pca_kmeans_nn[i,:]**4)*300)

plt.title("PCA --> kmeans --> NN val scores")
plt.xlabel("number of clusters")
plt.ylabel("number of dimensions")
plt.savefig("pca_kmeans_nn.png")
