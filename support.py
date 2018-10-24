from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

def k_means_2d(X, k, cols, xs, ys, clusters=[]):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = list(kmeans.labels_)

    if len(clusters) > 0:
        for i in range(len(labels)):
            label = labels[i]
            flag = True
            default_label = None
            if label not in clusters:
                if flag:
                    default_label = label
                    flag = False
                else:
                    labels[i] = default_label

    for x,y in zip(xs,ys):
        plt.scatter(X[:,x], X[:,y], c=labels)
        plt.xlabel(cols[x])
        plt.ylabel(cols[y])
        plt.title("k means, k =" + str(k))
        plt.show()

def k_means_per_feature(X, k, cols, box=True, scatter=True):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = list(kmeans.labels_)

    l = X.shape[0]

    for j in range(len(cols)):
        values_counts = []
        for _ in range(k):
            values_counts.append([])
        for p in range(k):
            values_counts[p] = {}
        for i in range(l):
            try:
                values_counts[labels[i]][str(X[i,j]//100)] += 1
            except:
                values_counts[labels[i]][str(X[i,j]//100)] = 0
        counts_per_cluster = [values_counts[c][str(v//100)] for c,v in zip(labels,X[:,j])]

        label_data = []
        for _ in range(k):
            label_data.append([])
        for i in range(l):
            label_data[labels[i]].append(X[i,j])

        if box:
            plt.boxplot(label_data)
            plt.ylabel(cols[j])
            plt.xlabel("cluster")
            plt.title("k means, k =" + str(k))
            plt.show()

        if scatter:
            plt.scatter(labels, X[:,j], c=labels, s=counts_per_cluster)
            plt.xlabel("cluster")
            plt.ylabel(cols[j])
            plt.title("k means, k =" + str(k))
            plt.show()

def em_2d(X, k, cols, xs, ys, clusters=[]):
    em = GaussianMixture(n_components=k)
    em.fit(X)
    labels = em.predict(X)

    if len(clusters) > 0:
        for i in range(len(labels)):
            label = labels[i]
            flag = True
            default_label = None
            if label not in clusters:
                if flag:
                    default_label = label
                    flag = False
                else:
                    labels[i] = default_label

    for x,y in zip(xs,ys):
        plt.scatter(X[:,x], X[:,y], c=labels)
        plt.xlabel(cols[x])
        plt.ylabel(cols[y])
        plt.title("exp max, k =" + str(k))
        plt.show()

def em_per_feature(X, k, cols, box=True, scatter=True):
    em = GaussianMixture(n_components=k)
    em.fit(X)
    labels = em.predict(X)

    l = X.shape[0]

    for j in range(len(cols)):
        values_counts = []
        for _ in range(k):
            values_counts.append([])
        for p in range(k):
            values_counts[p] = {}
        for i in range(l):
            try:
                values_counts[labels[i]][str(X[i,j]//100)] += 1
            except:
                values_counts[labels[i]][str(X[i,j]//100)] = 0
        counts_per_cluster = [values_counts[c][str(v//100)] for c,v in zip(labels,X[:,j])]

        label_data = []
        for _ in range(k):
            label_data.append([])
        for i in range(l):
            label_data[labels[i]].append(X[i,j])

        if box:
            plt.boxplot(label_data)
            plt.ylabel(cols[j])
            plt.xlabel("cluster")
            plt.title("exp max, k =" + str(k))
            plt.show()

        if scatter:
            plt.scatter(labels, X[:,j], c=labels, s=counts_per_cluster)
            plt.xlabel("cluster")
            plt.ylabel(cols[j])
            plt.title("exp max, k =" + str(k))
            plt.show()

def k_means_one_feature(X, k, cols, feature_idx, box=True, scatter=True):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = list(kmeans.labels_)

    l = X.shape[0]

    j = feature_idx
    values_counts = []
    for _ in range(k):
        values_counts.append([])
    for p in range(k):
        values_counts[p] = {}
    for i in range(l):
        try:
            values_counts[labels[i]][str(X[i,j]//100)] += 1
        except:
            values_counts[labels[i]][str(X[i,j]//100)] = 0
    counts_per_cluster = [values_counts[c][str(v//100)] for c,v in zip(labels,X[:,j])]

    label_data = []
    for _ in range(k):
        label_data.append([])
    for i in range(l):
        label_data[labels[i]].append(X[i,j])

    if box:
        plt.boxplot(label_data)
        plt.ylabel(cols[j])
        plt.xlabel("cluster")
        plt.title("k means, k =" + str(k))
        plt.show()

    if scatter:
        plt.scatter(labels, X[:,j], c=labels, s=counts_per_cluster)
        plt.xlabel("cluster")
        plt.ylabel(cols[j])
        plt.title("k means, k =" + str(k))
        plt.show()

def em_one_feature(X, k, cols, feature_idx, box=True, scatter=True):
    em = GaussianMixture(n_components=k)
    em.fit(X)
    labels = em.predict(X)

    l = X.shape[0]

    j = feature_idx
    values_counts = []
    for _ in range(k):
        values_counts.append([])
    for p in range(k):
        values_counts[p] = {}
    for i in range(l):
        try:
            values_counts[labels[i]][str(X[i,j]//100)] += 1
        except:
            values_counts[labels[i]][str(X[i,j]//100)] = 0
    counts_per_cluster = [values_counts[c][str(v//100)] for c,v in zip(labels,X[:,j])]

    label_data = []
    for _ in range(k):
        label_data.append([])
    for i in range(l):
        label_data[labels[i]].append(X[i,j])

    if box:
        plt.boxplot(label_data)
        plt.ylabel(cols[j])
        plt.xlabel("cluster")
        plt.title("exp max, k =" + str(k))
        plt.show()

    if scatter:
        plt.scatter(labels, X[:,j], c=labels, s=counts_per_cluster)
        plt.xlabel("cluster")
        plt.ylabel(cols[j])
        plt.title("exp max, k =" + str(k))
        plt.show()
