from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

NEGINF = -float("inf")

# IMPORT DATA

X_train1, y_train1, X_test, y_test = load_tennis_data()

# Split off validation set from training set

val_cutoff2 = len(X_train1)-1000
val_cutoff1 = val_cutoff2-1000

X_val1 = X_train1[val_cutoff1:val_cutoff2]
y_val1 = y_train1[val_cutoff1:val_cutoff2]
X_val2 = X_train1[val_cutoff2:]
y_val2 = y_train1[val_cutoff2:]

X_train = X_train1[:val_cutoff1]
y_train = y_train1[:val_cutoff1]

# print(X_train.shape[0])
# print(y_train.shape[0])
# print(X_val1.shape[0])
# print(y_val1.shape[0])
# print(X_val2.shape[0])
# print(y_val2.shape[0])
# print(X_test.shape[0])
# print(y_test.shape[0])

ten_df = load_tennis_data(form="original df")
ten_features, ten_labels = load_tennis_data(form="df")

ten_cols = ten_features.columns
print(ten_cols)

# Function to tune NN

def best_nn(X, y, X_val1, y_val1, X_val2, y_val2):
    max_score = NEGINF
    best_model = None
    best_h = None
    #best_a = None
    #for h in [2,3,4,5,6,7,8,9,10]:
    for h in [3,5,7,9,11]:
    #    for a in [0.0001, 0.001, 0.01, 0.1]:
        nn = MLPClassifier(batch_size=16, hidden_layer_sizes=h, alpha=0.001, learning_rate_init=0.01)
        nn.fit(X,y)
        score = nn.score(X_val1, y_val1)
        if score > max_score:
            max_score = score
            best_model = nn
            best_h = h
            #best_a = a
    new_score = nn.score(X_val2, y_val2)
    return best_model, new_score, best_h

def labels_to_matrix(labels, k):
    matrix = np.zeros((len(labels),k-1))
    for i in range(len(labels)):
        label = labels[i]
        if label != 0:
            matrix[i,label-1] = 1
    return matrix

# Feature transforms

# Dimensionality Reduce using PCA
pca_scores_d_by_k = []
ica_scores_d_by_k = []
rca_scores_d_by_k = []
rfe_scores_d_by_k = []

for d in range(2, 3):
    print("dim", d)
    ks = range(2, 16)

    if False:
        # Dimensionlity Reduce using PCA
        pca = PCA(n_components=d)
        pca.fit(X_train)
        pca_X_train = pca.transform(X_train)
        pca_X_val1 = pca.transform(X_val1)
        pca_X_val2 = pca.transform(X_val2)
        pca_scores_d_by_k.append([])
        for k in ks:
            print("pca clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(pca_X_train)
            train_labels = kmm.predict(pca_X_train)
            val_labels1 = kmm.predict(pca_X_val1)
            val_labels2 = kmm.predict(pca_X_val2)
            pca_kmeans_X_train = labels_to_matrix(train_labels, k)
            pca_kmeans_X_val1 = labels_to_matrix(val_labels1, k)
            pca_kmeans_X_val2 = labels_to_matrix(val_labels2, k)
            best_model, best_nn_score, best_nn_h = best_nn(pca_kmeans_X_train, y_train, pca_kmeans_X_val1, y_val1, pca_kmeans_X_val2, y_val2)
            print("nodes in hidden layer", best_nn_h)
            pca_scores_d_by_k[-1].append(best_nn_score)

    if True:
        # Dimensionlity Reduce using RCA
        rca = GaussianRandomProjection(n_components=d)
        rca.fit(X_train)
        rca_X_train = rca.transform(X_train)
        rca_X_val1 = rca.transform(X_val1)
        rca_X_val2 = rca.transform(X_val2)
        rca_scores_d_by_k.append([])
        for k in ks:
            print("rca clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(rca_X_train)
            train_labels = kmm.predict(rca_X_train)
            val_labels1 = kmm.predict(rca_X_val1)
            val_labels2 = kmm.predict(rca_X_val2)
            rca_kmeans_X_train = labels_to_matrix(train_labels, k)
            rca_kmeans_X_val1 = labels_to_matrix(val_labels1, k)
            rca_kmeans_X_val2 = labels_to_matrix(val_labels2, k)
            best_model, best_nn_score, best_nn_h = best_nn(rca_kmeans_X_train, y_train, rca_kmeans_X_val1, y_val1, rca_kmeans_X_val2, y_val2)
            print("nodes in hidden layer", best_nn_h)
            rca_scores_d_by_k[-1].append(best_nn_score)

    if False:
        # Dimensionlity Reduce using RFE
        logreg = LogisticRegression(solver='sag')
        rfe = RFE(logreg, n_features_to_select=d)
        rfe.fit(X_train, y_train)
        rfe_X_train = rfe.transform(X_train)
        rfe_X_val1 = rfe.transform(X_val1)
        rfe_X_val2 = rfe.transform(X_val2)
        rfe_scores_d_by_k.append([])
        for k in ks:
            print("rfe clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(rfe_X_train)
            train_labels = kmm.predict(rfe_X_train)
            val_labels1 = kmm.predict(rfe_X_val1)
            val_labels2 = kmm.predict(rfe_X_val2)
            rfe_kmeans_X_train = labels_to_matrix(train_labels, k)
            rfe_kmeans_X_val1 = labels_to_matrix(val_labels1, k)
            rfe_kmeans_X_val2 = labels_to_matrix(val_labels2, k)
            best_model, best_nn_score, best_nn_h = best_nn(rfe_kmeans_X_train, y_train, rfe_kmeans_X_val1, y_val1, rfe_kmeans_X_val2, y_val2)
            print("nodes in hidden layer", best_nn_h)
            rfe_scores_d_by_k[-1].append(best_nn_score)

    if False:
        # Dimensionality Reduce using ICA
        ica = FastICA(n_components=d, max_iter=1000000)
        ica.fit(X_train)
        ica_X_train = ica.transform(X_train)
        ica_X_val1 = ica.transform(X_val1)
        ica_X_val2 = ica.transform(X_val2)
        ica_scores_d_by_k.append([])
        for k in ks:
            print("ica clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(ica_X_train)
            train_labels = kmm.predict(ica_X_train)
            val_labels1 = kmm.predict(ica_X_val1)
            val_labels2 = kmm.predict(ica_X_val2)
            ica_kmeans_X_train = labels_to_matrix(train_labels, k)
            ica_kmeans_X_val1 = labels_to_matrix(val_labels1, k)
            ica_kmeans_X_val2 = labels_to_matrix(val_labels2, k)
            best_model, best_nn_score, best_nn_h = best_nn(ica_kmeans_X_train, y_train, ica_kmeans_X_val1, y_val1, ica_kmeans_X_val2, y_val2)
            print("nodes in hidden layer", best_nn_h)
            ica_scores_d_by_k[-1].append(best_nn_score)

print(pca_scores_d_by_k)
print(ica_scores_d_by_k)
print(rca_scores_d_by_k)
print(rfe_scores_d_by_k)



