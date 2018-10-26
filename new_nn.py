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

val_cutoff = 3*len(X_train1)//4
X_val = X_train1[val_cutoff:]
y_val = y_train1[val_cutoff:]
X_train = X_train1[:val_cutoff]
y_train = y_train1[:val_cutoff]

ten_df = load_tennis_data(form="original df")
ten_features, ten_labels = load_tennis_data(form="df")

ten_cols = ten_features.columns
print(ten_cols)

# Function to tune NN

def best_nn(X, y, X_val, y_val):
    max_score = NEGINF
    best_model = None
    best_h = None
    best_a = None
    #for h in [2,3,4,5,6,7,8,9,10]:
    for h in [5,7,9]:
    #    for a in [0.0001, 0.001, 0.01, 0.1]:
        nn = MLPClassifier(batch_size=16, hidden_layer_sizes=h, alpha=0.001, learning_rate_init=0.01)
        nn.fit(X,y)
        score = nn.score(X_val, y_val)
        if score > max_score:
            max_score = score
            best_model = nn
            best_h = h
            #best_a = a
    return best_model, max_score, best_h

def labels_to_matrix(labels, k):
    matrix = np.zeros((len(labels),k-1))
    for i in range(len(labels)):
        label = labels[i]
        if label != 0:
            matrix[i,label-1] = 1
    return matrix

# Feature transforms

colors = 'bgrcmyk'

# Dimensionality Reduce using PCA
pca_scores_d_by_k = []
ica_scores_d_by_k = []
rca_scores_d_by_k = []
rfe_scores_d_by_k = []

for d in range(1, 4):
    print("dim", d)

    if False:
        # Dimensionlity Reduce using PCA
        pca = PCA(n_components=d)
        pca.fit(X_train)
        pca_X_train = pca.transform(X_train)
        pca_X_val = pca.transform(X_val)
        pca_scores_d_by_k.append([])
        for k in range(2, 8):
            print("pca clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(pca_X_train)
            train_labels = kmm.predict(pca_X_train)
            val_labels = kmm.predict(pca_X_val)
            pca_kmeans_X_train = labels_to_matrix(train_labels, k)
            pca_kmeans_X_val = labels_to_matrix(val_labels, k)
            best_model, best_nn_score, best_nn_h = best_nn(pca_kmeans_X_train, y_train, pca_kmeans_X_val, y_val)
            pca_scores_d_by_k[-1].append(best_nn_score)

    if False:
        # Dimensionlity Reduce using RCA
        rca = GaussianRandomProjection(n_components=d)
        rca.fit(X_train)
        rca_X_train = rca.transform(X_train)
        rca_X_val = rca.transform(X_val)
        rca_scores_d_by_k.append([])
        for k in range(2, 8):
            print("rca clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(rca_X_train)
            train_labels = kmm.predict(rca_X_train)
            val_labels = kmm.predict(rca_X_val)
            rca_kmeans_X_train = labels_to_matrix(train_labels, k)
            rca_kmeans_X_val = labels_to_matrix(val_labels, k)
            best_model, best_nn_score, best_nn_h = best_nn(rca_kmeans_X_train, y_train, rca_kmeans_X_val, y_val)
            rca_scores_d_by_k[-1].append(best_nn_score)

    if True:
        # Dimensionlity Reduce using RFE
        logreg = LogisticRegression(solver='sag')
        rfe = RFE(logreg, n_features_to_select=d)
        rfe.fit(X_train, y_train)
        rfe_X_train = rfe.transform(X_train)
        rfe_X_val = rfe.transform(X_val)
        rfe_scores_d_by_k.append([])
        for k in range(2, 8):
            print("rfe clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(rfe_X_train)
            train_labels = kmm.predict(rfe_X_train)
            val_labels = kmm.predict(rfe_X_val)
            rfe_kmeans_X_train = labels_to_matrix(train_labels, k)
            rfe_kmeans_X_val = labels_to_matrix(val_labels, k)
            best_model, best_nn_score, best_nn_h = best_nn(rfe_kmeans_X_train, y_train, rfe_kmeans_X_val, y_val)
            rfe_scores_d_by_k[-1].append(best_nn_score)

    if False:
        # Dimensionality Reduce using ICA
        ica = FastICA(n_components=d, max_iter=1000000)
        ica.fit(X_train)
        ica_X_train = ica.transform(X_train)
        ica_X_val = ica.transform(X_val)
        ica_scores_d_by_k.append([])
        for k in range(2, 8):
            print("ica clustering", k)
            kmm = KMeans(n_clusters=k)
            kmm.fit(ica_X_train)
            train_labels = kmm.predict(ica_X_train)
            val_labels = kmm.predict(ica_X_val)
            ica_kmeans_X_train = labels_to_matrix(train_labels, k)
            ica_kmeans_X_val = labels_to_matrix(val_labels, k)
            best_model, best_nn_score, best_nn_h = best_nn(ica_kmeans_X_train, y_train, ica_kmeans_X_val, y_val)
            ica_scores_d_by_k[-1].append(best_nn_score)

print(pca_scores_d_by_k)
print(ica_scores_d_by_k)
print(rca_scores_d_by_k)
print(rfe_scores_d_by_k)

#[[0.5100548446069469, 0.5100548446069469, 0.49085923217550276, 0.5100548446069469, 0.5100548446069469, 0.4936014625228519]]
#[[0.5100548446069469, 0.5100548446069469, 0.5100548446069469, 0.5100548446069469, 0.5100548446069469, 0.489945155393053]]
#[[0.5082266910420475, 0.5489031078610603, 0.5301645338208409, 0.5447897623400365, 0.5415904936014625, 0.5557586837294333]]
#[[0.5406764168190128, 0.5776965265082267, 0.5845521023765996, 0.6060329067641682, 0.6046617915904936, 0.6046617915904936]]





