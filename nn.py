from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

# IMPORT DATA

tit_X_train1, tit_y_train1, tit_X_test, tit_y_test = load_titanic_data()

# Split off validation set from training set
val_cutoff = 3*len(tit_X_train1)//4
tit_X_val = tit_X_train1[val_cutoff:]
tit_y_val = tit_y_train1[val_cutoff:]
tit_X_train = tit_X_train1[:val_cutoff]
tit_y_train = tit_y_train1[:val_cutoff]

tit_df = load_titanic_data(form="original df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = tit_features.columns
print(tit_cols)

# ICA

print("\n ICA \n")

ica_models = []
for n in range(1,9):
    # max iter set high because of convergence warnings
    ica = FastICA(n_components=n, max_iter=1000000)
    ica_models.append(ica)

# ICA --> NN

print("\nICA to NN\n")

scores = []
nn = MLPClassifier(batch_size=16, hidden_layer_sizes=7, alpha=0.001, learning_rate_init=0.01, shuffle=False, random_state=20)
for ica_model in ica_models:
    ica_X_train = ica_model.fit_transform(tit_X_train)
    print(ica_X_train.shape)
    nn.fit(ica_X_train, tit_y_train)

    ica_X_val = ica_model.fit_transform(tit_X_val)
    score = nn.score(ica_X_val, tit_y_val)
    print("ICA to NN val set score", score)
    scores.append(score)

plt.scatter([1,2,3,4,5,6,7,8], scores)
plt.title("NN validation set score vs number of ICA components")
plt.xlabel("number of ICA components")
plt.ylabel("Neural Net val set accuracy")
plt.savefig("ica_to_nn_scores_titanic.png")
plt.clf()

# KMEANS --> NN

print("\nKMEANS to NN\n")

kmeans_models = []
for k in range(2,9):
    model = KMeans(n_clusters=k)
    kmeans_models.append(model)

scores = []
p = 1
for kmeans_model in kmeans_models:
    kmeans_model.fit(tit_X_train)
    l = tit_X_train.shape[0]
    kmeans_X_train = np.zeros((l,p))
    labels = kmeans_model.labels_
    for j in range(len(labels)):
        label = labels[j]
        if label != 0:
            kmeans_X_train[j,label-1] = 1

    # print("number of clusters", p+1)
    # print(kmeans_X_train[:10])
    nn.fit(kmeans_X_train, tit_y_train)

    val_labels = kmeans_model.predict(tit_X_val)
    l1 = tit_X_val.shape[0]
    kmeans_X_val = np.zeros((l1,p))
    for j in range(len(val_labels)):
        label = val_labels[j]
        if label != 0:
            kmeans_X_val[j,label-1] = 1
    print(kmeans_X_val[:10])
    score = nn.score(kmeans_X_val, tit_y_val)
    print("kmeans to NN val set score", score)
    scores.append(score)
    p += 1

plt.scatter([2,3,4,5,6,7,8], scores)
plt.title("NN validation set score vs number of clusters")
plt.xlabel("number of clusters")
plt.ylabel("Neural Net val set accuracy")
plt.savefig("kmeans_to_nn_scores_titanic.png")
plt.clf()

# Using cluster size of 6 to train over entire training set, score on test set

kmm = kmeans_models[4]
kmm.fit(tit_X_train1)
l = tit_X_train1.shape[0]
kmeans_X_train1 = np.zeros((l,5))
labels = kmm.labels_
for j in range(len(labels)):
    label = labels[j]
    if label != 0:
        kmeans_X_train1[j,label-1] = 1

nn.fit(kmeans_X_train1, tit_y_train1)

test_labels = kmm.predict(tit_X_test)
l = tit_X_test.shape[0]
kmeans_X_test = np.zeros((l,5))
for j in range(len(test_labels)):
    label = test_labels[j]
    if label != 0:
        kmeans_X_test[j,label-1] = 1

score = nn.score(kmeans_X_test, tit_y_test)
print("kmean to nn test score", score)