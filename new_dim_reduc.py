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

# IMPORT DATA

tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()

# Split off validation set from training set
val_cutoff = 3*len(tit_X_train)//4
tit_X_val = tit_X_train[val_cutoff:]
tit_y_val = tit_y_train[val_cutoff:]
tit_X_train = tit_X_train[:val_cutoff]
tit_y_train = tit_y_train[:val_cutoff]

tit_df = load_titanic_data(form="original df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = tit_features.columns
print(tit_cols)

# Feature transforms

colors = 'bgrcmyk'
#print("\n PCA \n")

# Dimensionality Reduce using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(tit_X_train)

# Dimensionality Reduce using GRP
rca = GaussianRandomProjection(n_components=2)
data_rca = rca.fit_transform(tit_X_train)

# Dimensionlity Reduce using RFE
logreg = LogisticRegression(solver='sag')
rfe = RFE(logreg, n_features_to_select=2)
data_rfe = rfe.fit_transform(tit_X_train, tit_y_train)

# Dimensionality Reduce using ICA
ica = FastICA(n_components=2, max_iter=1000000)
data_ica = ica.fit_transform(tit_X_train)

data = [data_pca, data_rca, data_rfe, data_ica]
methods = ["PCA", "RCA", "RFE", "ICA"]

# Iterate from 2 up to 7 clusters
for k in range(2, 8):
    # Create the KMeans clustering object for # of clusters k
    kmm = KMeans(n_clusters=k)

    # For each version of dimensionality reduced data...
    for (d, m) in zip(data, methods):

        # perform clustering
        kmm.fit(d)

        # get labels
        labels = kmm.predict(d)
        #print(labels)
        #print(type(labels))

        # plot data
        plt.figure()

        # cycle through each label and plot with different color
        for j in range(k):
            d1 = d[labels == j]
            plt.scatter(d1[:,0], d1[:,1], c=colors[j-2])
        plt.title('Method = %s, Number of clusters = %d' % (m, k))

        # Save the plot once we cycle through all clusters
        plt.savefig("%s_titanic_ndims2_nclusters%d.png" % (m, k))


