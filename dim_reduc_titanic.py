from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np
# from support import k_means_2d, k_means_per_feature
# from support import em_2d, em_per_feature
# from support import k_means_one_feature, em_one_feature

# IMPORT DATA

#ten_X_train, ten_y_train, ten_X_test, ten_y_test = load_tennis_data()
tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()
#ten_df = load_tennis_data(form="original df")
tit_df = load_titanic_data(form="original df")
#ten_features, ten_labels = load_tennis_data(form="df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = tit_features.columns
print(tit_cols)

# PCA

print("\n PCA \n")
pca = PCA()
pca.fit(tit_X_train)
pca.transform(tit_X_train)
sing_vals = pca.singular_values_
plt.scatter(range(len(sing_vals)), sing_vals)
plt.title("PCA singular values - Titanic")
plt.ylabel("singular values")
plt.xlabel("component")
plt.savefig("pca_titanic.png")

pca_matrix = np.array(pca.components_)
print(pca_matrix.shape)

# ICA

print("\n ICA \n")

ica_models = []
for n in range(1,9):
    #print("ICA with " + str(n) + " components")
    ica = FastICA(n_components=n)
    ica.fit(tit_X_train)
    ica.transform(tit_X_train)
    ica_models.append(ica)
for i in range(8):
    ica_matrix = np.array(ica_models[i].mixing_)
    print(ica_matrix.shape)

# RCA

print("\n RCA \n")

rca_models = []
for n in range(1,9):
    #print("RCA with " + str(n) + " components")
    rca = GaussianRandomProjection(n_components=n)
    rca.fit(tit_X_train)
    rca_models.append(rca)
for i in range(8):
    rca_matrix = np.array(rca_models[i].components_)
    print(rca_matrix.shape)

# RFE

logreg = LogisticRegression(solver='sag')

rfe_models = []
for n in range(1,8):
    rfe = RFE(logreg, n_features_to_select=n)
    rfe.fit(tit_X_train, tit_y_train)
    rfe_models.append(rfe)
for i,model in zip(range(len(rfe_models)),rfe_models):
    ranking = np.array(model.ranking_)
    order = np.argsort(ranking)
    print(tit_cols[order][:i+1])



