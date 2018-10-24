from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np
# from support import k_means_2d, k_means_per_feature
# from support import em_2d, em_per_feature
# from support import k_means_one_feature, em_one_feature

# IMPORT DATA ETC

#ten_X_train, ten_y_train, ten_X_test, ten_y_test = load_tennis_data()
tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()
#ten_df = load_tennis_data(form="original df")
tit_df = load_titanic_data(form="original df")
#ten_features, ten_labels = load_tennis_data(form="df")
tit_features, tit_labels = load_titanic_data(form="df")

# PCA

pca = PCA()
pca.fit(tit_X_train)
pca.transform(tit_X_train)
sing_vals = pca.singular_values_
plt.scatter(range(len(sing_vals)), sing_vals)
plt.title("PCA singular values - Titanic")
plt.ylabel("singular values")
plt.xlabel("component")
plt.savefig("pca_titanic.png")

# ICA

ica = FastICA()
ica.fit(tit_X_train)
ica.transform(tit_X_train)
print(ica.mixing_)


