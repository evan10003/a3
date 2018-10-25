from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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

tit_cols = list(tit_features.columns)
print(tit_cols)
tit_l = len(tit_X_train)

# K MEANS
#ks = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
#ks = [2,4,6,8,10,12,14,16]
ks = [4,8,12,16]
wss = []
wss_em = []
# xs = [1]
# ys = [4]
for k in ks:
    # if False:
    #     k_means_2d(tit_X_train, k, tit_cols, xs, ys)
    # if False:
    #     k_means_per_feature(tit_X_train, k, tit_cols)
    # if False:
    #     em_2d(tit_X_train, k, tit_cols, xs, ys)
    # if False:
    #     em_per_feature(tit_X_train, k, tit_cols)
    # if False:
    #     k_means_one_feature(tit_X_train, k, tit_cols, 1)
    # if False:
    #     em_one_feature(tit_X_train, k, tit_cols, 1)
    if True:
        model = KMeans(n_clusters=k)
        model.fit(tit_X_train)
        wss.append(model.inertia_)
    if True:
        em = GaussianMixture(n_components=k)
        em.fit(tit_X_train)
        labels = np.array(em.predict(tit_X_train))
        centers = np.array(em.means_)
        sum = 0
        for idx in range(len(labels)):
            label = labels[idx]
            center = centers[label]
            point = tit_X_train[idx]
            sum += np.sum((point-center)**2)
        wss_em.append(sum)

if True:
    plt.scatter(ks, wss)
    plt.title("k means elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("wss")
    plt.show()

if True:
    plt.scatter(ks, wss_em)
    plt.title("EM elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("wss")
    plt.show()
