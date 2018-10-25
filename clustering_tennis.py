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

ten_X_train, ten_y_train, ten_X_test, ten_y_test = load_tennis_data()
#tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()
ten_df = load_tennis_data(form="original df")
#tit_df = load_titanic_data(form="original df")
ten_features, ten_labels = load_tennis_data(form="df")
#tit_features, tit_labels = load_titanic_data(form="df")

ten_cols = list(ten_features.columns)
print(ten_cols)
ten_l = len(ten_X_train)

# K MEANS
# ks = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
ks = [5,10,15,20,26]
# ks = [26]
wss = []
wss_em = []
# xs = [1]
# ys = [6]
for k in ks:
    # if False:
    #     k_means_2d(ten_X_train, k, ten_cols, xs, ys)
    # if False:
    #     k_means_per_feature(ten_X_train, k, ten_cols)
    # if False:
    #     em_2d(ten_X_train, k, ten_cols, xs, ys)
    # if False:
    #     em_per_feature(ten_X_train, k, ten_cols)
    # if False:
    #     k_means_one_feature(ten_X_train, k, ten_cols, 1)
    # if False:
    #     k_means_2d(ten_X_train, k, ten_cols, xs, ys, clusters=[0,1,9,22,23])
    #     em_2d(ten_X_train, k, ten_cols, xs, ys)
    if True:
        model = KMeans(n_clusters=k)
        model.fit(ten_X_train)
        wss.append(model.inertia_)
    if True:
        em = GaussianMixture(n_components=k)
        em.fit(ten_X_train)
        labels = np.array(em.predict(ten_X_train))
        centers = np.array(em.means_)
        sum = 0
        for idx in range(len(labels)):
            label = labels[idx]
            center = centers[label]
            point = ten_X_train[idx]
            sum += np.sum((point-center)**2)
        wss_em.append(sum)

if True:
    plt.scatter(ks, wss)
    plt.title("k means elbow curve for Tennis")
    plt.xlabel("number of clusters")
    plt.ylabel("wss")
    plt.show()

if True:
    plt.scatter(ks, wss_em)
    plt.title("EM elbow curve for Tennis")
    plt.xlabel("number of clusters")
    plt.ylabel("wss")
    plt.show()
