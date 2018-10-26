from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
plt.clf()

pca_matrix = np.array(pca.components_)
print(pca_matrix.shape)

# ICA

print("\n ICA \n")

ica_models = []
for n in range(1,9):
    # max iter set high because of convergence warnings
    ica = FastICA(n_components=n, max_iter=1000000)
    ica_models.append(ica)

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


# ICA --> Neural Net
# Now in nn.py

# scores = []
# nn = MLPClassifier(batch_size=16, hidden_layer_sizes=7, alpha=0.001, learning_rate_init=0.01, shuffle=False, random_state=20)
# for ica_model in ica_models:
#     ica_X_train = ica_model.fit_transform(tit_X_train)
#     print(ica_X_train.shape)
#     nn.fit(ica_X_train, tit_y_train)

#     ica_X_val = ica_model.fit_transform(tit_X_val)
#     score = nn.score(ica_X_val, tit_y_val)
#     print("val set score", score)
#     scores.append(score)

# plt.scatter([1,2,3,4,5,6,7,8], scores)
# plt.title("NN validation set score vs number of ICA components")
# plt.xlabel("number of ICA components")
# plt.ylabel("Neural Net val set accuracy")
# plt.savefig("ica_to_nn_scores_titanic.png")
# plt.clf()













