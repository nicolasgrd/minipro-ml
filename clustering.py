from functions import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

### Victor Bercy ###

def test_eps(X, Y, N_min):
    """
    Allows the user to compare the F1-score w.r.t. the maximum distance eps of the DBSCAN clustering for the given dataset X
    :param X: training dataset
    :param Y: labels of the training dataset
    :param N_min: minimal number of samples to define a cluster
    :return: eps
    """
    E, Acc = [], []

    for k in np.linspace(2, 2.2, 100):
        db = DBSCAN(eps = k, min_samples=N_min).fit(X)
        Y_pred = db.labels_
        E.append(k)
        Acc.append(accuracy_score(Y, Y_pred))

    return E, Acc


# Chronic kidney disease dataset
df_kidney = cleaning('kidney_disease.csv')
df_kidney = df_kidney.drop('id', axis=1)
df_kidney = binarization(normalizing(df_kidney, []))

Xk = df_kidney.drop(["classification"], axis=1).to_numpy() # data
Yk = df_kidney["classification"].to_numpy() # labels

N_min_k = min(np.count_nonzero(Yk==1), np.count_nonzero(Yk==0)) - 10 # Minimal number of items in a class minus a random chosen number
E_k, Acc_k = test_eps(Xk, Yk, N_min_k)
eps_k = E_k[Acc_k.index(max(Acc_k))]
db_k = DBSCAN(eps=eps_k, min_samples=N_min_k).fit(Xk)
print(db_k.labels_)
print(f"F1 score of DBSCAN on the Banknote authentication dataset : {accuracy_score(Yk, db_k.labels_)}")

# Banknote authentication dataset
df_banknote = normalizing(cleaning('data_banknote_authentication.txt'), ["classification"])

Xb = df_banknote.drop("classification", axis=1).to_numpy() # data
Yb = df_banknote["classification"].to_numpy() # labels

N_min_b = min(np.count_nonzero(Yb==1), np.count_nonzero(Yb==0)) - 10 # Minimal number of items in a class minus a random chosen number
E_b, Acc_b = test_eps(Xb, Yb, N_min_b)
eps_b = E_b[Acc_b.index(max(Acc_b))]
db_b = DBSCAN(eps=eps_b, min_samples=N_min_b).fit(Xb)
print(db_b.labels_)
print(f"F1 score of DBSCAN on the Banknote authentication dataset : {accuracy_score(Yb, db_b.labels_)}")

# Plot (doesn't work on my laptop, see error message below)
# UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.    plt.show()
plt.subplot(211)
plt.plot(E_k, Acc_k, label="F1 score")
plt.title("Chronic kidney disease")
plt.subplot(212)
plt.plot(E_b, Acc_b, label="F1 score")
plt.title("Banknote authentication")
plt.show()
