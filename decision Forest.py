
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from functions import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# Chronic kidney disease dataset
df_kidney = cleaning('kidney_disease.csv')
df_kidney = df_kidney.drop('id', axis=1)
df_kidney = binarization(normalizing(df_kidney, []))

Xk = df_kidney.drop(["classification"], axis=1).to_numpy()
Yk = df_kidney["classification"].to_numpy()

# Banknote authentication dataset
df_banknote = normalizing(cleaning('data_banknote_authentication.txt'), ["classification"])

Xb = df_banknote.drop("classification", axis=1).to_numpy()
Yb = df_banknote["classification"].to_numpy()


def random_forest(X,Y,n):
    'author : Quentin RIBAUD'
    X_test, y_test, X_train, y_train = cross_validation_split(X, Y, n)

    op_max_depth = 14
    accs = np.zeros(n)

    for i in range(n):
        rf = RandomForestClassifier(n_estimators=150, max_depth=op_max_depth)
        rf.fit(X_train[i], y_train[i])
        acc = rf.score(X_test[i], y_test[i])
        accs[i] = acc

    return np.mean(accs)

print("Banknote accuracy for Decision Forest modelling is : ")
print(random_forest(Xb,Yb,6))

print("\n")

print("Kidney accuracy for Decision Forest modelling is : ")
print(random_forest(Xk,Yk,6))



#print("\n")
#print("Kidney accuracy for Decision Forest modelling  with PCA")
#print(random_forest(6, Xk_PCA, Yk))