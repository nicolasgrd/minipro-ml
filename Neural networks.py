from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

from functions import *


df_kidney = cleaning('kidney_disease.csv')
df_kidney = df_kidney.drop('id', axis=1)
df_kidney = binarization(normalizing(df_kidney, []))

df_banknote = normalizing(cleaning('data_banknote_authentication.txt'), ["classification"])


Xk = df_kidney.drop(["classification"], axis=1).to_numpy()
Yk = df_kidney["classification"].to_numpy()


Xb = df_banknote.drop("classification", axis=1).to_numpy()
Yb = df_banknote["classification"].to_numpy()

def neural_network(X,Y,data_type,n):

    #if data_type =='kidney':
     #   label=LabelEncoder()
      #  X = X.apply(label.fit_transform)

    accs = np.zeros(n)
    X_test, y_test, X_train, y_train = cross_validation_split(X, Y, n)

    for i in range(n):
        clf = MLPClassifier(random_state=1, max_iter=500).fit(X_train[i], y_train[i])
        acc = clf.score(X_test[i], y_test[i])
        accs[i] = acc

    return np.mean(accs)

print("Kidney accuracy for Neural Network modelling is : ")

print(neural_network(Xk, Yk,'Kidney', 6))

print("\n")

print("Banknote accuracy for Neural Network modelling is : ")
print(neural_network(Xb, Yb,'Bank', 6))

