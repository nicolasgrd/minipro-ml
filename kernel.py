from functions import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA


df_kidney = cleaning('kidney_disease.csv')
df_kidney = df_kidney.drop('id', axis=1)
df_kidney = binarization(normalizing(df_kidney, []))

df_banknote = normalizing(cleaning('data_banknote_authentication.txt'), ["classification"])


Xk = df_kidney.drop(["classification"], axis=1).to_numpy()
Yk = df_kidney["classification"].to_numpy()


Xb = df_banknote.drop("classification", axis=1).to_numpy()
Yb = df_banknote["classification"].to_numpy()

Xk_PCA = do_PCA(4, Xk)


def compute_mean_scores_kcross_kernel(n_splits, X, Y, kernels=None):
    scores_kernels = []
    if kernels is None:
        kernels = ["linear", "poly", "rbf", "sigmoid"]

    X_test, Y_test, X_train, Y_train = cross_validation_split(X, Y, n_splits)

    # Kcross validation
    for ker in kernels:
        scores = []
        for i in range(n_splits):
            # Training
            clf = SVC(kernel=ker)
            clf.fit(X_train[i], Y_train[i])

            # Testing
            scores.append(clf.score(X_test[i], Y_test[i]))
        scores_kernels.append(np.mean(scores))

    for i in range(len(kernels)):
        print("kernel", kernels[i], "mean cross val accuracy is ", scores_kernels[i])


print("Banknote")
compute_mean_scores_kcross_kernel(6, Xb, Yb)

print("\n")
print("Kidney")
compute_mean_scores_kcross_kernel(6, Xk, Yk)

print("\n")
print("Kidney with PCA")
compute_mean_scores_kcross_kernel(6, Xk, Yk)
