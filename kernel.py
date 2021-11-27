from functions import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict


df_kidney = normalizing(cleaning('kidney_disease.csv'), ["id"])
df_banknote = normalizing(cleaning('data_banknote_authentication.txt'), ["classification"])

Xk = df_kidney.drop(["classification", "id"], axis=1).to_numpy()
Yk = df_kidney["classification"].to_numpy()
Xb = df_banknote.drop("classification", axis=1).to_numpy()
Yb = df_banknote["classification"].to_numpy()

n_splits = 2
Xk_test, Yk_test, Xk_train, Yk_train = cross_validation_split(Xk, Yk, n_splits)
Xb_test, Yb_test, Xb_train, Yb_train = cross_validation_split(Xb, Yb, n_splits)


## KIDNEY
scores = []

rbf_feature = RBFSampler()

# Kcross validation
for i in range(n_splits):
    Xtraining_fold, Xtesting_fold, Ytraining_fold, Ytesting_fold = Xb_train[i], Xb_test[i], Yb_train[i], Yb_test[i]

    X_features_train = rbf_feature.fit_transform(Xtraining_fold)
    X_features_test = rbf_feature.fit_transform(Xtesting_fold)

    # Training
    clf = SGDClassifier(max_iter=50)
    clf.fit(X_features_train, Ytraining_fold)

    # Testing
    scores.append(clf.score(X_features_test, Ytesting_fold))
print(scores)