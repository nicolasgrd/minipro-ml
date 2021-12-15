import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


def df_mean(df):
    """
    :author: Nicolas Gouraud
    :param df: dataframe
    :return: the list containing the mean of each column. If this latter contains Strings, return the most present
    element from the column
    """
    means = []
    for column in df:
        means.append(mode(df[column]))
    return means


def df_std(df):
    """
    :author: Nicolas Gouraud
    :param df: dataframe
    :return: the list containing the std of each column. If this latter contains Strings, return None
    """
    var = []
    colnames = df.columns
    for j in range(len(colnames)):
        if type(df.at[0, colnames[j]]) is not str:
            var.append(np.std(df[colnames[j]]))
        else:
            var.append(None)
    return var

def binarization(df):
    """
    author : Nicolas Gouraud
    :param df: dataframe to binarize
    :return: a binarized df, based on the user inputs
    """
    colnames = df.columns
    for j in range(len(colnames)):
        if type(df.at[0, colnames[j]]) is str:

            print("press y if you want ", df.at[0, colnames[j]], " to be associated to 1, n for 0")
            v = input()
            if v == "y":
                df[colnames[j]] = np.where(df[colnames[j]] == df.at[0, colnames[j]], 1, 0)
            else:
                df[colnames[j]] = np.where(df[colnames[j]] == df.at[0, colnames[j]], 0, 1)
    return df


def cleaning(data_file):
    """
    :author: Nicolas Gouraud, Quentin Ribaud
    :param data_file: path to data to clean
    :return: a cleaned dataframe, with NaN values replaced by average correponding column value, with outlier values removed
    """

    df = pd.read_csv(data_file)
    means = df_mean(df)
    colnames = df.columns

    # if we find NaN values in means, the corresponding columns needs to be dropped
    col_todrop = []

    for i in range(len(means)):
        if pd.isnull(means[i]):
            col_todrop.append(colnames[i])

    df = df.drop(col_todrop, axis=1)
    colnames = df.columns
    means = df_mean(df)

    # We replace missing values by an average value
    for j in range(len(colnames)):
        for i in range(len(df[colnames[j]])):
            if pd.isnull(df.at[i, colnames[j]]):
                df.at[i, colnames[j]] = means[j]

    return df


def normalizing(df, excluded_columns):
    """
    :author: Nicolas Gouraud
    :param df: dataframe to normalize
    :param excluded_columns: columns to exclude from normalisation
    :return: a centered and reduced dataframe
    """
    means = df_mean(df)
    colnames = df.columns
    std = df_std(df)
    # we exclude str data from normalization
    for j in range(len(colnames)):
        if (type(df.at[0, colnames[j]]) is not str) and (colnames[j] not in excluded_columns):
            for i in range(len(df[colnames[j]])):
                df.at[i, colnames[j]] = (df.at[i, colnames[j]] - means[j])/std[j]
    return df


def cross_validation_split(X, Y, n_sp):
    """
    :author: Nicolas Gouraud
    :param X: features, numpy array
    :param Y: labels, numpy array
    :param n_sp: number of split for cross validation
    :return: X_test, Y_test, X_train, Y_train
    """
    X_test = []
    Y_test = []
    X_train = []
    Y_train = []
    kf = KFold(n_splits=n_sp, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_test.append(X[test_index])
        Y_test.append(Y[test_index])
        X_train.append(X[train_index])
        Y_train.append(Y[train_index])

    return X_test, Y_test, X_train, Y_train


def do_PCA(n_comp, X):
    """
    :author: Nicolas Gouraud
    :param n_comp: number of component you wish to keep
    :param X: data
    :return: reduced data
    """
    pca = PCA(n_components=4)
    pca.fit(X)
    return pca.transform(X)
