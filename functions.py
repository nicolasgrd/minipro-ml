import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import KFold

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


def cleaning(data_file):
    """
    :author: Nicolas Gouraud, Quentin Ribaud
    :param data_file: path to data to clean
    :return: a cleaned dataframe, with NaN values replaced by average correponding column value, with outlier values removed
    """
    df = pd.read_csv(data_file)
    means = df_mean(df)
    colnames = df.columns

    # We replace missing values by an average value
    for j in range(len(colnames)):
        for i in range(len(df[colnames[j]])):
            #print(df.at[i, colnames[j]])
            if pd.isnull(df.at[i, colnames[j]]):
                df.at[i, colnames[j]] = means[j]

    # Now, we want te replace the outlier values of the dataframe by more convenient values, it can happen if there are miss inputs in the DF

    #df.describe()
    #df.plot(kind='box', figsize=(12, 8))
    #plt.show()
    # removing the outlier value in life_sq column
    # out_value = outlier_value_seen_with_plt.show()
    #df = df.loc[df < out_value]

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
    # We replace missing values by an average value
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


