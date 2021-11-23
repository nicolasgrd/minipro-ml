import pandas as pd
import numpy as np
from statistics import mode


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
            print(df.at[i, colnames[j]])
            if pd.isnull(df.at[i, colnames[j]]):
                df.at[i, colnames[j]] = means[j]

    # Now, we want te replace the outlier values of the dataframe by more convenient values, it can happen if there are miss inputs in the DF

    df.describe()
    df.plot(kind='box', figsize=(12, 8))
    plt.show()
    # removing the outlier value in life_sq column
    # out_value = outlier_value_seen_with_plt.show()
    df = df.loc[df < out_value]

    return df


df = cleaning('kidney_disease.csv')
print(df)

