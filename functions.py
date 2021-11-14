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
    :author: Nicolas Gouraud
    :param data_file: path to data to clean
    :return: a cleaned dataframe, with NaN values replaced by average correponding column value
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
    return df

df = cleaning('kidney_disease.csv')
print(df)