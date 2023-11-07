import os
import random
from imutils import paths
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_test_dataframe():
    dirname = os.path.dirname(__file__)
    crr_dir = Path(dirname)
    prepared_data_path = Path(crr_dir, '../tmp/prepare_test_data/')
    prepared_data_list_filenames = (sorted(list(paths.list_images(prepared_data_path))))
    random.shuffle(prepared_data_list_filenames)
    prepared_data_list_labels = []

    for line in prepared_data_list_filenames:
        prepared_data_list_labels.append(line.split(os.path.sep)[-2])

    I_series = pd.Series(prepared_data_list_filenames, name='filenames')
    L_series = pd.Series(prepared_data_list_labels, name='labels')
    test_df = pd.concat([I_series, L_series], axis=1)

    print('-- test Datafarame --')
    print(test_df.head())
    return test_df


def create_train_dataframe():
    dirname = os.path.dirname(__file__)
    crr_dir = Path(dirname)
    prepared_data_path = Path(crr_dir, '../tmp/prepare_data/')
    prepared_data_list_filenames = (sorted(list(paths.list_images(prepared_data_path))))
    random.shuffle(prepared_data_list_filenames)
    prepared_data_list_labels = []

    for line in prepared_data_list_filenames:
        prepared_data_list_labels.append(line.split(os.path.sep)[-2])

    I_series = pd.Series(prepared_data_list_filenames, name='filenames')
    L_series = pd.Series(prepared_data_list_labels, name='labels')
    df = pd.concat([I_series, L_series], axis=1)

    print('-- train/valid Datafarame --')
    return df


def create_val_train_split(df):
    SPLIT = 0.90

    train_df, valid_df = train_test_split(df, train_size=SPLIT, shuffle=True, random_state=88)
    return train_df, valid_df
