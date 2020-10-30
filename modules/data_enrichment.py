import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class EnrichData(object):
    """
    Creates the train/validate/test split and encoding options.
    Args:
        csv_name (str): Name of the csv to be extracted

    Attributes:
        data_df (df): Contains raw data as passed from extraction.
        train_df (df): training df as created by the split.
        val_df (df): validation df as created by the split.

    """

    def __init__(self, data_df):
        self.data_df = data_df
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.__main_path = Path(__file__).resolve().parent.parent

    def split_data(self, target_col, save_idx_col, train_split=4):
        """
        Creates the train/validation/test splits and updates dataframes as class attributes.
        Balanced shuffle split based on number of classes in data.

        Args:
           target_col (str): name of the predicted column to balance the split.
           save_idx_col (str): name of the column to use as save index
           train_split (float): number of splits to produce for training,
                                e.g. if 4 splits, 3 will be kept for training.

        Returns
        -------
        train_df (dataframe): as class attribute the training df split
        val_df (dataframe): as class attribute the validation df split
        test_data.json: test set identifier saved under models

        """
        # Balanced class split
        main_split = StratifiedKFold(n_splits=train_split, shuffle=True)
        for train_idx, test_idx in main_split.split(self.data_df, self.data_df[target_col]):
            train = train_idx
            test_val = test_idx

        # Update the train df to be used later and create validation+test
        self.train_df = self.data_df.iloc[train, :]
        self.train_df.reset_index(inplace=True, drop=True)
        test_val_df = self.data_df.iloc[test_val, :]
        test_val_df.reset_index(inplace=True, drop=True)

        # Second split for validation / test only
        test_split = StratifiedKFold(n_splits=2, shuffle=True)
        for val_idx, test_idx in test_split.split(test_val_df, test_val_df[target_col]):
            val = val_idx
            test = test_idx

        # Update the validation/test df as class attribute to be used
        self.val_df = test_val_df.iloc[val, :]
        self.val_df.reset_index(inplace=True, drop=True)
        self.test_df = test_val_df.iloc[test, :]
        self.test_df.reset_index(inplace=True, drop=True)

        # Save the test/validation sets on target column which is the unique identifier
        val_save_path = os.path.join(self.__main_path, 'models_data',
                                     'validation_data.json')
        self.data_df.iloc[val, :][save_idx_col].to_json(val_save_path, orient='records')
        test_save_path = os.path.join(self.__main_path, 'models_data',
                                      'test_data.json')
        self.data_df.iloc[test, :][save_idx_col].to_json(test_save_path, orient='records')

    def mean_encode(self, target_col, col_encode_list, save=True):
        """
        Creates mean encoding for a list of columns.

        Args:
           target_col (str): name of the predicted column to use for mean encoding.
           col_encode_list (list): list of individual column names to mean encode.
           save (bol): whether to save the means separately as json under models.

        Returns
        -------
        train_df (dataframe): updates class attribute with encoded means
        val_df (dataframe): updates class attribute with encoded means

        """
        if self.train_df is None:
            raise ValueError('Create the train/validation/test split first!')

        for col in col_encode_list:
            current_mean_df = self.train_df.groupby([col])[target_col].mean().to_frame()
            # Reset index to make labels as column
            current_mean_df.reset_index(inplace=True)
            current_mean_df.rename(columns={'churn_status': col+'_mean'}, inplace=True)

            if save:
                temp_save_path = os.path.join(self.__main_path,
                                              'models_data',
                                              col+'_mean.json')
                current_mean_df.to_json(temp_save_path)

            # merge with the train and validation dataframes and update inplace
            self.train_df = self.train_df.merge(current_mean_df,
                                                left_on=col, right_on=col,
                                                how='left')
            self.val_df = self.val_df.merge(current_mean_df,
                                            left_on=col, right_on=col,
                                            how='left')
            self.test_df = self.test_df.merge(current_mean_df,
                                              left_on=col, right_on=col,
                                              how='left')

    def one_hot_encode(self):
        """
        Placeholder - we can also do one hot encode if wanted.
        Can use sk-learn one hot encode method but we need to save the models for later use.
        Can also implement our own method and save to json

        """
        pass

    def load_test_sets(self, target_col, col_encode_list, train=True):
        # Load the train and test set identifiers
        val_path = os.path.join(self.__main_path, 'models_data', 'validation_data.json')
        test_path = os.path.join(self.__main_path, 'models_data', 'test_data.json')
        # Transform the policy numbers in a list
        val_pol_list = pd.read_json(val_path, orient='records')[0].to_list()
        test_pol_list = pd.read_json(test_path, orient='records')[0].to_list()
        # Load encodings
        for col in col_encode_list:
            mean_path = os.path.join(self.__main_path, 'models_data',
                                     col+'_mean.json')
            current_encode = pd.read_json(mean_path)
            self.data_df = self.data_df.merge(current_encode,
                                              left_on=col, right_on=col,
                                              how='left')

        self.test_df = self.data_df[self.data_df[target_col].isin(test_pol_list)]
        self.val_df = self.data_df[self.data_df[target_col].isin(val_pol_list)]

        if train:
            self.train_df = self.data_df[~self.data_df[target_col].isin(test_pol_list +
                                                                        val_pol_list)]
