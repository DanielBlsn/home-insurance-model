import pandas as pd
import numpy as np
from pathlib import Path
import os


class ExtractData(object):
    """Raw and cleansed data extract class.
    Args:
        csv_name (str): Name of the csv to be extracted

    Attributes:
        data_df (df): Contains raw data as found in the csv.
        updated_df (df): Contains data cleansed and updated with correct features.

    """

    def __init__(self, csv_name):
        self.csv_name = csv_name
        self.data_df = None
        self.updated_df = None
        self.__main_path = Path(__file__).resolve().parent.parent
        # Instantiate extraction of the raw data df
        self.__get_data()

    def __get_data(self):
        """
        Loads the main data csv as class attribute.

        Returns
        -------
        data_df(self, pandas dataframe): df contains all the raw data from csv.
        updated_df(self, pandas dataframe): data_df duplicate, used to cleanse later.

        """
        csv_path = os.path.join(self.__main_path,
                                'data', self.csv_name+'.csv')
        self.data_df = pd.read_csv(csv_path)
        self.updated_df = self.data_df

    def binary_to_numerical(self, col_list=None, bin_col_list=['Y', 'N']):
        """
        Updates updated_df with binary cols Y/N transformed to 1/0.
        Args:
            col_list (list): list of columns to binarize
            bin_col_list(list): list of values to transform to [1, 0]

        Returns
        -------
        None.

        """
        if col_list is None:
            col_list = ['claim3years', 'bus_use', 'ad_buildings', 'ad_contents', 'contents_cover',
                        'buildings_cover', 'p1_policy_refused', 'appr_alarm', 'appr_locks',
                        'flooding', 'neigh_watch', 'safe_installed', 'sec_disc_req', 'subsidence',
                        'legal_addon_pre_ren', 'legal_addon_post_ren', 'home_em_addon_pre_ren',
                        'home_em_addon_post_ren', 'garden_addon_pre_ren', 'garden_addon_post_ren',
                        'keycare_addon_pre_ren', 'keycare_addon_post_ren', 'hp1_addon_pre_ren',
                        'hp1_addon_post_ren', 'hp2_addon_pre_ren', 'hp2_addon_post_ren',
                        'hp3_addon_pre_ren', 'hp3_addon_post_ren', 'mta_flag']
        self.updated_df[col_list] = self.updated_df[col_list].replace(bin_col_list,
                                                                      [1, 0])

    def remove_columns(self, col_list=None):
        """
        Updates updated_df to remove unneeded columns.
        Args:
            col_list (list): list of columns to remove

        Returns
        -------
        None.

        """
        if col_list is None:
            col_list = ['p1_pt_emp_status', 'i', 'campaign_desc']
        self.updated_df.drop(columns=col_list, inplace=True)

    def complete_update(self):
        """
        Take the raw dataframe through all the optimization steps.

        Returns
        -------
        self.updated_df(self, pandas df): updates the class attribute to cleansed format.

        """
        self.updated_df.columns = self.updated_df.columns.str.lower()
        # Remove rows where cover start data is missing
        # All other features are missing when cover start is blank
        self.updated_df.dropna(subset=['cover_start'], inplace=True)
        # Last annual premium cannot be negative? - remove the outliers
        self.updated_df = self.updated_df[self.updated_df['last_ann_prem_gross'] > 0]
        self.updated_df = self.updated_df[self.updated_df['last_ann_prem_gross'] < 2000]
        # Select only lasped/live rows for this analysis
        self.updated_df = self.updated_df[self.updated_df['pol_status'].isin(['Lapsed', 'Live'])]
        self.updated_df.rename(columns={'pol_status': 'churn_status',
                                        'police': 'policy_nr'}, inplace=True)
        # Other filtering
        self.updated_df = self.updated_df[self.updated_df['p1_sex'] != 'N']
        # Calculate person age
        self.updated_df['p1_dob'] = self.updated_df['p1_dob'].apply(pd.to_datetime)
        self.updated_df['p_age'] = 2012 - self.updated_df['p1_dob'].dt.year
        # Evaluate employment & maritial status
        self.updated_df['p_emp_status_upd'] = np.where(self.updated_df['p1_emp_status'] == 'R', 'Retired',
                                              np.where(self.updated_df['p1_emp_status'] == 'E', 'Employed', 'Other'))
        self.updated_df['p_mar_status_upd'] = np.where(self.updated_df['p1_mar_status'] == 'M', 'Married',
                                              np.where(self.updated_df['p1_mar_status'] == 'S', 'Single', 
                                              np.where(self.updated_df['p1_mar_status'] == 'P', 'Partnership',
                                              np.where(self.updated_df['p1_mar_status'] == 'W', 'Widowed', 'Other'))))
        # Always reset index after filtering
        self.updated_df.reset_index(inplace=True, drop=True)

        # Apply further cleansing functions
        self.binary_to_numerical()
        self.binary_to_numerical(col_list=['p1_sex'],
                                 bin_col_list=['M', 'F'])
        self.binary_to_numerical(col_list=['churn_status'],
                                 bin_col_list=['Lapsed', 'Live'])
        self.remove_columns()
