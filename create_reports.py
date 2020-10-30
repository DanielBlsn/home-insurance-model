from modules.extract_data import ExtractData
from pandas_profiling import ProfileReport
import os
from pathlib import Path

# Create an instance of the csv and update/cleanse it
data_instance = ExtractData(csv_name='home_insurance')
data_instance.complete_update()

raw_data_df = data_instance.data_df
cleansed_df = data_instance.updated_df

# instantiate the current file path and the reports save path
current_path = Path(__file__).resolve().parent
save_path = os.path.join(current_path, 'reports')

# define columns to select for localized report
col_sel_list = ['claim3years', 'ncd_granted_years_b', 'ncd_granted_years_c', 'spec_sum_insured',
                'contents_cover', 'buildings_cover', 'spec_item_prem', 'sec_disc_req',
                'bedrooms', 'sum_insured_contents', 'last_ann_prem_gross', 'churn_status']

if __name__ == '__main__':

    # full_profile = ProfileReport(raw_data_df,
    #                              title='Pandas Profiling Report', explorative=True)
    # full_profile.to_file(os.path.join(save_path, 'complete_report.html'))
    partial_profile = ProfileReport(cleansed_df.loc[:, col_sel_list],
                                    title='Pandas Profiling Report', explorative=True)
    partial_profile.to_file(os.path.join(save_path, 'filtered_report.html'))
