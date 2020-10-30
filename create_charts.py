from modules.extract_data import ExtractData
import os
from pathlib import Path
from scipy import stats
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt


# Create an instance of the csv and update/cleanse it
data_instance = ExtractData(csv_name='home_insurance')
data_instance.complete_update()
# Extract raw data
raw_data_df = data_instance.data_df
cleansed_df = data_instance.updated_df
# Instantiate the current file path and the reports save path
current_path = Path(__file__).resolve().parent
save_path = os.path.join(current_path, 'reports')


# Visualize class distribution
values = cleansed_df.groupby(['churn_status'])['churn_status'].count()
plt.pie(values,
        labels=['Live', 'Lapsed'],
        colors=['royalblue', 'silver'],
        explode=(0, 0.1),  # Separate pie parts
        autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(values)/100),  # % within pie chart
        shadow=True,
        textprops={'fontsize': 10})  # Text size
plt.title('Total Customer Churn Status')
plt.savefig(os.path.join(save_path,
                         'churn_class_distribution.png'))
plt.show()


# Visualize  the churn to premium distribution
prem_churn_plot = sns.violinplot(cleansed_df['churn_status'],
                                 cleansed_df['last_ann_prem_gross'],
                                 palette="muted")
plt.title('Premium Paid Distribution in Relation to Churn Status')
prem_churn_plot.set(xlabel='Churn Status', ylabel='Premium')
prem_fig = prem_churn_plot.get_figure()
prem_fig.savefig(os.path.join(save_path,
                              'last_ann_prem_gross churn_status.png'))


# Visualize sex differences
plt.pie(cleansed_df.groupby(['p1_sex'])['p1_sex'].count(),
        labels=['Female', 'Male'],
        colors=['lightcoral', 'dimgray'],
        explode=(0.1, 0.1),  # Separate pie parts
        autopct='%1.1f%%',  # % within pie chart
        pctdistance=0.8,  # distance from center
        startangle=90, frame=True,
        textprops={'fontsize': 13},  # Text size
        radius=4)
# Subplot for gender in relation to churn
plt.pie(cleansed_df.groupby(['p1_sex', 'churn_status'])['churn_status'].count(),
        colors=['royalblue', 'silver'],
        explode=(0.1, 0.1, 0.1, 0.1),  # Separate pie parts
        autopct='%1.1f%%',  # % within pie chart
        pctdistance=0.7,  # distance from center
        startangle=90, frame=True,
        textprops={'fontsize': 13},  # Text size
        radius=2.2)
# Draw circle
centre_circle = plt.Circle((0, 0), 1, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.title('Customer Gender Differences')
plt.savefig(os.path.join(save_path,
                         'gender_churn_distribution.png'))
plt.show()


# Visualize Gender - Churn - Premium relationship
prem_gen_ax = sns.boxplot(x='p1_sex',
                          y='last_ann_prem_gross',
                          hue='churn_status',
                          data=cleansed_df)
prem_gen_ax.set(xlabel='Gender', ylabel='Premium')
plt.title('Gender Differences Compared to Premium Paid')
prem_gen_fig = prem_gen_ax.get_figure()
prem_gen_fig.savefig(os.path.join(save_path,
                                  'gender_churn_prem_rel.png'))

# Other checks
count_emp = sns.countplot(x='churn_status',
                          hue='p_emp_status_upd',
                          data=cleansed_df)
plt.show()

count_mar = sns.countplot(x='churn_status',
                          hue='p_emp_status_upd',
                          data=cleansed_df)
plt.show()

count_pay = sns.countplot(x='churn_status',
                          hue='payment_method',
                          data=cleansed_df)
count_pay.set(xlabel='Payment Method', ylabel='Churn Status')
plt.title('Churn Status in Relation to Payment Method')
pay_fig = count_pay.get_figure()
pay_fig.savefig(os.path.join(save_path,
                             'pay_met_churn_.png'))
plt.show()


def evaluate_correlations(data_df, x_list, y, correlation_type):
    """
    Evaluate the correlation between multiple variables within a df.
    Depending on the type of correlation it will use:
    - Jaccard for binary to binary data
    - Pearson for continuous data
    - Point Bisection for binary to continuous

    Parameters
    ----------
    x_list (list): list of column names to evaluate correlation against
    y (str): target variable to calculate correlation against
    correlation_type (str): binary, continuous, mixed

    Returns
    -------
    Correlation values list.

    """
    cor_type_dict = {'binary': matthews_corrcoef,  # phi coefficient, = pearson for binary
                     'continuous': stats.pearsonr,
                     'binary-mixed': stats.pointbiserialr}
    if correlation_type not in cor_type_dict:
        raise ValueError('Correlation type incorrect.')

    cor_list = []
    # Apply the current correlation to each value in x
    for col in x_list:

        cor = cor_type_dict[correlation_type](data_df[col].values,
                                              data_df[y].values)
        # Create the dict of values, except when binary since it's a distance with no p val
        temp_dict = {'x': col,
                     'y': y,
                     'cor': cor[0] if correlation_type != 'binary' else cor,
                     'pvalue': cor[1] if correlation_type != 'binary' else 'na'}
        cor_list.append(temp_dict)

    return cor_list


mixed_cols_list = ['ncd_granted_years_b', 'ncd_granted_years_c', 'spec_sum_insured',
                   'spec_item_prem', 'sec_disc_req', 'bedrooms', 'sum_insured_contents',
                   'last_ann_prem_gross']
mixed_cor = evaluate_correlations(data_df=cleansed_df,
                                  x_list=mixed_cols_list,
                                  y='churn_status',
                                  correlation_type='binary-mixed')

binary_cols_list = ['p1_sex', 'claim3years']
binary_cor = evaluate_correlations(data_df=cleansed_df,
                                   x_list=binary_cols_list,
                                   y='churn_status',
                                   correlation_type='binary')
