from modules.extract_data import ExtractData
from modules.data_enrichment import EnrichData
from build_classifier import BuildClf
from build_aggregator import BuildAggregator
from build_results import BuildResults

# Create an instance of the csv and update/cleanse it
data_instance = ExtractData(csv_name='home_insurance')
data_instance.complete_update()
# Extract raw data
raw_data_df = data_instance.data_df
cleansed_df = data_instance.updated_df

if __name__ == '__main__':
    # Create train/validation/test split and encoding
    data_enrich = EnrichData(cleansed_df)
    data_enrich.split_data(target_col='churn_status', save_idx_col='policy_nr')
    data_enrich.mean_encode(target_col='churn_status',
                            col_encode_list=['p1_emp_status', 'p1_mar_status',
                                             'payment_method'])
    train_df = data_enrich.train_df
    validation_df = data_enrich.val_df
    test_df = data_enrich.test_df

    # Define the classifier features used in training
    clf_features = ['claim3years', 'ncd_granted_years_b', 'ncd_granted_years_c',
                    'bedrooms', 'yearbuilt', 'risk_rated_area_c', 'appr_alarm', 'appr_locks',
                    'neigh_watch', 'sum_insured_contents', 'spec_sum_insured', 'ad_buildings',
                    'p_age', 'p1_sex', 'last_ann_prem_gross', 'p1_emp_status_mean',
                    'p1_mar_status_mean', 'payment_method_mean']

    # Build the xgb and rf classification layers
    xgb_clf = BuildClf(model_type='xgb_clf',
                       train_df=train_df,
                       validation_df=validation_df,
                       target_feature='churn_status',
                       clf_features=clf_features)
    xgb_clf.train_xgb_model()

    rf_clf = BuildClf(model_type='random_forest',
                      train_df=train_df,
                      validation_df=validation_df,
                      target_feature='churn_status',
                      clf_features=clf_features)
    rf_clf.train_rf_model()

    # Build the neural network layer
    clf_agg_features = ['xgb_prob_0', 'xgb_prob_1',
                        'rf_prob_0', 'rf_prob_1']

    nn_agg_clf = BuildAggregator(data_df=train_df,
                                 target_feature='churn_status',
                                 clf_features=clf_features,
                                 clf_agg_features=clf_agg_features)
    nn_agg_clf.train_aggregator()

    # Evaluate results
    test_res = BuildResults(data_df=test_df,
                            target_feature='churn_status',
                            clf_features=clf_features,
                            clf_agg_features=clf_agg_features)
    test_res.evaluate_results(save_name='test')

    train_res = BuildResults(data_df=train_df,
                             target_feature='churn_status',
                             clf_features=clf_features,
                             clf_agg_features=clf_agg_features)
    train_res.evaluate_results(save_name='train')
