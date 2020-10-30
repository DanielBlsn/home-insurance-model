from modules import ml_models
import os
from pathlib import Path
import json
from sklearn import metrics
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt


class BuildResults():
    """
    Evaluate results over the given dataset.

    Args:
        data_df (dataframe): dataframe used for predictions
        target_feature (str): col name to be used as predicted variable
        clf_features (list): col names of predictor variables
        clf_agg_features (list): col names used at the aggregation layer

    Attributes:
        results (dict): dict containing results

    """

    def __init__(self, data_df,
                 target_feature, clf_features,
                 clf_agg_features):
        self.data_df = data_df
        self.target_feature = target_feature
        self.clf_features = clf_features
        self.clf_agg_features = clf_agg_features
        self.results = {}
        self.__main_path = Path(__file__).resolve().parent
        self.__save_path = os.path.join(self.__main_path, 'models_data')
        self.__report_path = os.path.join(self.__main_path, 'reports')

    def predict_first_layer(self):
        """
        Creates the predictions and class probabilities for xgb and rf models.

        Returns
        -------
        data_df (dataframe): updates class attribute with predictions

        """
        # Xgb prediction
        xgb_prediction = ml_models.general_predictor(data_df=self.data_df,
                                                     features=self.clf_features,
                                                     model_name='xgb_classifier.pickle.dat')
        # Add column with assign to surpress warning
        # Second item from returned list are probabilities and column 0 are for class 0
        self.data_df = self.data_df.assign(xgb_pred=xgb_prediction[0])
        self.data_df = self.data_df.assign(xgb_prob_0=xgb_prediction[1][:, 0])
        self.data_df = self.data_df.assign(xgb_prob_1=xgb_prediction[1][:, 1])

        # Rf prediction
        rf_prediction = ml_models.general_predictor(data_df=self.data_df,
                                                    features=self.clf_features,
                                                    model_type='rfc',
                                                    model_name='rf_classifier.pickle.dat',
                                                    imputation=True)
        self.data_df = self.data_df.assign(rf_pred=rf_prediction[0])
        self.data_df = self.data_df.assign(rf_prob_0=rf_prediction[1][:, 0])
        self.data_df = self.data_df.assign(rf_prob_1=rf_prediction[1][:, 1])

    def predict_agg_layer(self):
        """
        Creates the predictions and class probabilities for the aggregation neural network model.

        Returns
        -------
        data_df (dataframe): updates class attribute with predictions

        """
        nn_prediction = ml_models.nn_predict(data_df=self.data_df,
                                             features=self.clf_agg_features,
                                             model_name='neural_network.pth')
        # Neural network will return probabilities - round them to get the class prediction
        self.data_df = self.data_df.assign(nn_prob=nn_prediction)
        self.data_df['nn_pred'] = self.data_df['nn_prob'].round()

    def evaluate_results(self, save_name, save=True, save_data=True):
        """
        Evaluate the results by creating confusion matrix, and accuracy measures.

        Returns
        -------
        data_df (dataframe): updates class attribute with predictions

        """
        # Add predictions to main df
        self.predict_first_layer()
        self.predict_agg_layer()

        # Define the arrays to use in predictions
        y_true = self.data_df[self.target_feature].values
        y_pred_xgb = self.data_df['xgb_pred'].values
        y_pred_rf = self.data_df['rf_pred'].values
        y_pred_nn = self.data_df['nn_pred'].values

        # Draw the confusion matrix
        cm_nn = metrics.confusion_matrix(y_true, y_pred_nn)
        cm_hm = sns.heatmap(cm_nn, annot=True, fmt='d')
        plt.title('Neural Network \nAccuracy:{0:.3f}'.format(metrics.accuracy_score(y_true, y_pred_nn)))
        plt.ylabel('Churn Status')
        plt.xlabel('Predicted Status')
        cm_fig = cm_hm.get_figure()
        cm_fig.savefig(os.path.join(self.__report_path,
                                    save_name+'_confustion_matrix.png'))
        plt.show()

        # Draw the PROC - class 0 and class 1 probabilities required
        self.data_df['nn_prob0'] = 1-self.data_df['nn_prob'].values
        skplt.metrics.plot_precision_recall(y_true, self.data_df[['nn_prob0', 'nn_prob']].values,
                                            plot_micro=False)
        plt.show()

        # Lists for ease of looping through results for all models
        pred_list = ['xgb', 'rf', 'nn']
        pred_list_array = [y_pred_xgb, y_pred_rf, y_pred_nn]

        for i in range(0, len(pred_list)):
            self.results[pred_list[i]] = {'recall': metrics.recall_score(y_true,
                                                                         pred_list_array[i]),
                                          'precision': metrics.precision_score(y_true,
                                                                               pred_list_array[i]),
                                          'f1_score': metrics.f1_score(y_true,
                                                                       pred_list_array[i])}

        if save:
            results_path = os.path.join(self.__save_path, save_name+'_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=4)
        if save_data:
            excel_path = os.path.join(self.__report_path,
                                      save_name+'_predictions_data.xlsx')
            self.data_df.to_excel(excel_path, index=False)
