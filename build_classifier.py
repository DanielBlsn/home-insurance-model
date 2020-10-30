from modules.xgboost_optimization import XGBoostOptimization
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


class BuildClf(object):
    """
    Creates an instance of the classifier used to optimize and predict.

    Args:
        model_type (str): random_forest or xgb_clf
        train_df (dataframe)
        validation_df (dataframe)
        target_feature (str): col name to be used as predicted variable
        clf_features (list): col names of predictor variables

    Attributes:
        clf (model): sk-learn/xgb model that can be used as predictors
        val_score (float): score of the current classifier model
        self.y_pred_val (array): array of predictions

    """

    def __init__(self, model_type,
                 train_df, validation_df,
                 target_feature, clf_features):
        self.model_type = model_type
        self.train_df = train_df
        self.validation_df = validation_df
        self.target_feature = target_feature
        self.clf_features = clf_features
        self.clf = None
        self.val_score = None
        self.y_pred_val = None
        self.__main_path = Path(__file__).resolve().parent
        self.__save_path = os.path.join(self.__main_path, 'models_data')
        # Instantiate the x/y features as arrays
        self.__define_features()

    def __model_instance(self):
        """
        Instantiates model type as defined within the class instance.

        Args:
           None.

        Returns
        -------
        model: instance of sk-learn random forest or xgb classifier

        """
        potential_models = {'random_forest': RandomForestClassifier(),
                            'xgb_clf': xgb.XGBClassifier()}
        if self.model_type not in potential_models:
            raise ValueError('Current model not supported!')

        return potential_models[self.model_type]

    def __define_features(self):
        """
        Transforms dataframes to numpy arrays to be used in models.

        Args:
           None.

        Returns
        -------
        X_train, X_val, y_train, y_val: updates training data as np arrays within class attributes

        """
        self.X_train = self.train_df[self.clf_features].values
        self.X_train = self.X_train.reshape(len(self.X_train), len(self.clf_features))
        self.X_val = self.validation_df[self.clf_features].values
        self.X_val = self.X_val.reshape(len(self.X_val), len(self.clf_features))
        self.y_train = self.train_df[self.target_feature].values
        self.y_val = self.validation_df[self.target_feature].values

    def train_xgb_model(self, save=True):
        """
        Train xgb model by doing cross-validation and using optimal parameters.

        Args:
           save (bol): whether to save the rf model in models_data.

        Returns
        -------
        clf (xgb model): updates class attribute with classifier to be used for predictions
        val_score (float): error %

        """
        # Perform Grid Search Optimization
        xgb_optimizer = XGBoostOptimization(predictor_features=self.X_train,
                                            target_feature=self.y_train,
                                            objective_function='binary:logistic',
                                            stopping_rounds=50,
                                            grid_scoring='error')
        xgb_optimizer.full_optimization()

        params = xgb_optimizer.best_model_params
        # Train the final classifier with the best parameters
        self.clf = self.__model_instance()
        self.clf.set_params(objective='binary:logistic', n_estimators=params['n_estimators'],
                            learning_rate=params['learning_rate'], max_depth=params['max_depth'],
                            min_child_weight=params['min_child_weight'], gamma=params['gamma'],
                            reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],
                            subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
        self.clf.fit(self.X_train, self.y_train,
                     eval_metric='error',  # evaluation metric used in validation of the model
                     eval_set=[(self.X_val, self.y_val)],  # evaluation set for validating model
                     early_stopping_rounds=20,  # stop training if model doesn't improve after x rounds
                     verbose=True)
        self.y_pred_val = self.clf.predict(self.X_val)
        self.val_score = self.clf.best_score

        # Save th e model and the parameter search
        if save:
            xgb_path = os.path.join(self.__save_path,
                                    'xgb_classifier.pickle.dat')
            pickle.dump(self.clf, open(xgb_path, 'wb'))
            # also save xgb params
            all_eval = pd.DataFrame(xgb_optimizer.evaluated_models_params)
            all_eval = pd.concat([all_eval['evaluated_params'].apply(pd.Series),
                                  all_eval.drop(columns=['evaluated_params'])], axis=1)
            param_save_path = os.path.join(self.__save_path,
                                           'xgb_param_eval.csv')
            all_eval.to_csv(param_save_path, index=False)

    def create_missing_val_imputer(self, strategy='most_frequent'):
        """
        Creates a missing values imputations method for random forest.

        Args:
           strategy (str): based on sk-learn what to replace missing values with.

        Returns
        -------
        X_train_imp (sk-learn model): updates training data within class attributes
        X_val_imp (sk-learn model): updates validation data within class attributes

        """
        # Create our imputer to replace missing values
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp = imp.fit(self.X_train)

        # Impute our data, then train
        self.X_train_imp = imp.transform(self.X_train)
        self.X_val_imp = imp.transform(self.X_val)
        imp_path = os.path.join(self.__save_path,
                                'missing_val_imputer.pickle.dat')
        pickle.dump(imp, open(imp_path, 'wb'))

    def train_rf_model(self, save=True):
        """
        Train random forest model.

        Args:
           save (bol): whether to save the rf model in models_data.

        Returns
        -------
        clf (sk-learn model): updates class attribute with classifier to be used for predictions
        val_score (float): error %

        """
        self.create_missing_val_imputer()
        self.clf = self.__model_instance()
        self.clf.set_params(max_depth=7)
        self.clf.fit(self.X_train_imp, self.y_train)
        self.y_pred_val = self.clf.predict(self.X_val_imp)
        if save:
            rfc_path = os.path.join(self.__save_path,
                                    'rf_classifier.pickle.dat')
            pickle.dump(self.clf, open(rfc_path, 'wb'))
