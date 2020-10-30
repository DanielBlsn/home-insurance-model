import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedShuffleSplit   # Split into folds
from itertools import product


class XGBoostOptimization:
    """
    Analyse the optimal parameters for XGBoost models for a given dataset
    Run full_optimization which will create the optimal XGB params
    
    Args:
        predictor_features (numpy array): features/dimensions used for modelling
        target_feature (numpy array): predicted data array
        objective_function (str): 'multi:softmax' for multi-classification or 'reg:squarederror' for regression
        cv_folds (int): number of folds to split the data, default 4
        stopping_rounds (int): number of early stopping rounds, default 20
        grid_scoring (str): metric used to optimize gboost parameters (e.g. merror, rmse)
        
    Usage:
        xgboost_optimization = XGBoostOptimization(predictor_features, target_feature)
        xgboost_optimization.full_optimization()
        print(xgboost_optimization.best_model_params)

    Attributes:
        best_depth (int): optimal xgb tree depth
        best_cw (int): optimal xgb child weight
        best_gamma (float): optimal xgb gamma
        best_subsample (float): optimal xgb subsample
        best_colsample (float): optimal xgb colsample
        best_reg_lambda (float): optimal xgb lambda regularization
        best_reg_alpha (float): optimal xgb alpha regularization
        best_lr (float): optimal xgb learning rate
        best_num_trees (int): optimal xgb number of trees
        evaluated_models_params (list): list of dicts containing all evaluated parameters and scores
        best_model_params (dict): contains aggregation of best parameters obtained for the dataset
    """

    def __init__(self, predictor_features, target_feature, objective_function='multi:softmax',
                 cv_folds=4, stopping_rounds=20, grid_scoring='merror'):

        self.predictor_features = predictor_features
        self.target_feature = target_feature
        self.cv_folds = cv_folds
        self.stopping_rounds = stopping_rounds
        self.objective_function = objective_function
        self.grid_scoring = grid_scoring
        self.best_score = None
        self.evaluated_models_params = []
        self.best_model_params = {'colsample_bytree': 1,
                                  'gamma': 0,
                                  'learning_rate': 0.1,
                                  'max_depth': 6,
                                  'min_child_weight': 7,
                                  'n_estimators': 1000,
                                  'reg_alpha': 0,
                                  'reg_lambda': 1,
                                  'subsample': 1}

    def determine_objective(self, **kwargs):
        """
        Will determine whether classification or regression is the task
        Based on the objective function declared with instantiating the class
        """
        if self.objective_function == 'multi:softmax':
            init_estimator = xgb.XGBClassifier(objective=self.objective_function)
        elif self.objective_function == 'reg:squarederror':
            init_estimator = xgb.XGBRegressor(objective=self.objective_function)
        elif self.objective_function == 'binary:logistic':
            init_estimator = xgb.XGBClassifier(objective=self.objective_function)
        else:
            print("Objective function needs to be multi:softmax or reg:squarederror")

        # initiate with best/default model params
        current_params = self.best_model_params
        # update current iteration with current params
        for key, value in kwargs.items():
            current_params[key] = value
        # update initial parameters
        init_estimator.set_params(colsample_bytree=current_params['colsample_bytree'],
                                  gamma=current_params['gamma'],
                                  learning_rate=current_params['learning_rate'],
                                  max_depth=current_params['max_depth'],
                                  min_child_weight=current_params['min_child_weight'],
                                  n_estimators=current_params['n_estimators'],
                                  reg_alpha=current_params['reg_alpha'],
                                  reg_lambda=current_params['reg_lambda'],
                                  subsample=current_params['subsample'])

        return init_estimator

    def default_estimator(self, estimator):
        """
        Estimator with initial values based on default parameters
        Returns:
            default parameters updated in self
        """

        # perform the grid search and extract the best parameters
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=self.best_model_params)

        # update best model params
        self.best_score = gscore


    def optimize_depth_chweight(self, estimator):
        """
        Optimize max_depth and min_child_weight parameters with grid_search
        Returns:
            best_depth: optimal max_depth as init class variable
            best_cw: optimal min_child_weight as init class variable
        """

        param_test_one_a = {'max_depth': [i for i in range(3, 10, 2)],
                            'min_child_weight': [i for i in range(1, 8, 2)]}

        # perform the grid search and extract the best parameters
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_one_a)

        best_depth = gsearch['max_depth']
        best_cw = gsearch['min_child_weight']

        # narrowing down of values +1 around optimals
        param_test_one_b = {'max_depth': [i for i in range(best_depth-1, best_depth+2, 1)],
                            'min_child_weight': [i for i in range(best_cw-1, best_cw+2, 1)]}
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_one_b)

        print(gscore < self.best_score)
        if gscore < self.best_score:
            # update best model params
            self.best_score = gscore
            self.best_model_params['max_depth'] = gsearch['max_depth']
            self.best_model_params['min_child_weight'] = gsearch['min_child_weight']

    def optimize_gamma(self, estimator):
        """
        Optimize gamma parameter with grid_search
        Returns:
            best_gamma: optimal gamma as init class variable
        """

        param_test_two = {'gamma': [0.1, 0.2, 0.3, 0.5, 1]}
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_two)

        if gscore < self.best_score:
            # update best model params
            self.best_score = gscore
            self.best_model_params['gamma'] = gsearch['gamma']

    def optimize_subsample_colsample(self, estimator):
        """
        Optimize subsample and colsample_bytree parameters with grid_search
        Returns:
            best_subsample: optimal subsample as init class variable
            best_colsample: optimal colsample_bytree as init class variable
        """

        param_test_three_a = {'subsample': [.6, .7, .8, .9, .10],
                              'colsample_bytree': [.6, .7, .8, .9, .10]}
        # perform the grid search and extract the best parameters
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_three_a)
        # placeholder temp best values
        best_subsample = gsearch['subsample']
        best_colsample = gsearch['colsample_bytree']

        # narrowing down of values 0.05 around the optimals that cannot exceed 1
        param_test_three_b = {'subsample':[i for i in np.arange(best_subsample-0.05, best_subsample+0.1, 0.05) if i<=1],
                              'colsample_bytree':[i for i in np.arange(best_colsample-0.05, best_colsample+0.1, 0.05) if i<=1]}
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_three_b)

        if gscore < self.best_score:
            # update best model params
            self.best_score = gscore
            self.best_model_params['subsample'] = gsearch['subsample']
            self.best_model_params['colsample_bytree'] = gsearch['colsample_bytree']

    def optimize_lambda_alpha(self, estimator):
        """
        Optimize lambda and alpha regularization parameters with grid_search
        Returns:
            best_reg_lambda: optimal alpha as init class variable
            best_reg_alpha: optimal lambda as init class variable
        """
        param_test_four = {'reg_alpha': [0, 0.3, 0.7, 1],
                           'reg_lambda': [1, 1.5, 2, 3, 4.5]}
        # perform the grid search and extract the best parameters
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test_four)

        if gscore < self.best_score:
            # update best model params
            self.best_score = gscore
            self.best_model_params['reg_lambda'] = gsearch['reg_lambda']
            self.best_model_params['reg_alpha'] = gsearch['reg_alpha']

    def optimize_lr_numtrees(self, estimator):
        """
        Optimize learning rate and number of trees to use
        Returns:
            best_reg_lambda: optimal alpha as init class variable
            best_reg_alpha: optimal lambda as init class variable
        """
        param_test= {'learning_rate':[0.1, 0.05, 0.2, 0.3, 0.5]}
        # perform the grid search and extract the best parameters
        gsearch, gscore = self.grid_search_cv(estimator=estimator, param_grid=param_test)

        if gscore < self.best_score: 
            # update best model params
            self.best_score = gscore
            self.best_model_params['learning_rate'] = gsearch['learning_rate']


    def full_optimization(self):
        """
        Run complete parameter optimization, step by step
        Afte completion, class attributes will be updated and accessible
        """
        
        # define the standard estimator and it's results
        estimator = self.determine_objective()
        self.default_estimator(estimator)
        # optimize max_depth and min_child_weight
        self.optimize_depth_chweight(estimator)
        print('Current best parameters are: ', self.best_model_params)
        # optimize gamma
        self.optimize_gamma(estimator)
        print('Current best parameters are: ', self.best_model_params)
        # optimize subsample and colsample
        self.optimize_subsample_colsample(estimator)
        print('Current best parameters are: ', self.best_model_params)
        # optimize lambda and alpha values
        self.optimize_lambda_alpha(estimator)
        print('Current best parameters are: ', self.best_model_params)
        # optimize learning rate and number of trees
        self.optimize_lr_numtrees(estimator)
        print('Current best parameters are: ', self.best_model_params)

    def parameter_grid(self, parameters_dict):
        """
        Creates the grid of parameters for cross-validation search
        Format needs to be {'searchable_item':[value1, value2]}
        Returns:
            param_grid (list): all possible combinations of values for each searchable_item
        """

        param_grid = []

        for p in [parameters_dict]:
            # iterator of tuples containing parameter on one side and list of parameter values
            keys, values = zip(*p.items())

            for v in product(*values):
                # dict out of all the combinations of keys and values
                params = dict(zip(keys, v))

                param_grid.append(params)

        return param_grid

    def create_folds(self, predictor_features, target_feature):
        """
        Creates folds of predictor/target variables to perform cross-validation on
        Returns:
            test_sets_x (list): list of np arrays with test predictor features
            test_sets_y (list): list of np arrays with test target features
            validation_sets_x (list): list of np arrays with valdiation predictor features
            validation_sets_y (list): list of np arrays with validation target features
        """
        # instantiace blank lists
        train_sets_x = []
        train_sets_y = []
        validation_sets_x = []
        validation_sets_y = []

        # instantiate the folds required and create splits
        kf = StratifiedShuffleSplit(n_splits=self.cv_folds)

        # iterate over indices to create the splits of data
        for train_index, test_index in kf.split(predictor_features, target_feature):

            train_sets_x.append(np.array(predictor_features)[train_index])
            train_sets_y.append(np.array(target_feature)[train_index])

            validation_sets_x.append(np.array(predictor_features)[test_index])
            validation_sets_y.append(np.array(target_feature)[test_index])

        return train_sets_x, train_sets_y, validation_sets_x, validation_sets_y

    def grid_search_cv(self, estimator, param_grid):
        """
        Performs grid search using an xgb estimator over a grid of parameters
        Returns:
            best_grid (dict): dictionary of best parameters selected in the respective grid
            temp_best_grid_score (float): average score for the best grid parametrs over the folds
        """

        # expand the parameter grid with all combinations
        for key, value in param_grid.items():
            if type(value) is not list:
                param_grid[key] = [value]
        expanded_param_grid = self.parameter_grid(param_grid)
        # create the folds to perform cv on
        X_train, y_train, X_test, y_test = self.create_folds(self.predictor_features, self.target_feature)

        temp_best_grid_score = None
        temp_best_validation_score = None
        best_grid = None

        # perform grid_search and save the results
        for grid in expanded_param_grid:

            # instantiate variable to store scores for each fold
            temp_grid_score = []
            temp_validation_score = []

            # define the temp parameter grid to use with best current values and the grid changes
            updated_grid = {k: grid[k] if k in grid else self.best_model_params[k] for k in self.best_model_params}
            print(updated_grid)
            # define the xgb estimator with the changed parameters first
            estimator = self.determine_objective(**updated_grid)

            # fit the estimator over each fold
            for fold in range(0, len(X_train)):
                estimator.fit(X_train[fold], y_train[fold],
                              eval_metric=self.grid_scoring,  # evaluation metric used in validation of the model
                              eval_set=[(X_test[fold], y_test[fold])],  # evaluation set for validating model
                              early_stopping_rounds=self.stopping_rounds,  # stop training if model doesn't improve after x rounds
                              verbose=False)
                temp_grid_score.append(estimator.best_score)

                # extract the validation score over the last n stopping rounds folds
                evals_result = estimator.evals_result()
                all_evals_result = []
                for validation in list(reversed(list(evals_result)))[0:self.stopping_rounds]:
                    all_evals_result.append(evals_result[validation][self.grid_scoring][0])

                temp_validation_score.append(np.mean(all_evals_result))

            print('Current parameters evaluated: {} with the score {} and the validation score: {}'.format(grid,
                                                                                                       np.mean(temp_grid_score),
                                                                                                       np.mean(temp_validation_score)))

            # update the best grid if this grid params check shows better values
            if temp_best_grid_score is None:
                temp_best_grid_score = np.mean(temp_grid_score)
                temp_best_validation_score = np.mean(temp_validation_score)
                best_grid = grid
            elif np.mean(temp_validation_score) < temp_best_grid_score:
                temp_best_grid_score = np.mean(temp_grid_score)
                temp_best_validation_score = np.mean(temp_best_validation_score)
                best_grid = grid

            # updated the evaluated models params
            self.evaluated_models_params.append({'evaluated_params': estimator.get_params(),
                                                 'score': np.mean(temp_grid_score),
                                                 'validation_score': np.mean(temp_best_validation_score)})

        return best_grid, temp_best_validation_score
