import pickle
from pathlib import Path
import os
import torch
from torch import nn


def general_predictor(data_df, features, model_name, model_type='xgb', imputation=False):
    """
    Will apply prediction model to a dataframe
    Args:
        data_df (dataframe): dataframe containing all accounting variables
        features ('str' or list['str']): dimension or list of dimensions used for predicting
        model_name ('str'): the model file path name
        model_type ('str'): whether the models are xgb, to use best tree iteration
    Returns:
        Array of predictions, probabilities
    """

    # Create variable for the features to train the classifier
    features_data = data_df[features].values
    # Reshape to 2d array
    features_data = features_data.reshape(len(features_data), len(features))
    # Define the path for the model
    model_path = os.path.join(str(Path(__file__).resolve().parent.parent),
                              'models_data', model_name)
    # Load the classifier inside model
    model = pickle.load(open(model_path, 'rb'))

    if imputation:
        imp_model_path = os.path.join(str(Path(__file__).resolve().parent.parent),
                                      'models_data', 'missing_val_imputer.pickle.dat')
        imp_model = pickle.load(open(imp_model_path, 'rb'))
        features_data = imp_model.transform(features_data)

    if model_type == 'xgb':
        # Use best iteration for xgb
        predictions = model.predict(features_data, ntree_limit=model.best_iteration)
        probabilities = model.predict_proba(features_data, ntree_limit=model.best_iteration)
    else:
        predictions = model.predict(features_data)
        probabilities = model.predict_proba(features_data)

    return predictions, probabilities


def nn_predict(data_df, features, model_name):
    """
    Will apply neural network prediction model to a dataframe
    Args:
        data_df (dataframe): dataframe containing all accounting variables
        features ('str' or list['str']): dimension or list of dimensions used for predicting
    Returns:
        Array of probabilities of class 1
    """

    # use gpu if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Instantiate the nn and load the model
    model = nn.Sequential(nn.Linear(len(features), 6),
                          nn.Sigmoid(),
                          nn.Linear(6, 1),
                          nn.Sigmoid()).to(device)

    # Define the model path and load the weights
    model_path = os.path.join(str(Path(__file__).resolve().parent.parent),
                              'models_data', model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # create variable for the features to train the classifier
    features_data = data_df[features].values
    # reshape to 2d array
    features_data = features_data.reshape(len(features_data), len(features))
    # transform to tensor
    features_data_tensor = torch.Tensor(features_data).to(device)
    # use the model to make predictions
    clf_pred = model(features_data_tensor).cpu().detach().numpy()

    return clf_pred
