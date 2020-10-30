from modules import ml_models
from build_results import BuildResults
import pandas as pd
import numpy as np
from pathlib import Path
import os
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


class BuildAggregator(BuildResults):
    def __init__(self, data_df,
                 target_feature, clf_features,
                 clf_agg_features):
        # Inherit the build results class to make first layer of predictions
        super().__init__(data_df,
                         target_feature, clf_features,
                         clf_agg_features)

    def train_aggregator(self):
        """
        Train neural network aggregator.

        Args:
           None.

        Returns
        -------
        data_df (df): updates class attribute with nn predictions

        """
        self.predict_first_layer()
        self.data_df.sample(frac=1)
        self.data_df.reset_index(inplace=True, drop=True)

        # If gpu is available
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Extract the train and test features
        X_train = self.data_df[self.clf_agg_features].values
        X_train = X_train.reshape(len(X_train), len(self.clf_agg_features))
        y_train = self.data_df[self.target_feature].values

        # Transform input/output data to tensors and reshape to correct format
        output_data_tensor = torch.Tensor(y_train).view(len(y_train), 1).to(device)
        input_data_tensor = torch.Tensor(X_train).view(X_train.shape[0],
                                                       X_train.shape[1]).to(device)

        # Define the neural network shape
        nn_model = nn.Sequential(nn.Linear(input_data_tensor.shape[1], 6),
                                 nn.Sigmoid(),
                                 nn.Linear(6, 1),
                                 nn.Sigmoid()).to(device)
        # Define the loss criterion - Binary Cross Entropy
        criterion = nn.BCELoss()
        # Define the optimizer and learning rate
        optimizer = optim.Adam(nn_model.parameters(), lr=0.05)
        # Define number of epochs
        epochs = 5000

        for e in range(epochs):

            # Forward propagaion based on input data
            forward_prop = nn_model(input_data_tensor)
            # Calculate the loss
            loss = criterion(forward_prop, output_data_tensor)
            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()
            # Backward propagation
            loss.backward()

            optimizer.step()
            print(loss.item(), e)
            train_pred = np.round(forward_prop.cpu().detach().numpy())
            print(accuracy_score(y_train, train_pred))

        # Switch model to eval mode and create predictions
        nn_model.eval()
        nn_model.state_dict()
        predictions = nn_model(input_data_tensor).cpu().detach().numpy()
        self.data_df = self.data_df.assign(predicted_ev=predictions)

        # Save model using the inherited class dynamic path
        nn_path = os.path.join(self._BuildResults__save_path, 'neural_network.pth')
        torch.save(nn_model.state_dict(), nn_path)
