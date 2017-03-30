import numpy as np
import os
import sys
nn_tools_path = os.path.expanduser("~\\deep\\neural\\tools") 
sys.path.append(nn_tools_path)
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """
    Given Sigmoid, calculate sigmoid derivative
    """
    return x*(1-x)


# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
print("features shape: {}",format(n_features))
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.9

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        #  Calculate the output
        h = np.dot(x,weights)
        output = sigmoid(h)

        #  Calculate the error
        error = y-output

        #  Calculate the error term
        output_gradient = sigmoid_prime(output)
        error_term = error * output_gradient

        #  Calculate the change in weights for this sample
        #       and add it to the total weight change
        
        del_w += learnrate * error_term * x

    #  Update weights using the learning rate and the average change in weights
    weights += learnrate*del_w/n_records

    # Printing out the mean square error on the training set
    
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing at :", e)
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))