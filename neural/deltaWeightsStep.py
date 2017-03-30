import numpy as np

#TODO : move sigmoid and derivate to the functions file in tools directory

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight

# TODO: Calculate the node's linear combination of inputs and weights
h = np.dot(x,w)

# TODO: Calculate output of neural network
nn_output = sigmoid(h)

# TODO: Calculate error of neural network (output error)
error = y-nn_output

# TODO: Calculate output gradient (derivative of sigmoid)

output_gradient = sigmoid_prime(h)

# TODO: Calculate the error term (delta)

error_term = error * output_gradient

# TODO: Calculate change in weights [list]

del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)