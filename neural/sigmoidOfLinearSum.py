import numpy as np
import sys
sys.path.append("..\tools")
from functions import sigmoid


inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
weights_sum = np.dot(inputs, weights)
print("weights_sum: {} "  .format(weights_sum))
linear_sum = weights_sum + bias
print("linear_sum: {}" .format(linear_sum))

#calculate sigmoid of the linear sum
output = sigmoid(linear_sum)

print('Output:')
print(output)
