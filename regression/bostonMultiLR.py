from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# Load the data from the the boston house-prices dataset 
# attributes : https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
# data : https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
# source code for load_boston : https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/base.py

boston_data = load_boston()

#print stats

print boston_data['target'] #1D numpy array of target attribute values
print boston_data['data'].shape # 2D numpy array of attribute values
print boston_data['feature_names'] # 1D numpy array of names of the attributes
print boston_data['DESCR']  # text description of the dataset


x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to the model variable
model = LinearRegression()
model.fit(x,y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = model.predict(sample_house)
print prediction