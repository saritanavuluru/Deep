
# TODO: Add import statements
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_fwf("bmi_and_life_expectancy.csv")
x = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]


# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x,y)


# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)


#visualize the fit model
bmi_life_predict = bmi_life_model.predict(x)
plt.scatter(x,y)
plt.plot(x,bmi_life_predict)
plt.show()

#To figure - how do i plot the point <21.07931,laos_life_exp> in a different color on the same scatter plot ?