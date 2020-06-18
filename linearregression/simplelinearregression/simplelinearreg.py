import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#READING THE DATA
data = pd.read_csv('Salary_Data.csv')


#ESTABLISHING DEPENDENT AND INDEPENDENT VARIABLES
x = data.iloc[0:30, 0:-1].values
y = data.iloc[0:30, 1].values

#SPLITTING INTO TRAINING AND TESTING
xtraining, xtest, ytraining, ytest = train_test_split(x, y, test_size=0.25, random_state=20)


#TRAINING AND TESTING
regressor = LinearRegression()
regressor.fit(xtraining, ytraining)
ypredicted = regressor.predict(xtraining)



mpl.scatter(xtraining, ytraining, color = '#0b5e14')
mpl.scatter(xtest, ytest, color = '#24cbd4')
mpl.title("EXPERIENCE VS SALARY")
mpl.xlabel("YEARS OF EXPERIENCE")
mpl.ylabel("SALARY")
mpl.plot(xtraining, ypredicted, color = '#ff0000')
mpl.show()



