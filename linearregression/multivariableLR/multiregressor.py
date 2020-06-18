import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Preprocessing():
    x, y = None, None


    def readingdata(self, filename):
        data = pd.read_csv(filename)
        self.x = data.iloc[:, 0:4].values
        self.y = data.iloc[:, 4].values
    def impute(self):
        pass

    def encoder(self):
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        self.x = np.array(transformer.fit_transform(self.x))
    def splitting (self):
        xtraining, xtest, ytraining, ytest = train_test_split(self.x, self.y, test_size=0.3, random_state=10)
        return xtraining, xtest, ytraining, ytest

class Multiregressor():
    x, y, regressor = None, None, None

    def trainmodel(self, x, y):
        self.regressor = LinearRegression()
        self.regressor.fit(x, y)

    def predict(self, xtest):
        y_predicted = self.regressor.predict(xtest)
        return y_predicted

def runner():
    regressor = Preprocessing()
    regressor.readingdata('50_Startups.csv')
    regressor.impute()
    regressor.encoder()
    xtraining, xtest, ytraining, ytest = regressor.splitting()
    regressor2 = Multiregressor()
    regressor2.trainmodel(xtraining, ytraining)
    ypredicted = regressor2.predict(xtest)




    i = 0
    print("{} {}".format("Predicted", "Actual"))
    while i<len(ytest):
        print("{:.2f} {:.2f}".format(ypredicted[i], ytest[i]))
        i = i+1

runner()










