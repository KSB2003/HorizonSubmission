import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mpl
from sklearn.preprocessing import PolynomialFeatures


class Preprocessing():
    x, y = None, None
    def readingdata(self, filename):
        data = pd.read_csv(filename)
        self.x = data.iloc[:, 1:2].values
        self.y = data.iloc[:, 2].values
    def impute(self):
        pass
    def encoder(self):
        pass
    def splitting(self):
        xtraining, xtest, ytraining, ytest = train_test_split(self.x, self.y, test_size=0, random_state=100)
        return xtraining, xtest, ytraining, ytest

class Regressor():
    x, y, regressorl, regressorp, polynomialfeatures = None, None, None, None, None
    def loaddata(self, x, y):
        self.x = x
        self.y = y

    def trainlinear(self):
        self.regressorl = LinearRegression()
        self.regressorl.fit(self.x, self.y)

    def predictl(self, xtest):
        ypredicted = self.regressorl.predict(xtest)
        return ypredicted



    def trainpoly(self):
        self.polynomialfeatures = PolynomialFeatures(degree=3)
        self.xpolynomial = self.polynomialfeatures.fit_transform(self.x)
        self.regressorp = LinearRegression()
        self.regressorp.fit(self.xpolynomial, self.y)

    def predictp(self, xtestp):
        ypredicted = self.regressorp.predict(self.polynomialfeatures.fit_transform(xtestp))
        return ypredicted

    def plotter(self):
        mpl.title("level vs salaries")
        mpl.xlabel("Level")
        mpl.ylabel("Salary")
        mpl.scatter(self.x, self.y, color='#2cb833')
        mpl.plot(self.x, self.regressorl.predict(self.x), color='#2cb833')
        mpl.plot(self.x, self.regressorp.predict(self.xpolynomial), color='#ff0000')
        mpl.show()



def runner():
    preprocesser = Preprocessing()
    preprocesser.readingdata('Position_Salaries.csv')
    # xtrain, xtest, ytrain, ytest = preprocesser.splitting()
    regressor = Regressor()
    regressor.loaddata(preprocesser.x, preprocesser.y)
    #LINEAR
    regressor.trainlinear()
    xtest = [[9.8]]
    ypredictl = regressor.predictl(xtest)

    #POLYNOMIAL
    regressor.trainpoly()
    ypredictp = regressor.predictp(xtest)

    print(ypredictl, ypredictp)



    # i = 0
    # while i<len(ypredictl):
    #     print("{} {:.2f} {:.2f}".format(ytest[i], ypredictl[i], ypredictp[i]))
    #     i = i+1

    regressor.plotter()


runner()





