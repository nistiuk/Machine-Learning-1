import pandas
import numpy
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score

#Linear Regression
h= pandas.read_csv("house.csv")
size=h['Size ']
price= h['Price ']
slope, intercepts, r,p,err= stats.linregress(size,price)
def myfunc(x):
    return slope * x + intercepts
mymodel= list(map(myfunc,size))
predicted_linear= myfunc(1520)

#Multiple Regression
X= h[['Size ','Bedrooms', 'Age ','Distance']]
y= h['Price ']
regr= linear_model.LinearRegression()
regr.fit(X,y)
predicted_multi= regr.predict([[1520,2,0,10]])

#Polynomial Regression
mymodel1= numpy.poly1d(numpy.polyfit(size,price,3))
a= numpy.linspace(0,2000,1000)
print(r2_score(price, mymodel1(size)))
predicted_poly= mymodel1(1520)

print("Linear Regression: ",predicted_linear,"  R: ",r,"\n")
print("Multiple Regression: ",predicted_multi,"   Coeff: ",regr.coef_,"\n")
print("Polynomial Regression: ",predicted_poly,"    R2: " ,r2_score(mymodel1(size),price),"\n")


