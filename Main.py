import pandas as pd
from sklearn.linear_model import LinearRegression
from MultipleLinearRegression import *

#Load Stack Overflow dataset
df = pd.read_csv("stack_overflow_tags.csv")

#After going through the data on jupyter, all nulls have already been removed and there are no
# edge cases that would cause any unexpected issues
reviewed = df

#There are the features that were selected by me
X = reviewed[["total_comments", "total_answers"]]

#This is the label that was selected by me
y = reviewed["total_posts"]

#Train model on data
reg = LinearRegression().fit(X, y)
myReg = MultipleLinearRegression().fit(X, y)

#Results for sklearn Linear Regression
print("***sklearn Linear Regression***")
print(f"R Squared: {reg.score(X, y)}")
print(f"Intercept: {reg.intercept_}")
print(f"Coefficients: {reg.coef_}")
print(f"RMSE: {np.sqrt(np.average((y-reg.predict(X))**2.0))}")

print()

#Results for my Linear Regression
print("***My Linear Regression***")
print(f"R Squared: {myReg.score(X, y)}")
print(f"Intercept: {myReg.intercept_}")
print(f"Coefficients: {myReg.coef_}")
print(f"RMSE: {myReg.RMSE(X, y)}")
