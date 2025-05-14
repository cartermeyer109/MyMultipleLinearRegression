# Import the numpy library for math operations
import numpy as np

# Define the MultipleLinearRegression class
class MultipleLinearRegression:
    # Default constructor sets coefficients and intercept to none
    def __init__(self):
        self.coef_ = None  # This will hold the coefficients (weights) for the features
        self.intercept_ = None  # This will hold the intercept (bias term)

    # The fit function calculates the coefficients and intercepts by training it on the entered data with
    # the features and labels
    def fit(self, X, y):
        # M is the feature matrix X with a column of ones prepended. The column of ones will be multiplied with
        # the intercept term (bias).
        #column_stack = combine vec/mat, np.ones(X.shape[0]) = vector of 1's with the row count of X, X = the matrix X
        M = np.column_stack((np.ones(X.shape[0]), X))

        # beta is the vector of coefficients
        # For linear regression B = ((XT*X)**-1)XT*y
        # np.linalg.inv() = inverse of a matrix, @ = matrix multiplication operator, .T = transpose a vector/matrix
        beta = np.linalg.inv(M.T @ M) @ M.T @ y

        # The first element of beta is the intercept, and the rest are the coefficients for each feature.
        self.intercept_ = beta[0]  # The intercept is the first element of beta
        self.coef_ = beta[1:]  # The coefficients for the features are the remaining elements of beta

        return self

    # Using the model trained from the "fit" function, you can enter features to see what the predicted label values
    # are. Multiplies X matrix of features with the coefficient (beta vector) + intercept
    def predict(self, X):
        return X @ self.coef_ + self.intercept_  # Return the predicted values

    # The score function calculates how well the model performs by returning the R-squared score, which is one of
    # the ways to score the accuracy of a model. This shows how much of the variance is explained by the model.
    def score(self, X, y):
        # np.sum() adds all elements in the array, and np.average() calculates the average of the elements.
        # This returns R squared score = 1 - ((summation(yi - yhati)**2)/(summation(yi-ymean)**2))
        return 1.0 - np.sum((y - self.predict(X)) ** 2.0) / np.sum((y - np.average(y)) ** 2.0)

    # The RMSE function calculates the Root Mean Squared Error, which is another method of scoring the accuracy
    # of a model. It is the average error between predicted and actual values.
    def RMSE(self, X, y):
        # np.sqrt() calculates the square root, and np.average() computes the mean of the squared errors.
        #This returns RMSE = sqrt((1/N)summation(yhati-yi)**2)
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))
