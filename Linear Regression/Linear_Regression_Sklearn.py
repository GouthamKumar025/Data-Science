from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_diabetes(return_X_y=True)
X = X[:,np.newaxis,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(X_test.shape,y_test.shape)
print(X_test.ndim,y_test.ndim)
print(y_pred)

# calculating the coefficients
print("Coefficients: ", reg.coef_)
# calculating the mean squared error
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
# calculating the coefficient of determination
print("Coefficient of Determination: ", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()

