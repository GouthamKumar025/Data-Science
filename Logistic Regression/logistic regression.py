from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

lreg = LogisticRegression()

lreg.fit(X_train,y_train)

y_pred =lreg.predict(X_test)

print(y_pred)

#calculating the coefficient
print("Coefficient: ",lreg.coef_)
#calculating the mean squared error
print("Mean Squared error: ",mean_squared_error(y_test, y_pred))
#calculating the coefficient of determination
print("Coefficient of determination: ",r2_score(y_test, y_pred))
