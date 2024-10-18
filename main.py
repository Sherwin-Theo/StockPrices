import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

data1 = pd.read_csv('indexData.csv')
data2 = data1[data1['Index'].notna()]

forimp = data2.drop(['Index', 'Date'], axis=1)
op2 = data2[['Index', 'Date']]


my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(forimp))
imputed_X_train.columns = forimp.columns
data = pd.concat([op2, imputed_X_train], axis=1)

X_full = data.drop(['Close', 'Date'], axis=1)
y = data['Close']
X_full = pd.get_dummies(X_full)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

model = LinearRegression()
model.fit(X_train_full, y_train)
pred = model.predict(X_valid_full)

print(mean_absolute_error(y_valid, pred))


"""
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train_full, y_train)
pred = forest_model.predict(X_valid_full)

"""

# 0.6812089259747929 with random forest
# 0.62 with XGB