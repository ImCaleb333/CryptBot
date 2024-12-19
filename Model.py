import pandas as pd
import numpy as np
import joblib

csv_file = "1INCH.csv"
data = pd.read_csv(csv_file)

# CLEAN DATA
from scipy.stats import linregress
from scipy.stats import zscore
from scipy import stats
import matplotlib.pyplot as plt
import seaborn

# drop unwanted things
data = data.dropna()
data = data.drop(["ticker","date"],axis=1)

# SPLIT DATA
from sklearn.model_selection import train_test_split

x = data.drop("close",axis=1)
y = data["close"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# BUILD MODEL
from sklearn.tree import DecisionTreeRegressor

# establish model
model = DecisionTreeRegressor()
model.fit(x_train,y_train)

# EVALUATE MODELS
y_model_train_pred = model.predict(x_train)
y_model_test_pred = model.predict(x_test)

# evaluate models performance
from sklearn.metrics import mean_squared_error, r2_score

# training predictions
training_mse = mean_squared_error(y_train,y_model_train_pred)
training_r2 = r2_score(y_train,y_model_train_pred)

# testing predictions
testing_mse = mean_squared_error(y_test,y_model_test_pred)
testing_r2 = r2_score(y_test,y_model_test_pred)

#get results
model_results = pd.DataFrame(["LinearRegression", training_mse, training_r2, testing_mse, testing_r2]).transpose()
model_results.columns = ["Model","Training_MSE","Training_R2","Testing_MSE", "Testing_R2"]

# TEST MODEL
input_data = np.array([[1.244,1.547,1.1000]])

model_prediction = model.predict(input_data)

print(data.head())