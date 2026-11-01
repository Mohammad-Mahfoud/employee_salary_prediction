import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("Dataset/Employee-salary-prediction.csv")

## dropping null values: 
df = df.dropna()
print(df.head())

## applying one hot encoding on string columns so the model can work on them : 
df = pd.get_dummies(df,columns=["Gender","Education Level","Job Title"],dtype=int)

print(df.info())

## splitting into features and target : 

X = df.drop("Salary",axis=1)
y = df["Salary"]

## creating train and test variables on features and target :
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## implementing the linear model: 
model = LinearRegression()

## fitting the training data into  the model: 
model.fit(X_train,y_train)

# predicting on the model : 
y_pred = model.predict(X_test)


## using mae and mse to see our results :
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")