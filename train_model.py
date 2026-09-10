import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

#sample training data
data = pd.DataFrame({
    "rooms": [2, 3, 4, 5, 3, 4],
    "age": [20, 15, 10, 5, 12, 7],
    "distance": [10, 8, 5, 3, 6, 4],
    "price": [100, 150, 200, 280, 180, 250]
})

X = data[["rooms", "age", "distance"]]
y = data["price"]

#pipeline for preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X, y)