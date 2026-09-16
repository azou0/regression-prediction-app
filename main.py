from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="House Price Prediction API")

#load model once at startup
model = joblib.load("house_price_model.joblib")

class HouseInput(BaseModel):
    rooms:int
    age:float
    distance:float

@app.post("/predict")
def predict_price(data:HouseInput):
    features = [[
        data.rooms,
        data.age,
        data.distance
    ]]

    prediction = model.predict(features)

    return {
        "predicted price": round(prediction[0],2)
    }