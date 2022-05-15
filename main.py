from fastapi import FastAPI
from pydantic import BaseModel
from model import ModelApp
from typing import List
import numpy as np

app = FastAPI(
    description="FastAPI for MachineLearning Application",
    version="0.1"
)

# 实例化 应该就加载了
model_app = ModelApp()


# api
@app.get('/')
async def index():
    return {"info": "Boston House Pricing"}


class PredictRequest(BaseModel):
    data: List[List[float]]


class PredictResponse(BaseModel):
    data: List[float]


@app.post("/predict", response_model=PredictResponse)
async def predict(input_data: PredictRequest):
    print(input_data)
    input_data = np.array(input_data.data[0])


    # feature selection
    input_data = input_data[[True, False, False, False, True, True, False, True, False,
                   False, False, False, True]]

    input_data = model_app.x_scaler.transform(input_data.reshape((1,-1)))

    model = model_app.model
    out = model.predict(input_data)
    out = model_app.y_scaler.inverse_transform(out.reshape((1,-1)))
    print(out)

    return PredictResponse(data=[out])
