from fastapi import FastAPI
from pydantic import BaseModel,ValidationError,validator
from ML_Module.model import ModelApp
from typing import List
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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


    @validator("data")
    def check_dimensionality(cls, v):
        n_features = 13
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")

        return v


class PredictResponse(BaseModel):
    data: List[float]


@app.post("/predict", response_model=PredictResponse)
def predict(input_data: PredictRequest):

    # input data process
    input_data = np.array(input_data.data[0])
    input_data = model_app.input_process_component(input_data)

    # model prediction
    out = model_app.model.predict(input_data)

    # output data process
    out = model_app.output_process_component(out)

    return PredictResponse(data=[out])
