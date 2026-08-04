from fastapi import FastAPI
from src.predict import predict_match

app = FastAPI()

@app.get("/predict")
def get_prediction(rank_diff: float, points_diff: float, points_data_missing: int,
                   career_win_rate_diff: float, surface_win_rate_diff: float,
                   recent_form_diff: float, surface_clay: bool, surface_grass: bool,
                   surface_hard: bool):
    return predict_match(rank_diff, points_diff, points_data_missing,
                         career_win_rate_diff, surface_win_rate_diff,
                         recent_form_diff, surface_clay, surface_grass, surface_hard)
@app.get("/")
def read_root():
    return{
        "message": "ATP Tennis Match Predictor API",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }