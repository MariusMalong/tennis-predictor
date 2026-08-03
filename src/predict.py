import joblib
import pandas as pd

model = joblib.load('models/xgb_model.pkl')
feature_cols = ('models/feature_cols.pkl')

def predict_match(rank_diff, points_diff, points_data_missing,
                  career_win_rate_diff, surface_win_rate_diff,
                  recent_form_diff, surface_clay, surface_grass, surface_hard):
    input_data = pd.DataFrame([{
        'rank_diff': rank_diff,
        'points_diff': points_diff,
        'points_data_missing': points_data_missing,
        'career_win_rate_diff': career_win_rate_diff,
        'surface_win_rate_diff': surface_win_rate_diff,
        'recent_form_diff': recent_form_diff,
        'Surface_Clay': surface_clay,
        'Surface_Grass': surface_grass,
        'Surface_Hard': surface_hard
    }])[feature_cols]

    probability = model.predict_proba(input_data)[0][1]
    return {'player_1_win_probability': round(float(probability), 4)}