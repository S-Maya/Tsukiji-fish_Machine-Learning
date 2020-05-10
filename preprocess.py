import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.preprocessing import Normalizer
def BuildDataFrame(path):
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return pd.read_csv(f, header='infer')

def BuildAndTrainModel(csv_path,choice_model,DROP_FEATURES=[]):
    SELECTED_X_FEATURES=['Length1', 'Length2', 'Length3', 'Height', 'Width']
    Y_COL = ['Weight']
    COLS = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    all_data=BuildDataFrame(csv_path)
    y = all_data.drop(columns=[c for c in COLS if c not in Y_COL], inplace=False)
    X = all_data.drop(columns=Y_COL+DROP_FEATURES, inplace=False)
    choice_model.fit(X,y)
    return choice_model