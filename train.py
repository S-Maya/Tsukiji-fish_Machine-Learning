import pandas as pd
import numpy as np
import os
from preprocess import prep_data
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

csv_path = os.path.join("fish_participant.csv")
df = pd.read_csv(csv_path)

X, y = prep_data(df)

gbr = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, loss='ls')
    
gbr.fit(X, y)

dump(gbr, "reg.joblib")