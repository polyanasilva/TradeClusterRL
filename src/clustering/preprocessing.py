import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CandlePreprocessor:
    def __init__(self, config=None):
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self, path):
        df = pd.read_csv(path)
        return df.dropna()

    def organize_datetime(self, df):
        return df[::-1]

    def generate_features(self, df):
        
        # Absolute values
        df = df.copy()
        df['body'] = df['Close'] - df['Open']
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['total_length'] = (df['High'] - df['Low']).replace(0, 1e-08) #avoid division by zero

        # Ratios
        df['body_ratio'] = df['body'] / df['total_length']
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_length']
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_length']

        # Add logaritmic size scale to help clustering differentiate sizes
        df['range'] = np.log(df['total_length'])

        features = df[['range', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio']]
        features = self.scaler.fit_transform(features)

        return features