from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """Transforms raw market data into engineered features for machine learning models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Processes input DataFrame to create engineered features."""
        # Calculate moving averages
        ma_7 = df['price'].rolling(7).mean()
        ma_30 = df['price'].rolling(30).mean()
        
        # Compute price trends
        df['price_diff'] = df['price'].diff()
        trend_sma_5 = (df['price_diff'].rolling(5).mean() > 0).astype(int)
        
        # Calculate demand elasticity
        df['demand_elasticity'] = df['quantity'].rolling(7).std() / df['quantity'].rolling(7).mean()
        
        features = {
            'moving_avg_7': ma_7.values,
            'moving_avg_30': ma_30.values,
            'trend_sma_5': trend_sma_5.values,
            'demand_elasticity': df['demand_elasticity'].values
        }
        
        # Normalize features
        feature_matrix = self._normalize_features(features)
        
        return {
            'features': feature_matrix,
            'targets': df[['price', 'quantity']].values
        }
    
    def _normalize_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Normalizes feature values to improve model performance."""
        df = pd.DataFrame(features)
        df_scaled = self.scaler.fit_transform(df)
        return df_scaled