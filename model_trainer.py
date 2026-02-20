import logging
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and validates machine learning models for pricing prediction."""
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor()
        }
        
    def train_models(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Trains and validates models using the provided data."""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Validate
            score = self._validate_model(model, X_test, y_test)
            results[name] = {'model': model, 'score': score}
            
        logger.info("Model training completed.")
        return results
    
    def _validate_model