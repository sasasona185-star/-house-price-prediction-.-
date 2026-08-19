import os
import joblib
import numpy as np
from app.core.config import settings
from app.utils.logging_config import logger
from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.preprocessing import prepare_input_dataframe, format_inr_price, VALID_LOCATIONS

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the trained Scikit-learn Pipeline from disk."""
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Successfully loaded model from {settings.MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found at {settings.MODEL_PATH}. Prediction requests will fail until trained.")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Please train the model first.")

        # Prepare DataFrame
        df_input = prepare_input_dataframe(input_data)
        
        # Model predicts log-transformed price
        pred_log = self.model.predict(df_input)[0]
        
        # Invert log1p transform
        pred_rupees = float(np.expm1(pred_log))
        
        # Ensure non-negative price
        pred_rupees = max(0.0, pred_rupees)
        
        formatted = format_inr_price(pred_rupees)
        
        return PredictionOutput(
            predicted_price_rupees=round(pred_rupees, 2),
            formatted_price=formatted,
            currency="INR",
            status="success"
        )

model_service = ModelService()
