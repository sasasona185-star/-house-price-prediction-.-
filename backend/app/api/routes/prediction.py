from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import (
    PredictionInput, 
    PredictionOutput, 
    HealthResponse, 
    LocationsResponse
)
from app.services.inference import model_service
from app.services.preprocessing import VALID_LOCATIONS
from app.core.config import settings
from app.utils.logging_config import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    is_loaded = model_service.is_ready()
    return HealthResponse(
        status="healthy" if is_loaded else "degraded (model not loaded)",
        model_loaded=is_loaded,
        version=settings.VERSION,
        locations_count=len(VALID_LOCATIONS)
    )

@router.get("/locations", response_model=LocationsResponse, tags=["Metadata"])
async def get_locations():
    """Returns list of supported high-frequency locations for frontend dropdowns."""
    return LocationsResponse(
        locations=VALID_LOCATIONS,
        total=len(VALID_LOCATIONS)
    )

@router.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_house_price(payload: PredictionInput):
    """
    Predicts the price of a house given features like location, area, floor, furnishing, etc.
    """
    try:
        prediction = model_service.predict(payload)
        logger.info(f"Prediction generated successfully: {prediction.formatted_price}")
        return prediction
    except RuntimeError as re:
        logger.error(f"Runtime error during prediction: {re}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction. Please check input parameters."
        )
