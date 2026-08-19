from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    location: str = Field(
        ..., 
        description="Location/locality of the property (e.g. 'thane', 'whitefield', 'andheri west')",
        example="thane"
    )
    carpet_area_sqft: float = Field(
        ..., 
        gt=100.0, 
        le=20000.0,
        description="Carpet area in square feet",
        example=1200.0
    )
    floor_num: int = Field(
        default=1,
        ge=-2,
        le=100,
        description="Floor number (0 for Ground, -1 for Basement)",
        example=3
    )
    furnishing: str = Field(
        default="Semi-Furnished",
        description="Furnishing status ('Furnished', 'Semi-Furnished', 'Unfurnished')",
        example="Semi-Furnished"
    )
    transaction: str = Field(
        default="Resale",
        description="Transaction type ('Resale', 'New Property')",
        example="Resale"
    )
    bathrooms: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of bathrooms",
        example=2
    )
    balconies: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of balconies",
        example=1
    )

class PredictionOutput(BaseModel):
    predicted_price_rupees: float = Field(..., description="Estimated raw price in Indian Rupees")
    formatted_price: str = Field(..., description="Human-friendly formatted price (e.g., 45.50 Lacs / 1.25 Cr)")
    currency: str = Field(default="INR", description="Currency symbol")
    status: str = Field(default="success", description="Prediction status")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    locations_count: int

class LocationsResponse(BaseModel):
    locations: List[str]
    total: int
