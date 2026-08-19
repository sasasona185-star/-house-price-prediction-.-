import json
import os
import pandas as pd
from app.schemas.prediction import PredictionInput
from app.core.config import settings

def load_valid_locations() -> list[str]:
    """Loads the list of supported locations exported during training."""
    if os.path.exists(settings.LOCATIONS_PATH):
        with open(settings.LOCATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["other"]

VALID_LOCATIONS = load_valid_locations()

def format_inr_price(price_rupees: float) -> str:
    """Formats numeric rupees into standard Indian financial notation (Crores / Lacs)."""
    if price_rupees >= 10000000.0:
        crores = price_rupees / 10000000.0
        return f"₹ {crores:.2f} Crore"
    elif price_rupees >= 100000.0:
        lacs = price_rupees / 100000.0
        return f"₹ {lacs:.2f} Lacs"
    else:
        return f"₹ {price_rupees:,.2f}"

def prepare_input_dataframe(input_data: PredictionInput) -> pd.DataFrame:
    """
    Transforms Pydantic input into a single-row pandas DataFrame
    matching the feature names and formats expected by the trained pipeline.
    """
    loc_clean = input_data.location.strip().lower()
    
    # Check if location is known, otherwise fallback to 'other'
    if loc_clean not in VALID_LOCATIONS:
        loc_clean = "other"
        
    data_dict = {
        "location": [loc_clean],
        "carpet_area_sqft": [float(input_data.carpet_area_sqft)],
        "floor_num": [int(input_data.floor_num)],
        "furnishing": [str(input_data.furnishing).strip()],
        "transaction": [str(input_data.transaction).strip()],
        "bathrooms": [int(input_data.bathrooms)],
        "balconies": [int(input_data.balconies)]
    }
    
    return pd.DataFrame(data_dict)
