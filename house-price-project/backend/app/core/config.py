import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "House Price Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Paths to exported ML artifacts
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH: str = os.path.join(os.path.dirname(BASE_DIR), "models", "house_price.pkl")
    LOCATIONS_PATH: str = os.path.join(os.path.dirname(BASE_DIR), "models", "locations.json")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
