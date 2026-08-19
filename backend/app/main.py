from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.prediction import router as prediction_router
from app.services.inference import model_service
from app.utils.logging_config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure model is loaded into memory
    logger.info("Initializing House Price Prediction API...")
    if not model_service.is_ready():
        model_service.load_model()
    yield
    # Shutdown
    logger.info("Shutting down House Price Prediction API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI Backend for Real Estate House Price Prediction (ITI AI Track Project)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(prediction_router, prefix=settings.API_V1_STR)
# Also include at root for convenient testing
app.include_router(prediction_router, prefix="")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to House Price Prediction API",
        "docs_url": "/docs",
        "health_check": "/health",
        "api_v1": settings.API_V1_STR
    }
