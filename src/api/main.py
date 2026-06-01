"""
FastAPI Serving Entry Point
===========================
Configures the NIDS serving API, lifespan loaders, endpoints, logging,
and global exception mapping.
"""

import time
import uvicorn
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from src.utils.logger import setup_logger
from src.api.config import (
    PREPROCESSOR_PATH,
    MODEL_PATH,
    HOST,
    PORT,
    APP_NAME,
    VERSION,
    DESCRIPTION,
)
from src.api.schemas import (
    ConnectionRecord,
    PredictionResponse,
    BatchPredictionResponse,
)
from src.api.predictor import NIDSPredictionEngine

logger = setup_logger("api_main")

# Global serving predictor
predictor: NIDSPredictionEngine = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup model caching and shutdown release hooks."""
    global predictor
    logger.info("Initializing REST API Serving Lifespan...")
    try:
        predictor = NIDSPredictionEngine(
            preprocessor_path=PREPROCESSOR_PATH,
            model_path=MODEL_PATH
        )
        logger.info("Model assets successfully loaded in memory. Service ready.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize serving predictor: {e}")
        raise e
    yield
    logger.info("Shutting down NIDS Serving Lifespan.")


app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description=DESCRIPTION,
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_performance_telemetry(request: Request, call_next):
    """Logs incoming request endpoints and tracks routing latency."""
    t_start = time.perf_counter()
    response = await call_next(request)
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Latency: {t_elapsed:.2f} ms"
    )
    return response

# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Maps value errors (schema deviations) to 400 Bad Request."""
    logger.error(f"Value validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Schema validation error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Maps uncaught internal exceptions to 500 Internal Server Error."""
    logger.critical(f"Unhandled system error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal system error occurred."}
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", status_code=status.HTTP_200_OK)
async def read_root():
    """Returns application name, version metadata, and desc."""
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "description": DESCRIPTION,
        "status": "online"
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Confirms serving health and verifies model parameters are loaded."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine not initialized."
        )
    return {
        "status": "healthy",
        "models_loaded": True,
        "preprocessor_path": PREPROCESSOR_PATH,
        "model_path": MODEL_PATH
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single network connection"
)
async def predict_single(record: ConnectionRecord):
    """
    Exposes real-time threat evaluation for a single network log.
    Validates features using Pydantic, applies robust preprocessing,
    and classifies Zero-Day hybrid anomalies.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine not initialized."
        )
    
    # Map record to dict and execute
    payload = record.model_dump()
    predictions = predictor.predict_records([payload])
    
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction returned empty result."
        )
        
    return predictions[0]


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a batch of network connections"
)
async def predict_batch(records: List[ConnectionRecord]):
    """
    Exposes high-speed real-time threat evaluation for multiple network logs.
    Takes a batch array list of connection telemetry records.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine not initialized."
        )
        
    if not records:
        return {"predictions": [], "count": 0}
        
    payloads = [r.model_dump() for r in records]
    predictions = predictor.predict_records(payloads)
    
    return {
        "predictions": predictions,
        "count": len(predictions)
    }


if __name__ == "__main__":
    logger.info(f"Starting uvicorn server on {HOST}:{PORT}...")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
