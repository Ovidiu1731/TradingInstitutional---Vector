from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import logging

from services.config import get_settings
from services.market_data import MarketDataService
from services.market_analysis import MarketAnalysisService
from routers import candles

# Initialize settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests}/{settings.rate_limit_period}"]
)

# Initialize FastAPI app
app = FastAPI(
    title="Market Analysis API",
    description="API for analyzing market structure and candlestick patterns",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candles.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    app.state.market_data_service = MarketDataService()
    app.state.market_analysis_service = MarketAnalysisService()
    logger.info("Market analysis services initialized")

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": settings.version,
        "services": {
            "market_data": "initialized" if hasattr(app.state, "market_data_service") else "not_initialized",
            "market_analysis": "initialized" if hasattr(app.state, "market_analysis_service") else "not_initialized"
        }
    }

@app.get("/version")
@limiter.limit("5/minute")
async def version(request: Request):
    """Version endpoint"""
    return {"version": settings.version}

@app.get("/ready")
@limiter.limit("5/minute")
async def ready(request: Request):
    """Readiness check endpoint"""
    return {
        "ready": True,
        "services": {
            "market_data": "ready" if hasattr(app.state, "market_data_service") else "not_ready",
            "market_analysis": "ready" if hasattr(app.state, "market_analysis_service") else "not_ready"
        }
    } 