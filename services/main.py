from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import logging
from .config import get_settings
from .routers import candles

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
# candles.router removed - market analysis functionality no longer supported

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

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"status": "ok"}

@app.get("/version")
@limiter.limit("5/minute")
async def version(request: Request):
    return {"version": settings.version}

@app.get("/ready")
@limiter.limit("5/minute")
async def ready(request: Request):
    return {"ready": True} 