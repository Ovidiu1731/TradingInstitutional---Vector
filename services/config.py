from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
from pydantic import ConfigDict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Candlestick Market Analysis"
    version: str = "1.0.0"
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Server Configuration
    max_workers: int = 4
    request_timeout: int = 30

    # Rate Limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_period: str = os.getenv("RATE_LIMIT_PERIOD", "1minute")

    # CORS
    allowed_origins: List[str] = [
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "https://yourdomain.com"  # Add your production domain
    ]

    # GCP Configuration
    gcp_project_id: Optional[str] = None

    # Market Analysis Settings
    MIN_CANDLES_FOR_ANALYSIS: int = 20
    TREND_LOOKBACK_PERIOD: int = 20
    LIQUIDITY_ZONE_THRESHOLD: float = 0.0001
    MIN_TOUCHES_FOR_ZONE: int = 3

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )

    def __init__(self, **values):
        super().__init__(**values)
        # Optionally adjust model_config based on environment
        if self.environment == "production":
            self.model_config = ConfigDict(
                case_sensitive=True,
                extra="allow"
            )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 