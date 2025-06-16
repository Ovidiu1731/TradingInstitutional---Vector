from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
from pydantic import ConfigDict
import os

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Candlestick Market Analysis"
    version: str = "1.0.0"
    FMP_API_KEY: str

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = False
    log_level: str = "INFO"

    # Server Configuration
    max_workers: int = 4
    request_timeout: int = 30

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60

    # CORS
    allowed_origins: List[str] = ["*"]

    # GCP Configuration
    gcp_project_id: Optional[str] = None

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