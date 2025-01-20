from pydantic import BaseSettings

class Settings(BaseSettings):
    api_url: str = "http://localhost:8000"
    batch_size: int = 10
    processing_delay: float = 1.0  # seconds between batches
    
    class Config:
        env_file = ".env"

settings = Settings() 