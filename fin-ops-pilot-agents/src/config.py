import os
from pydantic_settings import BaseSettings
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Get the project root directory (assuming this file is in src/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings."""

    OPENAI_API_KEY: str

    # Using Path for better path manipulation
    UPLOAD_DIRECTORY: Path = PROJECT_ROOT / "static"
    CONTEXT_DIR: Path = Path(
        os.getenv("CONTEXT_DIR", r"C:\Developer\MumbaiHacks25\fin-ops-pilot-livekitServer\context")
    )


settings = Settings()