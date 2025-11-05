from pathlib import Path
import os
from dotenv import load_dotenv

# Load local .env if present
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Gateway configuration
DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "https://ellm.nrp-nautilus.io/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3")
PORT = int(os.getenv("PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
