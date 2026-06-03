import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
TMP_DIR = BASE_DIR / "tmp"
LOGS_DIR = BASE_DIR / "logs"

TMP_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 2048))
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", 7200))

CHUNK_DURATION_SEC = int(os.getenv("CHUNK_DURATION_SEC", 300))
CHUNK_OVERLAP_SEC = int(os.getenv("CHUNK_OVERLAP_SEC", 10))

OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}
