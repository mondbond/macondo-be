import os
from pathlib import Path
from src.util.logger import logger

from dotenv import load_dotenv

def get_active_env():
  return os.getenv("ACTIVE_ENV", "local")

ACTIVE_ENV = get_active_env()

def get_env_property(key: str, default: str = None) -> str:
    if ACTIVE_ENV == "local":
      env_file = Path(__file__).resolve().parent.parent.parent / f".env.{ACTIVE_ENV}"
    else:
      env_file = f".env.{ACTIVE_ENV}"

    load_dotenv(env_file)
    return os.getenv(key, default)

