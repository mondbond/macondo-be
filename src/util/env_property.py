import os
from pathlib import Path

from src.util.logger import logger
from decouple import config

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

# LLM SOURCE
LLM_SOURCE=config('LLM_SOURCE', 'ollama')
LLM_SOURCE_CHEAP=config('LLM_SOURCE_CHEAP', 'ollama')
LLM_SOURCE_SUMMARY=config('LLM_SOURCE_SUMMARY', 'ollama')
LLM_SOURCE_REASONING=config('LLM_SOURCE_REASONING', 'ollama')
LLM_SOURCE_JUDGE=config('LLM_SOURCE_JUDGE', 'ollama')

#GOOGLE
GOOGLE_API_KEY = config('GOOGLE_API_KEY', None)
GOOGLE_AI_MODEL = config('GOOGLE_AI_MODEL', "gemini-1.5-flash")
# "gemini-1.5-pro", "gemini-1.5-flash"

# AWS
BEDROCK_MODEL=config('BEDROCK_MODEL', None)
BEDROCK_MODEL_REASONING=config('BEDROCK_MODEL_REASONING', None)
BEDROCK_MODEL_CHEAP=config('BEDROCK_MODEL_CHEAP', None)
BEDROCK_MODEL_SUMMARY=config('BEDROCK_MODEL_SUMMARY', None)
BEDROCK_MODEL_JUDGE=config('BEDROCK_MODEL_JUDGE', None)

# LOCAL
LLM_URL = config('LLM_URL', 'http://localhost:11434')


# 3rd PARTY API KEYS
FINNHUB_API_KEY = config('FINNHUB_API_KEY', None)
TWELVE_DATA_API_KEY = config('TWELVE_DATA_API_KEY', None)

# MCP
MCP_FIN_URL=config('MCP_FIN_URL', 'http://localhost:8887/mcp')

# LANGSMITH CONFIGURATION
LANGSMITH_TRACING=config('LANGSMITH_TRACING', True)
LANGSMITH_ENDPOINT=config('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
LANGSMITH_API_KEY=config('LANGSMITH_API_KEY', None)
LANGSMITH_PROJECT=config('LANGSMITH_PROJECT', 'default')

# TRULENS CONFIGURATION
TRULENS_MAX_WORKERS=config('TRULENS_MAX_WORKERS', 1)

logger.info(f"Active environment: {ACTIVE_ENV}")
logger.info(f"LLM_URL: {LLM_URL}")
logger.info(f"LLM_SOURCE: {LLM_SOURCE}")
logger.info(f"LLM_SOURCE_CHEAP: {LLM_SOURCE_CHEAP}")
logger.info(f"LLM_SOURCE_REASONING: {LLM_SOURCE_REASONING}")
logger.info(f"LLM_SOURCE_JUDGE: {LLM_SOURCE_JUDGE}")
logger.info(f"BEDROCK_MODEL: {BEDROCK_MODEL}")
logger.info(f"FINNHUB_API_KEY: {'enabled' if FINNHUB_API_KEY else 'disabled'}")
logger.info(f"TWELVE_DATA_API_KEY: {'enabled' if TWELVE_DATA_API_KEY else 'disabled'}")
logger.info(f"MCP_FIN_URL: {MCP_FIN_URL}")
logger.info(f"LANGSMITH_TRACING: {LANGSMITH_TRACING}")
logger.info(f"LANGSMITH_ENDPOINT: {LANGSMITH_ENDPOINT}")
logger.info(f"LANGSMITH_API_KEY: {'enabled' if LANGSMITH_API_KEY else 'disabled'}")
logger.info(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")
logger.info(f"TRULENS_MAX_WORKERS: {TRULENS_MAX_WORKERS}")
