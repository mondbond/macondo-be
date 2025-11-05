from src.util.env_property import get_env_property
from langchain_ollama import ChatOllama
import boto3
from langchain_aws import ChatBedrockConverse
from src.util.logger import logger
import time

# llm = OllamaClient(base_url="http://localhost:11434", model="mistral:instruct")
# llm = OllamaClient(base_url="http://host.docker.internal:11434", model="mistral:instruct")

# llmOldClient = OllamaClient(base_url=get_env_property("LLM_URL"), model="mistral:instruct")
  # llm = ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=0)

def local_ollama_client(temperature=0):
  return ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=temperature)

def bedrock_client(temperature=0):
  bedrock_id = get_env_property("BEDROCK_MODEL", None)
  time.sleep(5)

  # Optional: specify a profile name
  session = boto3.Session()
  # session = boto3.Session(profile_name="default", region_name="us-east-1")
  credentials = session.get_credentials().get_frozen_credentials()

  aws_access_key_id = credentials.access_key
  aws_secret_access_key = credentials.secret_key
  aws_session_token = credentials.token  # can be None if not using

  # ID_BEDROCK = "anthropic.claude-3-sonnet-20240229-v1:0"

  return ChatBedrockConverse(
      model_id=bedrock_id,
      region_name="us-east-1",
      aws_access_key_id=aws_access_key_id,
      aws_secret_access_key=aws_secret_access_key,
      aws_session_token=aws_session_token,  # optional
      temperature=temperature
  )


def get_llm(temperature=0, specific_source: str = "LLM_SOURCE", source=None):
    llm_source = get_env_property(specific_source, "ollama")

    if source is not None:
      llm_source = source

    if llm_source == "ollama":
        logger.info("llm_provider: using ollama client")
        return local_ollama_client(temperature)
    elif llm_source == "bedrock":
      logger.info("llm_provider: using bedrock client")
      return bedrock_client(temperature)

    return local_ollama_client()
