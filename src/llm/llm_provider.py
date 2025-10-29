from src.llm.ollama_client import OllamaClient
from src.util.env_property import get_env_property
from langchain_ollama import ChatOllama

# llm = OllamaClient(base_url="http://localhost:11434", model="mistral:instruct")
# llm = OllamaClient(base_url="http://host.docker.internal:11434", model="mistral:instruct")

# llmOldClient = OllamaClient(base_url=get_env_property("LLM_URL"), model="mistral:instruct")
  # llm = ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=0)


def local_ollama_client(temperature=0):
  return ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=temperature)

def bedrock_client(temperature=0):
  import boto3
  from langchain_aws import ChatBedrockConverse

  bedrock_id = get_env_property("BEDROCK_MODEL", None)

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


def get_llm(temperature=0):
    llm_source = get_env_property("LLM_SOURCE", "ollama")

    if llm_source == "ollama":
        return local_ollama_client(temperature)
    elif llm_source == "bedrock":
        return bedrock_client(temperature)

    # todo: make model configurable
    return local_ollama_client()
    # return  bedrock_client()

