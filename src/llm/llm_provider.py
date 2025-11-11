from src.util.env_property import LLM_URL, BEDROCK_MODEL, get_env_property, \
  GOOGLE_AI_MODEL, BEDROCK_MODEL_JUDGE, BEDROCK_MODEL_REASONING, \
  BEDROCK_MODEL_CHEAP, BEDROCK_MODEL_SUMMARY, LLM_SOURCE
from langchain_ollama import ChatOllama
import boto3
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from src.util.logger import logger
import time

# llm = OllamaClient(base_url="http://localhost:11434", model="mistral:instruct")
# llm = OllamaClient(base_url="http://host.docker.internal:11434", model="mistral:instruct")

# llmOldClient = OllamaClient(base_url=get_env_property("LLM_URL"), model="mistral:instruct")
  # llm = ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=0)

def local_ollama_client(temperature=0):
  return ChatOllama(base_url=LLM_URL, model="mistral:instruct", verbose=True, temperature=temperature)

def google_ai_client(temperature=0):
  time.sleep(20)
  return ChatGoogleGenerativeAI(model=GOOGLE_AI_MODEL)
  # "gemini-1.5-pro",  # or "gemini-1.5-flash")

def open_ai_model(temperature=0):
  time.sleep(5)
  return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

def bedrock_client(temperature=0, source: str = None):
  bedrock_id = None
  # bedrock_id = "us.meta.llama4-maverick-17b-instruct-v1:0"
  time.sleep(10)

  if source == 'bedrock_judge':
    logger.info(f"Using bedrock judge model id: {BEDROCK_MODEL_JUDGE}")
    bedrock_id = BEDROCK_MODEL_JUDGE
  elif source == 'bedrock_reasoning':
    logger.info(f"Using bedrock reasoning model id: {BEDROCK_MODEL_REASONING}")
    bedrock_id = BEDROCK_MODEL_REASONING
  elif source == 'bedrock_cheap':
    logger.info(f"Using bedrock cheap model id: {BEDROCK_MODEL_CHEAP}")
    bedrock_id = BEDROCK_MODEL_CHEAP
  elif source == 'bedrock_summary':
    logger.info(f"Using bedrock summary model id: {BEDROCK_MODEL_SUMMARY}")
    bedrock_id = BEDROCK_MODEL_SUMMARY
  elif source == 'bedrock_general':
    logger.info(f"Using bedrock general model {BEDROCK_MODEL}")
    bedrock_id = BEDROCK_MODEL
  else:
    bedrock_id = BEDROCK_MODEL

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


def get_llm(temperature=0, specific_source: str = LLM_SOURCE):
    # llm_source = get_env_property(specific_source, "ollama")
    llm_source = specific_source


    if "ollama" in llm_source:
        logger.info("llm_provider: using ollama client")
        return local_ollama_client(temperature)
    elif "bedrock" in llm_source:
      logger.info("llm_provider: using bedrock client")
      return bedrock_client(temperature, source=llm_source)
    elif llm_source == "google_ai":
      logger.info("llm_provider: using google ai client")
      return google_ai_client(temperature)
    elif llm_source == "open_ai":
      logger.info("llm_provider: using open ai client")
      return open_ai_model(temperature)

    return local_ollama_client()


if __name__ == "__main__":
    llm = bedrock_client()
    response = llm.invoke([HumanMessage(content="Say this is a test!")])
    print(response)
