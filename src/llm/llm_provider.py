from src.llm.ollama_client import OllamaClient
from src.util.env_property import get_env_property
from langchain_ollama import ChatOllama

# llm = OllamaClient(base_url="http://localhost:11434", model="mistral:instruct")
# llm = OllamaClient(base_url="http://host.docker.internal:11434", model="mistral:instruct")

# llmOldClient = OllamaClient(base_url=get_env_property("LLM_URL"), model="mistral:instruct")
  # llm = ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=0)


def get_llm(temperature=0):
    # todo: make model configurable
    return ChatOllama(base_url=get_env_property("LLM_URL"), model="mistral:instruct", verbose=True, temperature=temperature)

