from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, \
  MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from src.service.memory import SummaryChatMessageHistory

from src.llm.llm_provider import get_llm
from langchain.chains import create_retrieval_chain
from src.util.logger import logger

# THIS FILE IS DEPRECATED AND WILL BE REMOVED IN THE FUTURE

session_to_history = {}

def get_history(session_id: int):
  if session_id in session_to_history:
    return session_to_history[session_id]
  else:
    session_to_history[session_id] = InMemoryChatMessageHistory()

  return session_to_history[session_id]

def get_summary_history(session_id: int):
  if session_id in session_to_history:
    return session_to_history[session_id]
  else:
    session_to_history[session_id] = SummaryChatMessageHistory(max_messages=8)

  return session_to_history[session_id]


def chat_with_summary(query: str, session_id: int = 1):
  sysTemplate = SystemMessagePromptTemplate.from_template("You are a helpful financial assistant. Your name is Obadon. You response to only users question and do not add any additional information. If you don't know the answer, just say that you don't know. Do not try to make up an answer.")
  placeholder = MessagesPlaceholder(variable_name="history")
  userTemplate = HumanMessagePromptTemplate.from_template("{query}")
  chat_prompt = ChatPromptTemplate.from_messages([sysTemplate, placeholder, userTemplate])
  pipeline = chat_prompt | get_llm()


  chat_runnable = RunnableWithMessageHistory(
      pipeline,
      get_summary_history,
      input_messages_key="query",
      history_messages_key="history"
  )

  return chat_runnable.invoke({"query": query}, config={"session_id": session_id})


def chat(query: str, session_id: int = 1):
  sysTemplate = SystemMessagePromptTemplate.from_template("You are a helpful financial assistant. Your name is Obadon. You response to only users question and do not add any additional information. If you don't know the answer, just say that you don't know. Do not try to make up an answer.")
  placeholder = MessagesPlaceholder(variable_name="history")
  userTemplate = HumanMessagePromptTemplate.from_template("{query}")
  chat_prompt = ChatPromptTemplate.from_messages([sysTemplate, placeholder, userTemplate])
  pipeline = chat_prompt | get_llm()

  chat_runnable = RunnableWithMessageHistory(
      pipeline,
      get_history,
      input_messages_key="query",
      history_messages_key="history"
  )

  return chat_runnable.invoke({"query": query}, config={"session_id": session_id})


if __name__ == "__main__":
  response = chat_with_summary("Hello, My name is Wroclaw. Who are you?")
