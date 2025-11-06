from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.memory import InMemoryStore
from src.util.logger import logger

from src.llm.llm_provider import get_llm
from src.util.prompt_manager import prompt_manager

class SummaryChatHistory():
  def __init__(self, llm, summarize_prompt, summary_trim_coeficient = 0.5, window_character_size=2000):
    super().__init__()
    self.memory = InMemoryStore()
    self.window_size = window_character_size
    self.coeficient = summary_trim_coeficient
    self.llm = llm
    self.summarizer_prompt = summarize_prompt

  def add_message(self, user_id, role, message):
    namespace = (user_id, "history")
    history = self.memory.get(namespace, "chat")

    if history is None:
      history = []
      history.append((role, message))
      self.memory.put(namespace, "chat", history)
    else:
      history = history.value
      history.append((role, message))
      self.memory.put(namespace, "chat", history)

    if self.is_history_too_long(history):
      summarized_history = self.summarise_history(history)
      self.memory.put(namespace, "chat", summarized_history)

    logger.info(f"HISTORY IS: {history}")

  def is_history_too_long(self, history):
    logger.info(f"IS HISTORY TO LONG?: {history}")
    total_length = sum(len(msg) for role, msg in history if msg is not None)
    return total_length > self.window_size

  def sanitize_msg(self, msg: str) -> str:
    return msg.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ")

  def summarise_history(self, history):
    if not self.llm or not self.summarizer_prompt:
      return history

    summary_prompt = ChatPromptTemplate.from_messages(
      [
        ("system", self.summarizer_prompt),
        ("user", "\n".join([f"{role}: {self.sanitize_msg(msg)}" for role, msg in history]))
      ]
    )

    summary_chain = summary_prompt | self.llm

    summary = summary_chain.invoke({'character_limit': self.coeficient * self.window_size}).content
    return [("system", summary)]


  def get_history(self, user_id):
    namespace = (user_id, "history")
    history = self.memory.get(namespace, "chat")

    if history is None:
        history = []

    return history.value

summary_memory = SummaryChatHistory(get_llm(), prompt_manager.get_prompt('trim_chat_history'), window_character_size=200)


if __name__ == "__main__":
  history = SummaryChatHistory(get_llm(), prompt_manager.get_prompt('trim_chat_history'), window_character_size=200)
  history.add_message("user1", "user", "Hello, how are you?")
  history.add_message("user1", "system", "I'm fine and you?")
  history.add_message("user1", "user", "Have you heard about the latest news in tech?")
  history.add_message("user1", "system", "Nah, not really. What is it about?. What I really know is that Apple has released a new iPhone model. And Google is working on new AI features. And Microsoft is expanding its cloud services. and Amazon is investing in renewable energy. And Tesla is launching a new electric car. And Facebook is rebranding to Meta. And Twitter is introducing new subscription plans. And Netflix is producing more original content. And Spotify is acquiring podcast companies. and Uber is testing flying cars. And Airbnb is going public. And Zoom is adding new security features. And Slack is integrating with more apps. And Salesforce is acquiring more startups. And Adobe is launching new creative tools. And Intel is developing new processors. And AMD is competing with Nvidia. And IBM is focusing on hybrid cloud. And Oracle is enhancing its database services. And Cisco is expanding its networking solutions. And HP is releasing new laptops. And Dell is improving its servers. And Lenovo is entering the smartphone market.")
  history.add_message("user1", "user", "wow you know so much!")
  history.add_message("user1", "system", "I know so much!")
