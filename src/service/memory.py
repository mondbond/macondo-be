from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, \
  HumanMessagePromptTemplate, ChatPromptTemplate

from src.llm.llm_provider import get_llm


class SummaryChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, max_messages=4):
        super().__init__()
        self.messages: list[BaseMessage] = []
        self.max_messages = max_messages

    def add_message(self, message):
        self.messages.append(message)
        print(str(self.messages))
        if len(self.messages) > self.max_messages:
            # Summarize the oldest messages
            self.summarize_oldest_messages()

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> list[BaseMessage]:
        return self.messages

    def summarize_oldest_messages(self):
      print(str(self.messages))

      template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant that summarizes conversations. You are not continuing the conversation, just summarizing it. Summarize the following messages into a concise summary that captures the key points and context. The summary should be brief and to the point."),
        HumanMessagePromptTemplate.from_template("Messages: {messages}"),
      ])
      chain = template | get_llm()

      msg_str = "\n".join([f"{m.type}: {m.content}" for m in self.messages])
      result = chain.invoke({"messages": msg_str})

      self.clear()
      self.add_message(SystemMessage(content=result))

      print(str(self.messages))
      print("========= Summarizing messages =========")

