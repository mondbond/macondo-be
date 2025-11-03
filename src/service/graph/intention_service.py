from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_provider import get_llm
from src.models.router import RouterDto
from src.util.prompt_manager import prompt_manager

def classify_intent_with_prompt(state):
  router_chain = get_llm().with_structured_output(RouterDto)
  route_prompt = prompt_manager.get_prompt("classify_intent")

  user_message = ""
  if state["history"]:
    user_message = "Previous conversation history: " + str(
      state["history"]).replace("{", "").replace("}",
                                               "") + " \n New message: "

  user_message += state['user_message']

  prompt = ChatPromptTemplate.from_messages([
    ("system", route_prompt),
    ("user", user_message)
  ])
  chain = prompt | router_chain
  user_intention = chain.invoke({})
  return user_intention
