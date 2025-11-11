from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from src.service.graph.fall_explanation_graph import \
  run_company_fall_explanation_graph
from src.service.graph.intention_service import classify_intent_with_prompt
from src.service.graph.news_search_reflection_summary_graph import \
  run_news_graph
from src.usecase.image_uc import search_image_embeddings_link
from src.util.logger import logger
from src.service.react.react_agent import call_agent
from langgraph.types import Command
from src.llm.llm_provider import get_llm
from src.models.router import UserIntentionEnum
from src.models.summarised_chat_history_memory import SummaryChatHistory
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report
from src.util.prompt_manager import prompt_manager

class RouterState(TypedDict, total=False):
  user_message: str
  bot_message: str
  history: list
  data: SummaryChatHistory
  ticker: list[str]
  intent: UserIntentionEnum
  image_wanted: bool
  action: str

class MemoryNode(SummaryChatHistory):

  def __call__(self, *args, **kwargs):
    state = args[0]
    if 'bot_message' in state:
      self.add_message("user1", "assistant", state["bot_message"])
      return Command(goto=END)
    else:
      self.add_message("user1", "user", state["user_message"])
      state['history'] = self.get_history("user1")
      return Command(update=state, goto="classify_intent")

def classify_intent(state: RouterState) -> dict:
  user_intention = classify_intent_with_prompt(state)

  logger.info("ROUTER OUTPUT: " + str(user_intention))

  ticker = user_intention.ticker
  state['ticker'] = ticker
  state['intent'] = user_intention.intention
  state['image_wanted'] = user_intention.image_wanted
  return state

async def run_intent(state: RouterState) -> dict:
  intent = state["intent"]

  if state['image_wanted']:
    link = search_image_embeddings_link(state['user_message'])
    state['bot_message'] += f"\nI found this image that might help: {link}"
    return state

  if intent == UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT and state["ticker"]:
    answer = run_subquery_search_in_report(state['ticker'][0], state['user_message'])
    state['bot_message'] = answer
  elif intent == UserIntentionEnum.NEWS_ABOUT_COMPANY and state['ticker']:
    answer = run_news_graph(state['ticker'][0], state['user_message'])
    state['bot_message'] = answer
  elif intent == UserIntentionEnum.ANALYSE_SHARE_PRISE and state["ticker"]:
    answer = run_company_fall_explanation_graph(state["ticker"])
    state['bot_message'] = answer
  else:
    rs = await call_agent(state['user_message'], state['history'])
    state['bot_message'] =rs['output']

  return state

def answer_node(state: RouterState) -> dict:
  logger.info(f"Final state: {state}")
  return state

# GRAPH DEFINITION
memory_node = MemoryNode(get_llm(), prompt_manager.get_prompt('trim_chat_history'), window_character_size=5000, summary_trim_coeficient=0.2)

graph = StateGraph(RouterState)

graph.add_node("update_history", memory_node)
graph.add_node("classify_intent", classify_intent)
graph.add_node("run_intent", run_intent)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "update_history")
graph.add_edge("classify_intent", "run_intent")
graph.add_edge("run_intent", "answer_node")
graph.add_edge("answer_node", "update_history")

route_app = graph.compile()
