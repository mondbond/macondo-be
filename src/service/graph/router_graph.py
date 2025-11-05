from langchain_core.prompts import ChatPromptTemplate, \
  SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from src.service.graph.fall_explanation_graph import \
  run_company_fall_explanation_graph
from src.service.graph.intention_service import classify_intent_with_prompt
from src.service.graph.news_search_reflection_summary_graph import \
  run_news_graph
from src.usecase.image_uc import search_image_embeddings_link
from src.util.logger import logger

from langgraph.types import Command
from src.llm.llm_provider import get_llm
from src.models.router import RouterDto, UserIntentionEnum
from src.models.summarised_chat_history_memory import SummaryChatHistory, \
  summary_memory
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report
from src.tools.tools import wikipedia_info, final_result, search_company_news
from src.util.env_property import get_env_property
from src.util.prompt_manager import prompt_manager
from src.tools.tools import wikipedia_info
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

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


async def call_agent(query: str, history) -> str:
  tools = [wikipedia_info]
  llm = get_llm()

  mcp_url = get_env_property("MCP_FIN_URL")
  logger.info(f"MCP_FIN_URL: {mcp_url}")

  # http://mond_mcp:8887/mcp
  if mcp_url is not None and mcp_url != "":
    logger.info("MCP_FIN_URL:", mcp_url)
    mcp_client = MultiServerMCPClient(
    {

      "mond_mcp": {
        "transport": "streamable_http",
        "url": mcp_url,
      }
    }
    )
    toolss = await mcp_client.get_tools()
    tools.extend(toolss)
  else:
    logger.warning("No MCP_FIN_URL provided, skipping MCP tools loading.")

  # agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
  # answer = await agent.ainvoke(query)

  prompt = ChatPromptTemplate.from_template(prompt_manager.get_prompt("react_agent"))
  agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

  logger.info(f"History provided to agent: {history}")
  answer = await agent_executor.ainvoke({"input": query, "history": history})


  logger.info(answer)
  return answer

async def run_intent(state: RouterState) -> dict:
  intent = state["intent"]

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
    rs = await call_agent(state['user_message'], str(state['history']))
    state['bot_message'] =rs['output']

  if state['image_wanted']:
    link = search_image_embeddings_link(state['user_message'])
    state['bot_message'] += f"\nI found this image that might help: {link}"

  return state


def answer_node(state: RouterState) -> dict:
  # action = state["action"]
  # message = state["user_message"]

  # if action == "keep_conversation":
  #   return state
  # elif action == "answer_question":
  #   answer = f"That’s a great question about: '{message}'. Let me think..."
  # elif action == "execute_command":
  #   answer = f"Okay, executing your request: '{message}'."
  # else:
  #   answer = "I'm not sure what you mean. Could you rephrase?"

  return state


memory_node = MemoryNode(get_llm(), prompt_manager.get_prompt('trim_chat_history'), window_character_size=5000, summary_trim_coeficient=0.2)
# 🕸️ Build the router graph
graph = StateGraph(RouterState)

# Add functional nodes
graph.add_node("update_history", memory_node)
graph.add_node("classify_intent", classify_intent)
graph.add_node("run_intent", run_intent)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "update_history")
# graph.add_edge("update_history", "classify_intent")
graph.add_edge("classify_intent", "run_intent")
graph.add_edge("run_intent", "answer_node")
graph.add_edge("answer_node", "update_history")
# graph.add_edge("update_history", END)

route_app = graph.compile()

if __name__ == "__main__":
  import asyncio

  async def main():
    state = RouterState(
        user_message="What is the Ukraine?",
    )
    result = await route_app.ainvoke(state)
    logger.info("🤖 Bot:", result["bot_message"])

    # state = RouterState(
    #     user_message="Do you remember my name?",
    # )
    # result = await route_app.ainvoke(state)
    # logger.info("🤖 Bot:", result["bot_message"])

  asyncio.run(main())
