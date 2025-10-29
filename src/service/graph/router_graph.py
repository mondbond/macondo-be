from langchain_core.prompts import ChatPromptTemplate, \
  SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

import asyncio

import anyio

from langgraph.types import Command

from src.adapters.mcp_tools import get_mcp_config
from src.llm.llm_provider import get_llm
from src.models.router import RouterDto, UserIntentionEnum
from src.models.summarised_chat_history_memory import SummaryChatHistory, \
  summary_memory
from src.service.agent_executor_service import CustomAgentExecutor
from src.service.file_format_service import soup_html_to_text
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report
from src.tools.tools import wikipedia_info, final_result
from src.usecase.report_uc import save_text_report
from src.util.env_property import get_env_property
from src.util.prompt_manager import prompt_manager
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.tools.tools import wikipedia_info
from langchain.agents import initialize_agent
from langchain.agents import Tool, AgentType
from langchain_mcp_adapters.client import MultiServerMCPClient


class RouterState(TypedDict, total=False):
  user_message: str
  bot_message: str
  history: list
  data: SummaryChatHistory
  ticker: list[str]
  intent: UserIntentionEnum
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


#
# def history_node(state: RouterState) -> dict:
#   if state['bot_message']:
#     state['data'].add_message("bot1", "assistant", state["bot_message"])
#   else:
#     state['data'].add_message("user1", "user", state["user_message"])
#
#   return state

def classify_intent(state: RouterState) -> dict:

  router_chain = get_llm().with_structured_output(RouterDto)
  route_prompt = prompt_manager.get_prompt("classify_intent")

  prompt = ChatPromptTemplate.from_messages([
    ("system", route_prompt),
    ("user", "History is: " + str(state["history"]).replace("{","").replace("}", "") + " \n New message: " + state['user_message'])
  ])
  chain = prompt | router_chain
  user_intention = chain.invoke({})

  print("ROUTER OUTPUT: " + str(user_intention))

  ticker = user_intention.ticker
  state['ticker'] = ticker
  state['intent'] = user_intention.intention
  return state
#
# def run_intent(state: RouterState) -> dict:
#   intent = state["intent"]
#
#   if intent == UserIntentionEnum.NOT_RELATED:
#     action = "keep_conversation"
#     state['action'] = action
#
#     llm = get_llm()
#     agent = CustomAgentExecutor(llm, ChatPromptTemplate.from_messages([
#
#       SystemMessagePromptTemplate.from_template(
#          prompt_manager.get_prompt("not_related_topic_agent")),
#       HumanMessagePromptTemplate.from_template("Here is the user question you need to call final_result tool in tool_calls property: {input}. ")
#     ]), [], final_result,  max_iteration=2)
#
#     answer = agent.invoke({"input": state['user_message'], "agent_scratchpad": [], "history": state["history"]})
#     state['bot_message'] = answer
#   elif intent == UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT and state["ticker"]:
#     answer = run_subquery_search_in_report(state['ticker'][0], state['user_message'])
#     state['bot_message'] = answer
#     action = "keep_conversation"
#     state['action'] = action
#   elif intent == "command":
#     action = "execute_command"
#   else:
#     action = "fallback"
#
#   return state

async def call_this(query: str) -> str:
  tools = []
  llm = get_llm()

  mcp_url = get_env_property("MCP_FIN_URL")
  print("MCP_FIN_URL:", mcp_url)
  mcp_client = MultiServerMCPClient(
      # {
      #
      #   "mond_mcp": {
      #     "transport": "streamable_http",
      #     "url": "http://mond_mcp:8887/mcp",
      #   }
      # }

  {

    "mond_mcp": {
      "transport": "streamable_http",
      "url": mcp_url,
    }
  }
  )
  toolss = await mcp_client.get_tools()

  tools.extend(toolss)

  agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

  answer = await agent.ainvoke(query)
  print(answer)
  return answer


async def run_intent(state: RouterState) -> dict:
  intent = state["intent"]

  if intent == UserIntentionEnum.NOT_RELATED:
    action = "keep_conversation"
    state['action'] = action

    # tools = [wikipedia_info]
    tools = []
    llm = get_llm()

    # with MultiServerMCPClient(get_mcp_config()) as mcp_client:
    #   mcp_tools = mcp_client.get_all_tools()


    rs = await call_this(state['user_message'])
    state['bot_message'] =rs
  elif intent == UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT and state["ticker"]:
    answer = run_subquery_search_in_report(state['ticker'][0], state['user_message'])
    state['bot_message'] = answer
    action = "keep_conversation"
    state['action'] = action
  elif intent == "command":
    action = "execute_command"
  else:
    action = "fallback"

  return state


def answer_node(state: RouterState) -> dict:
  action = state["action"]
  message = state["user_message"]

  if action == "keep_conversation":
    return state
  elif action == "answer_question":
    answer = f"That’s a great question about: '{message}'. Let me think..."
  elif action == "execute_command":
    answer = f"Okay, executing your request: '{message}'."
  else:
    answer = "I'm not sure what you mean. Could you rephrase?"

  return {"bot_message": answer}


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
#
# state = RouterState(
#     user_message = "Hello, My name is Ivan.",
# )
# result = route_app.invoke(state)
# state = RouterState(
#     user_message = "Do you remember my name?",
# )
# result = route_app.invoke(state)



#
# with open("/Users/ibahr/Desktop/reports/UBER.html", "rb") as f:
#   pdf_content = f.read()
#   report = soup_html_to_text(pdf_content)
#
#   metadata = {
#     "ticker": "UBER",
#     "date": "2025-07-17"
#   }
#
# # save_report(pdf_content, metadata)
# save_text_report(report, metadata)
#
#
# with open("/Users/ibahr/Desktop/reports/AAPL.html", "rb") as f:
#   pdf_content = f.read()
#   report = soup_html_to_text(pdf_content)
#
#   metadata = {
#     "ticker": "AAPL",
#     "date": "2025-07-17"
#   }
#
# # save_report(pdf_content, metadata)
# save_text_report(report, metadata)
#
#
# state = RouterState(
#     user_message = "Hello, Look at the Apple company financial report and tell me who is their main competitor? ticker is AAPL",
# )
# result = app.invoke(state)
# state = RouterState(
#     user_message = "Now looking on financial report tell me what is the main source of revenue?",
# )
# result = app.invoke(state)
#
#
#
#
# print(memory_node.get_history("user1"))
# print("🤖 Bot:", result["bot_message"])

# async def ss(clin):
#   tools = await clin.get_tools()
#   for tool in tools:
#     print(f"- {tool.name}: {tool.description}")


# if __name__ == "__main__":
#   import asyncio
#   mcp_client = MultiServerMCPClient(
#       {
#
#          "mond_mcp": {
#             "transport": "streamable_http",
#             "url": "http://0.0.0.0:8887/mcp",
#           }
#       }
#   )
#   asyncio.run(ss(mcp_client))
