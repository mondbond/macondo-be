from langchain_core.prompts import ChatPromptTemplate

from src.models.router import UserIntentionEnum, RouterDto
from src.llm.llm_provider import get_llm
from src.service.graph.fall_explanation_graph import \
  run_company_fall_explanation_graph
from src.service.graph.news_search_reflection_summary_graph import \
  run_news_graph
from src.service.graph.report_search_reflection_graph import runt_report_search
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report
from src.service.graph.router_graph import route_app, RouterState


def route_user_intention(query: str) -> RouterDto:

  router_chain = get_llm().with_structured_output(RouterDto)
  route_prompt = """
  You are an information extraction system for financial queries. 
Your job is to STRICTLY extract structured fields from the user’s query. 
Do not provide explanations, guesses, or commentary — only extract values according to the schema.

Schema fields:
- ticker: list[str]
  • Stock ticker symbols (short codes like AAPL, GOOGL, MSFT).
  • Extract only if the ticker is explicitly mentioned.
  • If multiple tickers are present, return them as a list.
  • If none are explicitly given, return an empty list. Do NOT infer from company names.

- intention: UserIntentionEnum
  • Classify the user’s high-level intention:
    - NOT_RELATED → The query is unrelated to stocks, tickers, or finance.
    - COMPANY_INFORMATION_FROM_REPORT → The user wants specific financial/report data about a company (requires ticker).
    - ANALYSE_SHARE_PRISE → The user wants reasons or explanations about recent share price movements.

Rules:
- Always prefer precision over recall (better to miss than to guess).
- Never infer a ticker from a company name unless it is explicitly written as a ticker symbol.
- Output must always match the schema exactly.
  
  """


  prompt = ChatPromptTemplate.from_messages([("system", route_prompt), ("user", query)])
  chain = prompt | router_chain
  user_intention = chain.invoke({"query": query})

  print("ROUTER OUTPUT: " + str(user_intention))

  ticker = user_intention.ticker
  if ticker == "NOT_SET":
    user_intention.intention = UserIntentionEnum.NOT_RELATED

  return user_intention


def start_graph(query):
  user_intention = route_user_intention(query)

  print("USER INTENTION: " + str(user_intention))

  if user_intention.ticker is None:
   return "TICKER NOT SET"

  if user_intention.intention == UserIntentionEnum.NEWS_ABOUT_COMPANY:
    return run_news_graph(user_intention.ticker, query)

  if user_intention.intention == UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT:
    return runt_report_search(user_intention.ticker, query)

def start_graph_v1(query: str):
  user_intention = route_user_intention(query)

  if user_intention.ticker is None:
   return "TICKER NOT SET"

  if user_intention.intention == UserIntentionEnum.ANALYSE_SHARE_PRISE:
    return run_company_fall_explanation_graph(user_intention.ticker)

  if user_intention.intention == UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT:
    return run_subquery_search_in_report(user_intention.ticker[0], query)

  return "Can not identify the query intention"


async def start_graph_v2(query: str):
  state = RouterState(
      user_message = query,
  )
  result = await route_app.ainvoke(state)

  return result['bot_message']

if __name__ == "__main__":
  query = "ticker is AAPL,UBER,TSLA. Explain the latest share price changes from the news"
  print(route_user_intention(query))
