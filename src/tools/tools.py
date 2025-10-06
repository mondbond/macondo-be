from langchain_core.tools import tool
from pydantic import Field, BaseModel

from src.service.query_report_service import report_rephrase_retriever_search, base_query_report_question_answer
from src.service.thirdparty.news.finhub_news_service import fetch_company_news
import wikipediaapi

class WikipediaSearchByNoun(BaseModel):
  noun_to_search: str = Field(
      ...,
      description="Some noun related to anything you want to get general information for. Usually it is a name of a person, place, event, object, concept, etc. Cannot be null.")
@tool(args_schema=WikipediaSearchByNoun)
def wikipedia_info(noun_to_search: str) -> str:
  """Always call this tool if you need to know something to before answer to user. Fetch general summary information from Wikipedia for a given noun. Some noun (usually) is expected."""
  wiki = wikipediaapi.Wikipedia(
      language='en',
      user_agent='my-app-name/0.1 (https://mywebsite.example.com)')

  page = wiki.page(noun_to_search)
  return str(page.summary)


class FinalAnswer(BaseModel):
  final_answer: str = Field(
      ...,
      description="Final answer to be presented to the user. Cannot be null.")
@tool
def final_result(final_answer: str) -> str:
  """Tool to call when you have the final answer to provide to the user. This MUST be the last tool you call in any session."""
  print("FINAL RESULT tool called ========== " + str(final_answer))

  return "ANSWER FROM WIKIPEDIA: " + str(final_answer)



class SearchInReportInput(BaseModel):
  ticker: str = Field(
      ...,
      description="Stock ticker symbol (e.g. AAPL, GOOGL, MSFT). This field is required and cannot be null. Use value near the word 'ticker' in user's question.",
  )
  question: str = Field(
      ...,
      description="User's natural language question about the company's report. Cannot be null."
  )

@tool(args_schema=SearchInReportInput)
def search_company_news(ticker: str) -> str:
  """
  Call this tool in any situation providing question and ticker (AAPL) by default) to fetch latest news for given ticker
  """
  response = fetch_company_news(ticker)

  return response


@tool(args_schema=SearchInReportInput)
def search_in_report(question: str, ticker: str) -> str:
  """
  Call this tool in any situation providing question and ticker by default for searching in the report context.
  """
  response = base_query_report_question_answer(ticker, question)
  return response
