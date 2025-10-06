import json

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import MessageGraph, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.tools.tools import report_rephrase_retriever_search, search_in_report, search_company_news
from typing_extensions import TypedDict, Literal

from src.llm import llm_provider


mock = [
  {
    "headline": "Oracle soars, TSMC & Broadcom emerge winners: Market Wrap",
    "id": "0ORCL",
    "relevance_score": 0,
    "summary": "Here are the stocks that made the most significant market moves today.",
    "text": "Oracle soars, TSMC & Broadcom emerge winners: Market Wrap. Here are the stocks that made the most significant market moves today.",
    "url": "https://finnhub.io/api/news?id=97425574bd9da7160317e08e37f7d18ca6a377828eaa3ba3f424e2dd1c2f5b77",
  },
  {
    "headline": "Who is Larry Ellison, the new richest person in the world?",
    "id": "1ORCL",
    "relevance_score": 0,
    "summary": "Oracle co-founder Larry Ellison is now the world's richest person, according to Bloomberg's latest ranking.",
    "text": "Who is Larry Ellison, the new richest man in the world?\n\nOracle co-founder Larry Ellison is now the world's richest person, according to Bloomberg's latest ranking.",
    "url": "https://finnhub.io/api/news?id=b3b16844c485f9d80305ea56078597d8316523004dd7e42bc3035430834156a9",
  },
  {
    "headline": "Dow Jones Futures: Nvidia, GE Vernova Lead New Buys As Oracle Drives AI Stocks",
    "id": "2ORCL",
    "summary": "Oracle fueled AI stocks Wednesday, with Nvidia and GE Vernova leading new buys. But the stock market had a lackluster day.",
    "text": "There are no important events for this country at this time. Select \"All\" to see top events in other countries or view all events.",
    "url": "https://finnhub.io/api/news?id=fdc09589b2716b3b9c5611dc94b44834c44809b6b8acb7d0d5903854623d410a",
  },
  {
    "headline": "Jobs Data Mess & AI's Moment: Unpacking the Week's Biggest News",
    "id": "3ORCL",
    "summary": "Following a significant jobs revision that signals an weaker economy than previously thought, the AI industry's growth has become even more impressive. Recent news from Nebius and Oracle debunks key criticisms of the AI boom.",
    "text": "BLS Data Debacle\n\nEven though the market is just now making its way out of the ‘Summer doldrums and many institutional investors are returning from summer vacations, Wall Street is off to an eventful week...",
    "url": "https://finnhub.io/api/news?id=d3527b91e3f42727bc6d8c75318e621bd392ea008ef80930c6d5c34daf391984",
  },
  {
    "headline": "S&P, Nasdaq notch record-high closes as Oracle soars on AI optimism",
    "id": "4ORCL",
    "summary": "STORY: U.S. stocks ended mixed on Wednesday, as the Dow dropped almost half a percent, the S&P 500 gained three tenths of a percent to notch its second straight closing high, and the Nasdaq ticked up just enough to record its third consecutive record high close.",
    "text": "STORY: U.S. stocks ended mixed on Wednesday, as the Dow dropped almost half a percent, the S&P 500 gained three tenths of a percent to notch its second straight closing high...",
    "url": "https://finnhub.io/api/news?id=71f34f2b3f1221903b3e3580efa5a3b84c96ed5f164a7aac7740e7159ac2c04e",
  },
]


ACT_NODE = "ACT"
REFLECT_NODE = "REFLECT"
SUMMARY_NODE = "SUMMARY"
END_NODE = "END"


class ArticleRelevance(BaseModel):
  """Schema for article relevance score."""
  relevance_score: int  = Field(description="Based on provided news article and user query - provide relevance score from 0 to 10, where 10 is highly relevant to user's query and 0 is not relevant at all.")

class NewsGraphReflectionState(TypedDict):
  """Graph reflection state."""
  counter: int
  ticker: str|None
  input_query: str|None
  tools: list
  relevant_news: list | None
  summary: str | None
  end_reason: str|None
  answer: str | None


def act_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
  print(f"ACT NODE - Step {state['counter'] + 1}")
  news_list = mock

  state['counter'] += 1
  state['relevant_news'].extend(news_list)
  return state


def reflect_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
  print(f"REFLECT NODE - Step {state['counter'] + 1}")

  """Condition to reflect and decide next action."""


  news_list = state['relevant_news']

  news_prompt = ChatPromptTemplate.from_messages([
    ("system", """
      You are an evaluator. Given a user query and one news article, output ONLY a relevance score 0–10 in the required schema.
      0 = company not mentioned at all.
      10 = article fully focused on the company/topic in the query.
    """),
    ("user", "USER QUERY: {input}\n\nNEWS ARTICLE: {news_article}")
  ])
  llm = llm_provider.get_llm().with_structured_output(ArticleRelevance)

  chain = news_prompt | llm


  for news in news_list:
    article_text_for_llm = f"HEADLINE: {news['headline']}  SUMMARY: {news['summary']} ARTICLE TEXT: {news['text']}"
    article_relevance: ArticleRelevance = chain.invoke({"input": str(state['input_query']), "news_article": article_text_for_llm})
    news['relevance_score'] = article_relevance.relevance_score

  return state


def summary_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
  print(f"SUMMARY NODE - Step {state['counter'] + 1}")

  """Condition to summarize the findings."""
  for news in state['relevant_news']:
    print(f"News: {news['headline']} \\n Relevance Score: {news.get('relevance_score', 'NO relevance score')}")

  relevant_news = [news for news in state['relevant_news'] if int(news.get('relevance_score', '0')) >= 5]

  summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """
      You're a helpful assistant that summarizes relevant news articles for a user's query. You get the original user query and a list of relevant news articles.
      Based on this, you need to provide a concise summary of the most relevant information from the articles that answers the user's query.
      """),
    ("human", "USER QUERY: {input} \n\n RELEVANT NEWS ARTICLES: {relevant_news}")
  ])

  llm = llm_provider.get_llm()

  chain = summary_prompt | llm

  relevant_news_str = "\n".join([f"HEADLINE: {news['headline']} SUMMARY: {news['summary']} ARTICLE TEXT: {news['text']}" for news in relevant_news])

  summary = chain.invoke({"input": str(state['input_query']), "relevant_news": relevant_news_str})

  state['summary'] = summary.content
  state['end_reason'] = f"Summary of relevant news articles provided"

  return state




def form_response_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
  """Condition to end the graph process."""
  if state['end_reason'] is not "Summary of relevant news articles provided":
    state['answer'] = "No relevant news found for given query."
    return state
  else:
    news_ref = ""

    for news in state['relevant_news']:
      news_ref += f"- (Relevance Score: {news.get('relevance_score', 'NO relevance score')})\nHEADLINE: {news['headline']}\nURL: {news['url']}\n"

    state['answer'] = news_ref + " \n \n SUMMARY OF RELEVANT NEWS:\n" + state['summary']


  return state

graph = StateGraph(NewsGraphReflectionState)
graph.add_node(ACT_NODE, act_node)
graph.add_node(REFLECT_NODE, reflect_node)
graph.add_node(SUMMARY_NODE, summary_node)
graph.add_node(END_NODE, form_response_node)

graph.set_entry_point(ACT_NODE)
graph.set_finish_point(END_NODE)

graph.add_edge(ACT_NODE, REFLECT_NODE)
graph.add_edge(REFLECT_NODE, SUMMARY_NODE)
graph.add_edge(SUMMARY_NODE, END_NODE)

app = graph.compile(debug=True)

def run_news_graph(ticker, query) -> NewsGraphReflectionState:
  initial_state = NewsGraphReflectionState(
      counter=0,
      ticker=ticker,
      input_query=query,
      tools=[search_company_news],
      relevant_news=[],
      summary=None,
      end_reason=None
  )

  final_state = app.invoke(initial_state)
  print(final_state)
  return final_state['answer']

if __name__ == "__main__":

  query = "What is the last meaningful news about ORCL Oracle company?"
  initial_state = NewsGraphReflectionState(
      counter=0,
      ticker="ORCL",
      input_query=query,
      tools=[search_company_news],
      relevant_news=[],
      summary=None,
      end_reason=None
  )

  final_state = app.invoke(initial_state)
  print("Final State:", final_state)
