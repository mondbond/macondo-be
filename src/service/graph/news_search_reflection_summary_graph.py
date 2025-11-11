from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from src.service.thirdparty.news.finhub_news_service import fetch_company_news
from src.tools.tools import search_company_news
from typing_extensions import TypedDict
from src.util.logger import logger
from src.llm import llm_provider


ACT_NODE = "ACT"
REFLECT_NODE = "REFLECT"
SUMMARY_NODE = "SUMMARY"
END_NODE = "END"


class ArticleRelevance(BaseModel):
    """Schema for article relevance score."""
    relevance_score: int = Field(description="Based on provided news article and user query - provide relevance score from 0 to 10, where 10 is highly relevant to user's query and 0 is not relevant at all.")


class NewsGraphReflectionState(TypedDict):
    """Graph reflection state."""
    counter: int
    ticker: str | None
    input_query: str | None
    tools: list
    relevant_news: list | None
    summary: str | None
    end_reason: str | None
    answer: str | None


# Main graph node: act on the ticker and fetch news
def act_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
    logger.info(f"ACT NODE - Step {state['counter'] + 1}")
    news_list = fetch_company_news(state['ticker'])
    state['counter'] += 1
    state['relevant_news'].extend(news_list)
    return state


def reflect_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
    logger.info(f"REFLECT NODE - Step {state['counter'] + 1}")
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
    logger.info(f"SUMMARY NODE - Step {state['counter'] + 1}")
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
    state['end_reason'] = "Summary of relevant news articles provided"
    return state


def form_response_node(state: NewsGraphReflectionState) -> NewsGraphReflectionState:
    if state['end_reason'] != "Summary of relevant news articles provided":
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
    logger.info(final_state)
    return final_state['answer']
