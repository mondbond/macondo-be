from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from src.service.graph.core.reflect_answer_graph import run_reflect_agent
from src.service.thirdparty.news.finhub_news_service import fetch_company_news
from src.service.thirdparty.stock_price_change_service import get_price_change_for_tickers
from src.tools.tools import report_rephrase_retriever_search
from typing_extensions import TypedDict
from src.llm import llm_provider
from src.util.env_property import LLM_SOURCE_REASONING
from src.util.prompt_manager import prompt_manager
from src.util.logger import logger

REASON_PROMPT = prompt_manager.get_prompt('stock_fall_explanation_uc_logic')
REFLECT_QUESTION = prompt_manager.get_prompt("stock_fall_explanation_uc_question")
TASK_ROLE_PROMPT = prompt_manager.get_prompt("stock_fall_explanation_uc_role")
MARK_ROLE_PROMPT = prompt_manager.get_prompt("stock_fall_explanation_uc_mark")
REVIEW_ROLE_PROMPT = prompt_manager.get_prompt('stock_fall_explanation_uc_review')

COLLECT_FALL_TICKERS_NODE = "COLLECT_FALL_TICKERS"
GENERATE_VERDICT_NODE = "GENERATE_VERDICT"
GENERATE_MSG_NODE = "GENERATE_MSG"


class FallenCompanyVerdict(BaseModel):
    """Structured output model for the verdict on a fallen company's stock price. Include 3 sentences summary and final verdict."""
    summary_text: str = Field(description="Using provided information, give a concise summary verdict around 7 sentences explaining why you are thinking so. And main answer - if this fall is temporary or long term. Think step by step explaining why exactly you think so based on provided information. If you don't have enough information to make a decision, say that you don't have enough information.")
    verdict_type: str = Field(description='Only three values are allowed: "Correction", "Structural Issue", "Not Enough Information". If you think that the fall is temporary and likely to rebound, choose "Correction". If you believe the fall is due to a structural issue that could lead to further decline, select "Structural Issue". If the information provided is insufficient to make a clear determination, select "Not Enough Information".')


class NegativeFiveToFiveMark(BaseModel):
    """Schema for making marks in range from -5 to 5 and explaining your reasoning."""
    reasoning: str = Field(description="Provide a brief explanation for your rating thinking step by step explaining your thoughts why you giving this mark.")
    mark: int = Field(description="Provide a score from -5 to 5 based on provided data to mark if I should buy the stock share or not where -5 means I absolutely should not to do it and 0 is full neutral mark and 5 - absolutely should buy.")


class CompanyFallExplanation(TypedDict):
    ticker: str
    change: float | None
    report_risk_factors: str | None
    news: list | None
    summary: str | None
    verdict: str | None
    reasoning: str | None
    verdict_type: str | None
    finished: bool | None
    reflection_state: dict | None


class GraphFallExplainState(TypedDict):
    tickers_to_check: list[str] | None
    company_fall_explanation: list[CompanyFallExplanation] | None
    answer: str | None


# Main graph node: collect fall tickers and news, risk factors
def collect_fall_change_tickers(
    state: GraphFallExplainState) -> GraphFallExplainState:
    tickers = state.get("tickers_to_check") or []
    tickers = [str(t) for t in tickers if isinstance(t, str)]
    ticker_to_change_map = get_price_change_for_tickers(tickers)
    for ticker in tickers:
        change = ticker_to_change_map[ticker]
        if change < 0:
            company_fall_explanation: CompanyFallExplanation = {
                "ticker": ticker,
                "change": change,
                "report_risk_factors": None,
                "news": None,
                "summary": None,
                "verdict": None,
                "reasoning": None,
                "verdict_type": None,
                "finished": None,
                "reflection_state": None
            }

            state['company_fall_explanation'] = state.get('company_fall_explanation',
                                                          []) + [
                                                  company_fall_explanation]

    for fall_company in state['company_fall_explanation']:
        news = fetch_company_news(fall_company['ticker'])
        fall_company['news'] = news
    logger.info("Tickers with significant fall:",
                state['company_fall_explanation'])

    for fall_company in state['company_fall_explanation']:
        if not fall_company.get('news') or len(fall_company['news']) == 0:
            fall_company['verdict'] = "No news found to explain the stock price fall."
            fall_company['finished'] = True

    for fall_company in state['company_fall_explanation']:
        if not fall_company.get('finished'):
            query = ("List the risk factors mentioned in the latest financial report. "
                     "Provide a concise summary of each risk factor for share stock prise. "
                     "With given score of impact for each risk factor by yourself using reasonong.")
            response = report_rephrase_retriever_search(fall_company['ticker'], query)['answer']
            fall_company['report_risk_factors'] = response

    return state


def generate_verdict_node(
    state: GraphFallExplainState) -> GraphFallExplainState:
    logger.info("VERDICT NODE")

    for idx, company_explanation in enumerate(state['company_fall_explanation']):
        if company_explanation.get('finished'):
            continue

        data = ""
        data += f"Ticker: {company_explanation['ticker']}\n"
        data += f"Share price change: {company_explanation['change']}%\n"
        data += f"Report Risk Factors: {company_explanation.get('report_risk_factors', 'No data')}\n"
        if 'news' in company_explanation and company_explanation['news'] and len(
            company_explanation['news']) > 0:
            news_summaries = [f"- {news_item['headline']}\n{news_item['text']}" for
                              news_item in company_explanation['news']]
            data += "Recent News that can be relevant and can affect the price:\n" + "\n".join(
                news_summaries) + "\n"
        else:
            data += "No relevant news found.\n"

        reflect_state = run_reflect_agent(
            question=REFLECT_QUESTION,
            task_data_prompt=data,
            task_role_prompt=TASK_ROLE_PROMPT,
            mark_role_prompt=MARK_ROLE_PROMPT,
            review_role_prompt=REVIEW_ROLE_PROMPT,

            max_iterations=3,
            accepted_mark=8
        )

        llm = llm_provider.get_llm(
          specific_source=LLM_SOURCE_REASONING).with_structured_output(
          NegativeFiveToFiveMark)
        prompt = ChatPromptTemplate.from_messages([
          ("system", REASON_PROMPT),
          ("human", """
          Ticker: {ticker}
          Change: {change}%
          Analytical summary: {analytical_summary}

          Please provide a verdict in a mark form if I should buy the stock share right now or not.
          """)
        ])

        chain = prompt | llm

        response: NegativeFiveToFiveMark = chain.invoke({
          "ticker": company_explanation['ticker'],
          "change": company_explanation['change'],
          "analytical_summary": reflect_state['final_answer'] if isinstance(reflect_state, dict) else None
        })

        state['company_fall_explanation'][idx]['verdict_type'] = str(response.mark)
        state['company_fall_explanation'][idx]['verdict'] = response.reasoning
        state['company_fall_explanation'][idx]['reasoning'] = reflect_state['final_answer'] if isinstance(reflect_state, dict) else None
        state['company_fall_explanation'][idx]['reflection_state'] = dict(reflect_state) if not isinstance(reflect_state, dict) and hasattr(reflect_state, 'items') else reflect_state
        state['company_fall_explanation'][idx]['finished'] = True

    return state


def form_response_node(state: GraphFallExplainState) -> GraphFallExplainState:
    logger.info("FORM RESPONSE NODE")

    answer = ""
    for idx, company_explanation in enumerate(state['company_fall_explanation']):
        if not company_explanation.get('finished'):
            continue

        answer += "\n\n"
        answer += company_explanation['ticker'] + "\n"
        answer += str(company_explanation['change']) + "\n"

        if 'news' in company_explanation and company_explanation['news'] and len(
            company_explanation['news']) > 0:
            news_summaries = [f"- {news_item['headline']}\n{news_item['url']}" for
                              news_item in company_explanation['news']]
            answer += "News:\n" + "\n".join(news_summaries) + "\n"
        else:
            answer += "No relevant news found.\n"
            continue

        if 'reasoning' in company_explanation and company_explanation['reasoning']:
            answer += "\nReasoning:\n" + company_explanation['reasoning'] + "\n"

        if 'verdict' in company_explanation and company_explanation['verdict']:
            answer += "\n\nVerdict:\n" + company_explanation['verdict'] + "\n"
            answer += "\nFinal Verdict:\n" + str(
                company_explanation['verdict_type']) + "\n"

    state['answer'] = answer

    return state


graph = StateGraph(GraphFallExplainState)
graph.add_node(COLLECT_FALL_TICKERS_NODE, collect_fall_change_tickers)
graph.add_node(GENERATE_VERDICT_NODE, generate_verdict_node)
graph.add_node(GENERATE_MSG_NODE, form_response_node)

graph.set_entry_point(COLLECT_FALL_TICKERS_NODE)
graph.set_finish_point(GENERATE_MSG_NODE)
graph.add_edge(COLLECT_FALL_TICKERS_NODE, GENERATE_VERDICT_NODE)
graph.add_edge(GENERATE_VERDICT_NODE, GENERATE_MSG_NODE)

app = graph.compile(debug=True)


def run_company_fall_explanation_graph(
    tickers: list[str]) -> GraphFallExplainState:
    initial_state = GraphFallExplainState(
        tickers_to_check=tickers,
        company_fall_explanation=[],
        answer=None
    )

    final_state = app.invoke(initial_state)
    logger.info("Final State:", final_state)
    return final_state['answer']
