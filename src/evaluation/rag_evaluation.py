from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from trulens.apps.app import TruApp
import json
from trulens.core import TruSession, Feedback
import time
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from src.llm.llm_provider import get_llm
from src.models.constants import LLM_CHEEP_SOURCE, LLM_JUDGE_SOURCE
from src.service.file_format_service import soup_html_to_text
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report_full_state
from src.service.query_report_service import base_query_report_question_answer_full_state
from src.usecase.report_uc import save_text_report
from src.util.prompt_manager import prompt_manager
from src.util.logger import logger
from src.util.logger import logger

from src.usecase.report_uc import save_report

#DATA PREPARATION
# with open("/Users/ibahr/Desktop/reports/AAPL.html", "rb") as f:
#   pdf_content = f.read()
#   report = soup_html_to_text(pdf_content)
#
#   metadata = {
#     "ticker": "AAPL",
#     "date": "2025-07-17"
#   }
#
#   save_text_report(report, metadata)

save_report("/Users/ibahr/Desktop/reports/AAPL.html", {"ticker": "AAPL", "date": "2025-07-08"}, "text/html")


# TEST QUESTIONS
test_question_list = [
  "What is the main competitor of AAPL by its report",
  "What is the main product of AAPL",
  "What is the last revenue of AAPL",
  "What is the CEO name of AAPL",
  "What is the main competitor of APPL by it's report?"
]

# INIT TESTING SESSION
session = TruSession()
session.reset_database()

# ANSWER RELEVANCE EVALUATION CHAIN
answer_relevance_criteria_prompt = SystemMessagePromptTemplate.from_template(prompt_manager.get_prompt("answer_relevance"))
human_prompt = HumanMessagePromptTemplate.from_template("""
User question: {question}
Model answer: {answer}
""")
answer_relevance_prompt = ChatPromptTemplate.from_messages([answer_relevance_criteria_prompt, human_prompt])

llm = get_llm(specific_source=LLM_JUDGE_SOURCE)
answer_relevance_criteria_chain = LLMChain(llm=llm, prompt=answer_relevance_prompt, verbose=True)


# GROUNDEDNESS EVALUATION CHAIN
groundedness_criteria_prompt = SystemMessagePromptTemplate.from_template(prompt_manager.get_prompt("groundedness"))
groundedness_human_prompt = HumanMessagePromptTemplate.from_template("""
Context: {context}
Answer: {answer}
Your groundedness score (0.0 to 1.0):
""")

groundedness_prompt = ChatPromptTemplate.from_messages([groundedness_criteria_prompt, groundedness_human_prompt])
llm = get_llm(specific_source=LLM_JUDGE_SOURCE)
groundedness_criteria_chain = LLMChain(llm=llm, prompt=groundedness_prompt, verbose=True)


# CONTEXT RELEVANCE EVALUATION CHAIN
context_relevance_criteria_prompt = SystemMessagePromptTemplate.from_template(prompt_manager.get_prompt("context_relevance"))
context_prompt = HumanMessagePromptTemplate.from_template("""
User question: {question}
Model context: {context}
""")
context_relevance_prompt = ChatPromptTemplate.from_messages([context_relevance_criteria_prompt, context_prompt])

llm = get_llm(specific_source=LLM_JUDGE_SOURCE)
context_relevance_criteria_chain = LLMChain(llm=llm, prompt=context_relevance_prompt, verbose=True)


# METRIC FUNCTIONS
def custom_relevance(prompt: str, res: dict) -> str:
  res = json.loads(res)
  final_answer = str(answer_relevance_criteria_chain.invoke({"question": res['question'], "answer": res['final_answer']})['text']).strip()
  return str(final_answer)

def custom_groundedness(prompt: str, res: dict) -> str:
  res = json.loads(res)
  answer = res['final_answer']
  context = res['context']
  final_answer = str(groundedness_criteria_chain.invoke({"context": context, "answer": answer})['text']).strip()
  return str(final_answer)

def custom_contex_relevance(prompt: str, res: dict) -> str:
  res = json.loads(res)
  question = res['question']
  context = res['context']
  final_answer = str(context_relevance_criteria_chain.invoke({"question": question, "context": context})['text']).strip()
  return str(final_answer)


# TEST WRAPPER

class RephraseCircularRagWrapper:

  @instrument(
      span_type=SpanAttributes.SpanType.RECORD_ROOT,
      attributes={
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
      },
  )
  def rag_wrapper(self, query: str) -> dict:
    out =  run_subquery_search_in_report_full_state(ticker="AAPL", question=query)
    return {"final_answer": out['final_answer'], "context": "".join(out['all_data']), "question": out['original_question']}

  def get_app_name(self):
    return "RephraseCircularRag"

class SimpleSearchRagWrapper:
  @instrument(
      span_type=SpanAttributes.SpanType.RECORD_ROOT,
      attributes={
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
      },
  )
  def rag_wrapper(self, query: str) -> dict:
    out =  base_query_report_question_answer_full_state(ticker="AAPL", query=query)
    return {"final_answer": out['answer'], "context": out["context"], "question": out['question']}

  def get_app_name(self):
    return "SimpleSearchRag"

# RUN TEST
rephrase_circular_rag_app = RephraseCircularRagWrapper()
simple_search_rag_app = SimpleSearchRagWrapper()

# Feedbacks
relevance_feedback = (Feedback(custom_relevance).on_input_output())
groundedness_feedback = (Feedback(custom_groundedness, name="GROUNDEDNESS").on_input_output())
context_relevance_feedback = (Feedback(custom_contex_relevance, name="CONTEXT RELEVANCE").on_input_output())


# # START TEST
circular_app_recorder = TruApp(
    rephrase_circular_rag_app,
    app_name=rephrase_circular_rag_app.get_app_name(),
    feedbacks=[relevance_feedback, groundedness_feedback, context_relevance_feedback],
    app_version="0.01",
    # session=session,
    # feedback_mode=FeedbackMode.DEFERRED
)

with circular_app_recorder as recording:
  for test_question in test_question_list:
    logger.info("Running test question: " + test_question)
    llm_response = rephrase_circular_rag_app.rag_wrapper(test_question)
    logger.info("LLM response:", llm_response)


simple_search_rag_recorder = TruApp(
    simple_search_rag_app,
    app_name=simple_search_rag_app.get_app_name(),
    feedbacks=[relevance_feedback, groundedness_feedback, context_relevance_feedback],
    app_version="0.01",
    # session=session,
    # feedback_mode=FeedbackMode.DEFERRED
)

with simple_search_rag_recorder as recording:
  for test_question in test_question_list:
    logger.info("Running test question: " + test_question)
    llm_response = simple_search_rag_app.rag_wrapper(test_question)
    logger.info("LLM response:", llm_response)


time.sleep(15)
records, feedback = session.get_records_and_feedback()

logger.info("Recorded feedback:", records)
logger.info("Recorded feedback:", feedback)


from trulens.dashboard import run_dashboard
run_dashboard(session)

for rec in records:
  logger.info(rec)

