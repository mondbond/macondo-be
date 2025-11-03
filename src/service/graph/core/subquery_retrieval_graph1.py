# fixed_subquery_retrieval.py
from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from src.llm.llm_provider import get_llm
from src.service.file_format_service import soup_html_to_text
from src.service.query_report_service import base_query_report_question_answer
from src.usecase.report_uc import save_text_report
from src.util.prompt_manager import prompt_manager
from src.util.logger import logger

FETCH_NODE = "FETCH_NODE"
ANSWER_NODE = "ANSWER_NODE"
EVALUATOR_NODE = "EVALUATOR_NODE"
SUBQUERY_NODE = "SUBQUERY_NODE"
COMPRESS_NODE = "COMPRESS_NODE"
EVALUATE_CONDITION = "EVALUATE_CONDITION"
END_NODE = "END_NODE"


class ComparableAnalysis(BaseModel):
  """A comparable analysis between two answers."""
  reasoning: str = Field(..., description="If second answer is worse than first answer, explain why step-by-step. And what information is missing in second answer to be as good as first answer.")
  comparable_numeric_value: int = Field(
      ...,
      description="return -1 if first answer better then second. return 1 if second answer is better, 0 if equal",
      ge=-1,
      le=1,
  )

class SubQuestion(BaseModel):
  subquestion: str = Field(description="A subquestion to ask in order to get more relevant context")

class SubqueryRetrievalConfig(TypedDict, total=False):
  # Outer information
  ticker: str
  original_question: str

  # prompts - keep as simple strings; we'll wrap into ChatPromptTemplate at runtime
  answer_prompt: SystemMessage
  compare_prompt: SystemMessage
  subquery_prompt: SystemMessage
  compression_prompt: SystemMessage
  synthetic_answer_prompt: SystemMessage

  # Inner properties
  questions: List[Dict[str, str]]
  all_data: List[str]
  original_answer: Optional[str]

  last_review: Optional[str]
  last_mark: Optional[int]

  context_threshold: int
  compression_coef: float

  # iteration control
  iteration: int
  max_iterations: int
  min_iterations: int | None

  # outputs
  end_reason: Optional[str]
  final_answer: Optional[str]


def generate_synthetic_answer(question: str, prompt: SystemMessage) -> str:
  llm = get_llm()
  prompt = ChatPromptTemplate.from_messages(
      [
        prompt,
        ("human", "QUESTION: {question}")
      ]
  )
  chain = prompt | llm
  return chain.invoke({"question": question}).content

def fetcher_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  if len(state["questions"]) == 0:

    synthetic_answer = generate_synthetic_answer(state['original_question'], state["synthetic_answer_prompt"])
    state["questions"].append({
      "question": state["original_question"],
      "synthetic_answer": synthetic_answer
    })

  last_question_data = state["questions"][-1]
  query_to_search = last_question_data["question"]
  synthetic_answer_to_serarch = last_question_data["synthetic_answer"]

  query = query_to_search + (f" \nEXAMPLE: {synthetic_answer_to_serarch}")

  reports = base_query_report_question_answer(ticker = state["ticker"], query=query, join=False)

  data = [d.page_content for d in reports]
  # append new unique data pieces
  for p in data:
    if p not in state["all_data"]:
      state["all_data"].append(p)

  state['iteration'] += 1

  return state

def compress_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  joined_data = "\n\n".join(state["all_data"])

  if len(joined_data) >= state["context_threshold"]:
    target_length = int(state["context_threshold"] * state["compression_coef"])
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
      (state["compression_prompt"]),
      ("human", "QUESTION:{question}\nDATA: {data}\nTARGET_CHARS_COUNT: {target_length}")
    ])
    chain = prompt | llm

    compressed_data = state["all_data"]
    try:
      compressed = chain.invoke({
        "question": state["original_question"],
        "data": joined_data,
        "target_length": target_length
      }).content
      compressed_data = [compressed]
    except Exception as e:
      pass

    # replace state data with compressed summary
    state["all_data"] = compressed_data
  return state


def answer_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages([
    (state["answer_prompt"]),
    ("human", "DATA: {data}\nQUESTION: {question}")
  ])
  chain = prompt | llm
  joined_data = "\n\n".join(state["all_data"])

  try:
    answer_response = chain.invoke({
      "data": joined_data,
      "question": state["original_question"]
    }).content

  except Exception as e:
    answer_response = f"LLM error generating answer: {e}"

  state["original_answer"] = answer_response
  return state

def evaluator_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  llm = get_llm().with_structured_output(ComparableAnalysis)

  synthetic = state["questions"][0]["synthetic_answer"]
  actual = state["original_answer"]

  prompt = ChatPromptTemplate.from_messages([
    (state["compare_prompt"]),
    ("human", "QUESTION: {question}\n ANSWER ONE: {synth}\nANSWER TWO: {actual}")
  ])
  chain = prompt | llm
  try:
    analysis: ComparableAnalysis = chain.invoke({
      "question": state["original_question"],
      "synth": synthetic,
      "actual": actual
    })
    state["last_review"] = analysis.reasoning
    state["last_mark"] = int(analysis.comparable_numeric_value)
  except Exception:
    pass

  return state

def evaluator_condition(state: SubqueryRetrievalConfig) -> str:
  if state['iteration'] >= state['max_iterations']:
    state["end_reason"] = "max_iterations"
    return END_NODE

  if state['min_iterations'] and state['iteration'] < state['min_iterations']:
    return SUBQUERY_NODE

  cmp_val = state.get("last_mark", 0)
  if cmp_val < 0:
    if state["iteration"] < state["max_iterations"]:
      return SUBQUERY_NODE
    else:
      state['end_reason'] = "answer good"
      return END_NODE

  return END_NODE

def subquery_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  llm = get_llm().with_structured_output(SubQuestion)

  prompt = ChatPromptTemplate.from_messages([
    (state["subquery_prompt"]),
    ("human", "ORIGINAL_QUESTION: {original_question}\nALREADY_ASKED: {already}")
  ])
  chain = prompt | llm
  already = " || ".join(q["question"] for q in state["questions"])

  subq_result:SubQuestion = chain.invoke({
    "original_question": state["original_question"],
    "already": already
  })

  synthetic_answert = generate_synthetic_answer(subq_result.subquestion, state["synthetic_answer_prompt"])
  state["questions"].append({
    "question": subq_result.subquestion,
    "synthetic_answer": synthetic_answert
  })
  return state

def end_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  state["final_answer"] = state["original_answer"]
  return state

graph = StateGraph(SubqueryRetrievalConfig)

graph.add_node(FETCH_NODE, fetcher_node)
graph.add_node(ANSWER_NODE, answer_node)
graph.add_node(EVALUATOR_NODE, evaluator_node)
graph.add_node(SUBQUERY_NODE, subquery_node)
graph.add_node(COMPRESS_NODE, compress_node)
graph.add_node(END_NODE, end_node)

graph.set_entry_point(FETCH_NODE)
graph.set_finish_point(END_NODE)

graph.add_edge(FETCH_NODE, COMPRESS_NODE)
graph.add_edge(COMPRESS_NODE, ANSWER_NODE)
graph.add_edge(ANSWER_NODE, EVALUATOR_NODE)

graph.add_conditional_edges(EVALUATOR_NODE, evaluator_condition, {
  SUBQUERY_NODE: SUBQUERY_NODE,
  END_NODE: END_NODE,
})

graph.add_edge(SUBQUERY_NODE, FETCH_NODE)

app = graph.compile(debug=True)


def run_subquery_search_in_report(ticker: str, question: str) -> str:
  initial_state: SubqueryRetrievalConfig = {
  "ticker": ticker,
  "original_question": question,
  "original_answer": None,

  "last_review": None,
  "iteration": 0,

  "questions": [],
  "all_data": [],

  "max_iterations": 3,
  "min_iterations": 0,

  "context_threshold": 4000,
  "compression_coef": 0.7,

  "answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_answer")),
  "compare_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_answer_comparator')),
  "subquery_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_rephrase_question')),
  "compression_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_compression")),
  "synthetic_answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_synthetic_answer")),
  "final_answer": None
}

  final_state = app.invoke(initial_state)
  logger.info("\n=== FINAL STATE ===")
  logger.info(final_state)
  return final_state['final_answer']



def run_subquery_search_in_report_full_state(ticker: str, question: str) -> dict:
  initial_state: SubqueryRetrievalConfig = {
    "ticker": ticker,
    "original_question": question,
    "original_answer": None,

    "last_review": None,
    "iteration": 0,

    "questions": [],
    "all_data": [],

    "max_iterations": 1,
    "min_iterations": 0,

    "context_threshold": 4000,
    "compression_coef": 0.7,

    "answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_answer")),
    "compare_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_answer_comparator')),
    "subquery_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_rephrase_question')),
    "compression_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_compression")),
    "synthetic_answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_synthetic_answer")),

    "final_answer": None
  }

  final_state = app.invoke(initial_state)
  logger.info("\n=== FINAL STATE ===")
  logger.info(final_state)
  return final_state


if __name__ == "__main__":

  # import os
  # for k in ["LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]:
  #   logger.info(k, "=", os.getenv(k))
  #
  # # with open("/Users/ibahr/Downloads/synthetic_report.pdf", "rb") as f:
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
  # initial_state: SubqueryRetrievalConfig = {
  #   "ticker": "UBER",
  #   "original_question": "What are the key risks for Company Uber in the latest 10-K report?",
  #   "original_answer": None,
  #
  #   "last_review": None,
  #   "iteration": 0,
  #
  #   "questions": [],
  #   "all_data": [],
  #
  #   "max_iterations": 1,
  #   "min_iterations": 0,
  #
  #   "context_threshold": 2000,
  #   "compression_coef": 0.7,
  #
  #   "answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_answer")),
  #   "compare_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_answer_comparator')),
  #   "subquery_prompt": SystemMessage(prompt_manager.get_prompt('rephrase_retrieval_uc_rephrase_question')),
  #   "compression_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_compression")),
  #   "synthetic_answer_prompt": SystemMessage(prompt_manager.get_prompt("rephrase_retrieval_uc_synthetic_answer")),
  #
  #   "final_answer": None
  # }
  #
  # final_state = app.invoke(initial_state)
  # logger.info("\n=== FINAL STATE ===")
  # logger.info(final_state)
  logger.info(app.get_graph().draw_mermaid())

