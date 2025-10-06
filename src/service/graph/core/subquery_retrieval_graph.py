from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from onnxruntime.transformers.models.stable_diffusion.diffusion_models import \
  BaseModel
from pydantic import Field
from typing_extensions import TypedDict
from typing import Optional
from src.llm.llm_provider import get_llm


class SubQuestion(TypedDict):
  subquestion: str = Field(description="A subquestion to ask in order to get more relevant context")
  synthetic_answer: str = Field(description="Ideal answer for a new subquestion")

class ComparableAnalysis(BaseModel):
  """A comparable analysis between two answers."""
  reasoning: str = Field(description="If the answer 2 is worse than answer 1, explain why. Think step by step. Expllain what information is missing in answer 2. To be equal to answer 1")
  comparable_numeric_value: int = Field(description="Here should be -1 if answer1 is better, 1 if answer2 is better, 0 if they are equal", ge=-1, le=1)

class SubqueryRetrievalConfig(TypedDict):
  original_question: str
  original_answer: str|None

  last_review: str|None

  iteration: int

  data_reflection: list[str]
  questions: list[str]

  data: set[str]

  processed_questions: list[map]
  all_data : set[str]

  # GIVEN
  max_iterations: int
  min_iterations: int | None

  context_threshold: int
  compression_coef: float

  synthetic_answer_enabled: bool

  # prompts
  answer_prompt: SystemMessage
  compare_prompt: SystemMessage
  subquery_prompt: SystemMessage
  compression_prompt: SystemMessage


  synthetic_answer_prompt: SystemMessage|None


def fetcher_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  # Replace with actual retriever call to Chroma / FAISS etc.

  if len(state['questions']) == 0:

    llm = get_llm().with_structured_output(ComparableAnalysis)
    prompt = ChatPromptTemplate.from_messages([state['synthetic_answer_prompt'], HumanMessage('QUESTION: {question}')])


    chain = prompt | llm
    synthetic_question = chain.invoke({"question": state['original_question']}).content

    state['questions'].append({
        "question": state["original_question"],
        "synthetic_answer": synthetic_question,
      })


  last_question_data = state['questions'][-1]

  retriever = ... # self.db.as_retriever(.) question_to_search
  docs = retriever.get_relevant_documents(last_question_data['question'] + " EXAMPLE: " + last_question_data['synthetic_answer'])

  data = [d.page_content for d in docs]
  state['data'].update(data)

  return state


def answer_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  llm = get_llm()
  prompt = ChatPromptTemplate.from_messages([state['answer_prompt'], HumanMessage('DATA: {data}')])
  chain = prompt | llm

  answer_response = chain.invoke({
    "data": state["data"]
  }).content

  state["questions"][-1]['answer'] = answer_response
  state['original_answer'] = answer_response
  return state

def summary_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  total_length = sum(len(d) for d in state["data"])
  if total_length > state["context_threshold"]:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([state['compression_prompt'], HumanMessage('DATA: {data}')])
    chain = prompt | llm
    compressed_data = chain.invoke({
      "data": "\n".join(state["data"])
    }).content
    state["data"] = {compressed_data}

  return state

def evaluator_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  llm = get_llm().with_structured_output(ComparableAnalysis)
  prompt = ChatPromptTemplate.from_messages([state['compare_prompt'],
   HumanMessage('QUESTION: {question} \n ANSWER1: {answer1} \n ANSWER2: {answer2}')])

  chain = prompt | llm
  # Evaluate relevance of retrieved docs
  analysis:ComparableAnalysis = chain.invoke({
    "question": state["original_question"],
    "answer1": state["questions"][0]["synthetic_answer"],
    "answer2": state["original_answer"]
  })

  if analysis.comparable_numeric_value < 0:
    state["last_review"] = analysis.reasoning  # could be structured
    return SUBQUERY_NODE

  # Otherwise go straight to summary
  return END_NODE


def subquery_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig | str:
  llm = get_llm().with_structured_output(SubQuestion)
  prompt = ChatPromptTemplate.from_messages([state['subquery_prompt'], HumanMessage('ORIGINAL QUESTION: {original_question} ALREADY EXISTED_QUESTIONS: {already_existed_questions}')])
  chain = prompt | llm

  already_existed_questions = "".join([q['question'] for q in state['questions']])

  subquery:SubQuestion = chain.invoke({
    "original_question": state["original_question"],
    "already_existed_questions": already_existed_questions,
  })


  state["questions"].append({"question": subquery.subquestion, "synthetic_answer": subquery.synthetic_answer})

  return "fetcher_node"

def end_node(state: SubqueryRetrievalConfig) -> SubqueryRetrievalConfig:
  return state

# -------------------------
# Graph Definition
# -------------------------
graph = StateGraph(SubqueryRetrievalConfig)

graph.add_node("fetcher_node", fetcher_node)
graph.add_node("evaluator_node", evaluator_node)
graph.add_node("subquery_node", subquery_node)
graph.add_node("compress_node", compress_node)
graph.add_node("summary_node", summary_node)

graph.set_entry_point("fetcher_node")
graph.set_finish_point("summary_node")

graph.add_edge("fetcher_node", "evaluator_node")
graph.add_conditional_edges("evaluator_node", evaluator_node, {
  "compress_node": "compress_node",
  "subquery_node": "subquery_node",
  "summary_node": "summary_node"
})
graph.add_edge("subquery_node", "fetcher_node")
graph.add_edge("compress_node", "summary_node")

app = graph.compile(debug=True)
