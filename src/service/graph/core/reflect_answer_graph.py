from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from src.models.mark import ZeroToTenMark
from src.llm.llm_provider import get_llm
from src.util.logger import logger

TASK_NODE = "task_node"
MARK_NODE = "mark_node"
RESOLVER_NODE = "resolver_node"
REVIEW_NODE = "review_node"
END_NODE = "end_node"

class ReflectAnswerState(TypedDict):
  answers: list[str]
  marks: list[int] | None
  accepted_mark: int
  reviews: list[str]
  iteration: int
  max_iterations: int
  min_iterations: int|None
  finished_reason: str | None

  task_role_prompt: str
  task_data_prompt: str
  question_prompt: str

  mark_role_prompt: str

  review_role_prompt: str

  final_answer: str | None


def task_node(state: ReflectAnswerState) -> ReflectAnswerState:
  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{task_role_prompt}\n SUGGESTION IF EXIST: {reviews}"),
    HumanMessagePromptTemplate.from_template("{question_prompt} DATA IS: {task_data_prompt}")])

  chain = prompt | llm

  answer = chain.invoke({"task_role_prompt": state['task_role_prompt'],
                "reviews": state['reviews'] if state['reviews'] is not None else "N/A",
                "task_data_prompt": state['task_data_prompt'],
                "question_prompt": state['question_prompt']}).content
  state['answers'].append(answer)
  state['iteration'] += 1
  return state

def mark_node(state: ReflectAnswerState) -> ReflectAnswerState:
  llm = get_llm().with_structured_output(ZeroToTenMark)

  prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{mark_role_prompt}"),
    HumanMessagePromptTemplate.from_template("INPUT QUESTION TO ANSWER: {question} INPUT DATA TO TAKE IN TO ACCOUNT: {task_data_prompt} INPUT ANSWER: {test_answer}.")])

  chain = prompt | llm

  mark: ZeroToTenMark = chain.invoke({
    "mark_role_prompt": state['mark_role_prompt'],
    "question": state['question_prompt'],
    "task_data_prompt": state['task_data_prompt'],
    "test_answer": state['answers'][-1]})

  state['marks'].append(mark.mark)
  return state

def resolver_node(state: ReflectAnswerState) -> ReflectAnswerState|str:
  logger.info(f"MARK RESOLVER NODE - Iteration {state['iteration']}")

  if 'min_iterations' in state and state['min_iterations'] > state['iteration']:
    return REVIEW_NODE

  if state['iteration'] >= state['max_iterations']:
    state['finished_reason'] = "MAX_ITERATIONS_REACHED"
    return END_NODE

  if state['marks'][-1] >= state['accepted_mark']:
    state['finished_reason'] = "ACCEPTED_MARK_REACHED"
    return END_NODE

  return REVIEW_NODE

def review_node(state: ReflectAnswerState) -> ReflectAnswerState:
  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{review_role_prompt}"),
    HumanMessagePromptTemplate.from_template("INPUT QUESTION TO ANSWER: {question} INPUT DATA TO TAKE IN TO ACCOUNT: {task_data_prompt} INPUT ANSWER: {test_answer}.")])

  chain = prompt | llm

  review = chain.invoke({
    "review_role_prompt": state['review_role_prompt'],
    "question": state['question_prompt'],
    "task_data_prompt": state['task_data_prompt'],
    "test_answer": state['answers'][-1]}).content

  state['reviews'].append(review)
  return state

def end_node(state: ReflectAnswerState) -> ReflectAnswerState:
  logger.info(f"END NODE - Iteration {state['iteration']}. Reason: {state['finished_reason']}. First mark {state['marks'][0] if state['marks'] else 'N/A'}, last mark {state['marks'][-1] if state['marks'] else 'N/A'}")
  # todo return best not last
  state['final_answer'] = state['answers'][-1]
  return state


graph = StateGraph(ReflectAnswerState)

graph.add_node(TASK_NODE, task_node)
graph.add_node(MARK_NODE, mark_node)

graph.add_node(REVIEW_NODE, review_node)
graph.add_node(END_NODE, end_node)

graph.set_entry_point(TASK_NODE)
graph.set_finish_point(END_NODE)

graph.add_edge(TASK_NODE, MARK_NODE)
graph.add_conditional_edges(MARK_NODE, resolver_node, {
  END_NODE: END_NODE,
  REVIEW_NODE: REVIEW_NODE
})
graph.add_edge(REVIEW_NODE, TASK_NODE)

app = graph.compile(debug=True)

# todo parametrize prompts should be prompts msgs not strings
def run_reflect_agent(
    question: str,
    task_data_prompt: str,
    task_role_prompt: str,
    mark_role_prompt: str,
    review_role_prompt: str,

    max_iterations: int,
    accepted_mark: int
) -> map:
  initial_state = ReflectAnswerState(
      answers=[],
      marks=[],
      accepted_mark=accepted_mark,
      reviews=[],
      iteration=0,
      max_iterations=max_iterations,
      finished_reason=None,

      task_role_prompt=task_role_prompt,
      task_data_prompt=task_data_prompt,
      question_prompt=question,

      mark_role_prompt=mark_role_prompt,

      review_role_prompt=review_role_prompt,

      final_answer=None
  )

  final_state = app.invoke(initial_state)
  logger.info(final_state)
  return final_state


if __name__ == "__main__":
  # initial_state = ReflectAnswerState(
  #     answers=[],
  #     marks=[],
  #     accepted_mark=8,
  #     review=None,
  #     iteration=0,
  #     max_iterations=5,
  #     finished_reason=None,
  #
  #     task_role_prompt="You are a highly intelligent AI tasked with answering questions based on provided data. Your goal is to provide accurate, concise, and relevant answers. If the data does not contain enough information to answer the question, respond with 'Insufficient data to answer the question.'",
  #     task_data_prompt="Germany’s society is technically structured into Länder (federal states), each with its own constitution, parliament, and police… but let’s keep it irrelevant like you asked:Think of Germany as a giant cuckoo clock: every hour, little figures pop out — some are engineers, some are bakers, and one is a techno DJ from Berlin. The gears inside are made of sausages (Bratwurst for local administration, Currywurst for federal laws). Beer gardens act as informal parliaments where debates about recycling bins carry the same weight as constitutional amendments. Trains pretend to run on time, but the secret structuring principle is actually the distribution of gnomes in Bavarian front yards. At the deepest level, the whole society is synchronized by the collective sound of opening beer bottles during Bundesliga matches.Want me to make it even more derailed — like Germany’s social structure explained through pretzels and techno beats only?",
  #     question_prompt="How France society is structured?",
  #
  #     mark_role_prompt="You are an impartial judge tasked with evaluating the quality of answers provided by an AI. You will rate each answer on a scale from 0 to 10, where 10 is an excellent answer that fully addresses the question with accuracy and relevance, and 0 is a poor answer that fails to address the question or is completely incorrect. Consider factors such as accuracy, relevance, completeness, clarity, and conciseness when assigning your score. Provide a brief explanation for your rating if necessary.",
  #
  #     review_role_prompt="You are an expert reviewer tasked with providing constructive feedback on answers generated by an AI. Your goal is to help improve the quality of future answers by identifying areas of weakness and suggesting specific improvements. When reviewing an answer, consider factors such as accuracy, relevance, completeness, clarity, and conciseness. Provide clear and actionable feedback that the AI can use to enhance its performance in subsequent iterations.",
  #
  #     final_answer=None
  # )
  #
  # final_state = app.invoke(initial_state)
  # logger.info(final_state)

  logger.info(app.get_graph().draw_mermaid())


