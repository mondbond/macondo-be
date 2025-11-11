import json

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import MessageGraph, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from src.util.logger import logger
from typing_extensions import TypedDict, Literal
from src.llm.llm_provider import get_llm

ACT_NODE = "ACT"
REFLECT_NODE = "REFLECT"
END_NODE = "END"


class ProcessedQuery(BaseModel):
    """Processed query with tools and response."""
    changed_query: str = Field(description="If question is not answered precise - this should be modified user query after to search better next time. Word 'ticker' should always be in the question")
    is_query_answered: bool = Field(description="Indicates if the query has been answered precise.")

class GraphReflectionState(TypedDict):
    """Graph reflection state."""
    counter: int
    ticker: str | None
    input_query: list[str]
    tools: list
    fetched_data: list[str] | None
    end_reason: str | None
    summary: str | None

def act_node(state: GraphReflectionState) -> GraphReflectionState:
    logger.info(f"ACT NODE - Step {state['counter'] + 1}")
    name2tool = {tool.name: tool for tool in state['tools']}
    llm = get_llm()
    llm_tool = llm.bind_tools(state['tools'], tool_choice="any")
    query = state['input_query'][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You're a helpful assistant. When answering a user's question "
         "you always MUST use one of the tools provided to obtain more information for user answer. You MUST NOT answer user's question directly. You must always use one of the tools. You must not describe function calling in context. You should use only tool calling "),
        ("human", "{input}"),
    ])
    chain = prompt | llm_tool
    tool_call_response = chain.invoke({"input": query})
    js = json.loads(tool_call_response.json())
    print("TOOL CALL RESPONSE JSON:", js)
    js = js["tool_calls"][0]
    tool_name = js["name"]
    tool_args = js["args"]
    logger.info(tool_args)
    tool_obj = name2tool[tool_name]
    output = tool_obj.invoke(tool_args)
    state['counter'] += 1
    state['fetched_data'].append(output)
    return state

def reflect_node(state: GraphReflectionState) -> Command[Literal[ACT_NODE, END_NODE]]:
    logger.info(f"REFLECT NODE - Step {state['counter'] + 1}")
    if state["counter"] >= 3:
        state['end_reason'] = "Max iterations reached"
        return Command(update=state, goto=END_NODE)
    reflect_llm = get_llm()
    reflect_llm = reflect_llm.with_structured_output(ProcessedQuery)
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", """
      You're a helpful assistant that reflects on the previous actions taken to answer a user's query. You get the original user query and the answer from similarity fetch from database to anser this query.
      Based on this, you need to decide if the query has been answered precisely in fetched data. If information in fetched data is not enough to answer the query, you should modify the query to be more precise for the next action.
      If not, you should modify the query to be more precise for the next action asking about information that you think is not covered in answer. Think step by ste pin order to resolve this.."""),
        ("human", "{input}"),
        ("assistant", "{fetched_data}")
    ])
    chain = reflection_prompt | reflect_llm
    processed_query: ProcessedQuery = chain.invoke({'input': str(state['input_query'][0]), 'fetched_data': "".join(state['fetched_data'])})
    processed_query.changed_query += f" ticker is {state['ticker']}"
    if processed_query.is_query_answered:
        state['end_reason'] = "Query answered precisely"
        return Command(update=state, goto=END_NODE)
    else:
        state['input_query'].append(processed_query.changed_query)
        return Command(update=state, goto=ACT_NODE)

def end_summary_node(state: GraphReflectionState) -> GraphReflectionState:
    logger.info(f"SUMMARY NODE - Step {state['counter'] + 1}")
    fetched_data = state['fetched_data']
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
      You're a helpful assistant that summarizes relevant information from financial report for a user's query. You get the original user query and a list of relevant fetch data from database.
      Based on this, you need to provide a concise summary of the most relevant information from the data that answers the user's query.
      """),
        ("human", "USER QUERY: {input} \n\n RELEVANT DATA FROM REPORT: {data}")
    ])
    llm = get_llm()
    chain = summary_prompt | llm
    data = "\n".join(fetched_data)
    summary = chain.invoke({"input": str(state['input_query']), "data": data})
    state['summary'] = summary.content
    state['end_reason'] = f"Summary of relevant data provided"
    return state

graph = StateGraph(GraphReflectionState)
graph.add_node(ACT_NODE, act_node)
graph.add_node(REFLECT_NODE, reflect_node)
graph.add_node(END_NODE, end_summary_node)
graph.set_entry_point(ACT_NODE)
graph.set_finish_point(END_NODE)
graph.add_edge(ACT_NODE, REFLECT_NODE)
app = graph.compile(debug=True)
