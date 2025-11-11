from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.util.logger import log_time
from src.util.logger import logger
from src.llm.llm_provider import get_llm
from src.util.prompt_manager import prompt_manager
from src.tools.tools import wikipedia_info
from langchain.agents import initialize_agent, AgentExecutor, \
  create_react_agent, AgentType
from src.service.mcp.mcp import mcp_client


@log_time
async def call_agent(query: str, history) -> str:
    tools = []
    llm = get_llm()
    memory = form_chat_history(history)
    chat_history = MessagesPlaceholder(variable_name="chat_history")

    if mcp_client is not None:
        mcp_client_tools = await mcp_client.get_tools()
        logger.info("MCP Client provided tools: " + str(mcp_client_tools))
        tools.extend(mcp_client_tools)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            "memory_prompts": [chat_history],
            "input_variables": ["input", "agent_scratchpad", "chat_history"]
        },
    )
    answer = await agent.ainvoke(query)
    logger.info(answer)
    return answer


def form_chat_history(history):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for role, msg in history:
        if role == "user":
            memory.chat_memory.add_user_message(msg)
        else:
            memory.chat_memory.add_ai_message(msg)
    return memory
