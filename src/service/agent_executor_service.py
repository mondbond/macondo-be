import json

from src.util.logger import logger
from pydantic import BaseModel

class ToolCalling(BaseModel):
    tool_name: str

# Main agent executor for tool-calling and iteration
class CustomAgentExecutor():
    def __init__(self, llm, prompt, tools, final_tool, max_iteration=3):
        self.name2tool = {tool.name: tool for tool in tools}
        self.end_tool = final_tool
        self.all_tools = tools + [final_tool]
        self.llm = llm
        self.runnable = (
            prompt
            | self.llm.bind_tools(self.all_tools, tool_choice="auto"))
        self.tools = tools
        self.max_iteration = max_iteration
        self.history = []

    def invoke(self, input):
        iteration = 0
        answer = ""
        agent_scratchpad = []
        while iteration < self.max_iteration:
            iteration += 1
            logger.info("Iteration:", iteration)
            logger.info("SCRATCHPAD:")
            logger.info(agent_scratchpad)
            response = self.runnable.invoke({"input": input["input"], "history": input['history'], "agent_scratchpad": agent_scratchpad})
            logger.info("Response:" + str(response))
            self.history.append(response)
            tool_name = None
            tool_args = None
            try:
                js = json.loads(response.json())
                js = js["tool_calls"][0]
                tool_name = js["name"]
                tool_args = js["args"]
            except Exception as e:
                logger.info("Error parsing JSON or extracting tool call:", e)
                logger.info("Response content:", response.content)
            if tool_name is None:
                try:
                    js = json.loads(response.json())['content']
                    js = json.loads(js)
                    js = js[0]
                    tool_name = js["name"]
                    tool_args = js["arguments"]
                    logger.info("TOOL CALLING INDERECTLY")
                except Exception as e:
                    logger.info("Second attempt failed. Error parsing JSON or extracting tool call:", e)
                    logger.info("Response content:", response.content)
            if tool_name in self.name2tool and tool_name != self.end_tool.name:
                tool_obj = self.name2tool[tool_name]
                output = tool_obj.invoke(tool_args)
                agent_scratchpad.append(output)
                continue
            if tool_name == self.end_tool.name:
                logger.info("Final tool called, stopping iterations.")
                answer = str(tool_args)
                break
            if response.content != "":
                logger.info("Final answer found, stopping iterations.")
                answer = response.content
                break
        logger.info(answer)
        return answer
