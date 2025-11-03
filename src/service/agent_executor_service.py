import json

from langchain_core.prompts import SystemMessagePromptTemplate, \
  HumanMessagePromptTemplate
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from src.llm.llm_provider import get_llm
from src.tools.tools import wikipedia_info, final_result
from src.util.logger import logger

class ToolCalling(BaseModel):
  tool_name: str

class CustomAgentExecutor():
    def __init__(self, llm,  prompt, tools, final_tool, max_iteration = 3):
      self.name2tool = {tool.name: tool for tool in tools}
      self.end_tool = final_tool

      self.all_tools = tools + [final_tool]
      self.llm = llm
      self.runnable = (
          # {
          #   "input": lambda x: x["input"],
          #   "agent_scratchpad": lambda x: x["agent_scratchpad"],
          # }
          # |
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
            response = self.runnable.invoke({"input": input["input"], "history" : input['history'], "agent_scratchpad": agent_scratchpad})
            logger.info("Response:" + str(response))

            self.history.append(response)

            tool_name = None
            tool_args = None

            try :
              js = json.loads(response.json())

              js = js["tool_calls"][0]
              tool_name = js["name"]
              tool_args = js["args"]
            except Exception as e:
              logger.info("Error parsing JSON or extracting tool call:", e)
              logger.info("Response content:", response.content)

            if tool_name is None:
              try :
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

            # Check if the response indicates that we should stop
            if response.content != "":
              logger.info("Final answer found, stopping iterations.")
              answer = response.content
              break
            # Retrieve the tool object and invoke it properly

        logger.info(answer)
        return answer


if __name__ == "__main__":

  llm = get_llm()
  agent = CustomAgentExecutor(llm, ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You're a helpful assistant. When answering a user's question 
        you should first use one of the tools provided to obtain more information for user answer. Answer from the tools you will get in the agent_scratchpad.
        Then based on the tool form the response for the user. You MUST:
        0. Always return tool calls using the structured tool_calls format, not JSON in content.
        1. If you want to give the user a final answer, call the final_result tool.
        
        HERE IS YOUR SCRATCHPAD WHERE PREVIOUS TOOL CALLING RESULTS ARE: {agent_scratchpad}
        """),
    HumanMessagePromptTemplate.from_template("Here is the user question you need to call final_result tool in tool_calls property: {input}. ")
  ]), [wikipedia_info], final_result,  max_iteration=2)

  agent.invoke({"input": "Hello, my name is Ivan.", "agent_scratchpad": []})
