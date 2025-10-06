from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.llms import Ollama
from langchain_core.prompts import HumanMessagePromptTemplate

from trulens.core import TruSession, Feedback
from trulens.apps.langchain import TruChain
from trulens.providers.litellm import LiteLLM
from pydantic import Field, BaseModel


import litellm

from src.llm.llm_provider import get_llm

class Score(BaseModel):
  score: float = Field(description="A relevance score between 0 and 1. can be double. never poot 1 if it's not a perfaect", ge=0, le=1)

criteria_prompt = SystemMessagePromptTemplate.from_template("""
You are an evaluator. Assess the following response for relevance to the user question.
Relevance score is a float number between 0 and 1, where 1 means the response is perfectly relevant to the question, and 0 means it is completely irrelevant. 0.5 is somehow relevant and 0.7 is quite relevant but not perfect.
EXAMPLE OF OUTPUT:
0.8
or
0.5
or 0.0 if completely irrelevant
no more text in answer - only the score. Output should be structural and parsed
""")

# 2. Create human prompt with user question and model answer
human_prompt = HumanMessagePromptTemplate.from_template("""
User question: {question}
Model answer: {answer}
""")

chat_prompt = ChatPromptTemplate.from_messages([criteria_prompt, human_prompt])

# 3. LLM setup
llm = get_llm()  # Or Ollama/any other model

criteria_chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)

session = TruSession()
session.reset_database()

ollama = Ollama(base_url="http://localhost:11434", model="mistral:instruct")

full_prompt = SystemMessagePromptTemplate.from_template(
    "Here is the user question you need to answer: {input}")

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

chain = LLMChain(llm=ollama, prompt=chat_prompt_template, verbose=True)


litellm.set_verbose = True
ollama_provider = LiteLLM(
    model_engine="ollama/mistral:instruct",
    api_base="http://localhost:11434",
)



def custom_relevance(prompt: str, response: str) -> str:
  # build your own evaluation prompt
  # eval_prompt = f"""
  #   The following user input was given:
  #   {prompt}
  #
  #   The model answered:
  #   {response}
  #
  #   On a scale of 0 to 1, how well does the answer address the question?
  #   Answer with a single number only. The score can be double like 0.7. if it's not perfect - put 0.9
  #   """
  # # call the LLM provider directly
  response = str(criteria_chain.invoke({"question": prompt, "answer": response})['text']).strip()
  return str(response)

relevance = Feedback(custom_relevance).on_input_output()

tru_recorder = TruChain(
    chain,
    app_name="Chain1_ChatApplication",
    feedbacks=[relevance]
)

# --- Run and record ---
with tru_recorder as recording:
  llm_response = chain({"input": "What is a good name for a store that sells colorful socks?"})
  llm_response = chain({"input": "Can it be true that the sky is green?"})
  llm_response = chain({"input": "Who was the mayor of Paris in the year 1234?"})

print("LLM response:", llm_response)

# --- Explore results ---
records, feedback = session.get_records_and_feedback()

print("Recorded feedback:", records)
print("Recorded feedback:", feedback)

import time

time.sleep(20)
# --- (Optional) Launch dashboard ---
from trulens.dashboard import run_dashboard
run_dashboard(session)
