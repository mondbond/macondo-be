from langchain_core.prompts import PromptTemplate
from src.util.prompt_manager import prompt_manager
from src.db.db import db_client
from src.llm.llm_provider import get_llm
from langchain.chains import create_retrieval_chain

from src.service.file_format_service import soup_html_to_text
from src.usecase.report_uc import save_text_report
from src.util.logger import logger

def base_query_report_question_answer(ticker: str, query: str, join=True):
  retrieval = db_client.get_base_retriever(type="similarity", ticker=ticker)
  result = retrieval.get_relevant_documents(query)

  if join:
    all_text = "\n".join(doc.page_content for doc in result)
    return all_text
  else:
    return result

def base_query_report_question_answer_full_state(ticker: str, query: str):
  retrieval = db_client.get_base_retriever(type="similarity", ticker=ticker)
  result = retrieval.get_relevant_documents(query)
  all_text = "\n".join(doc.page_content for doc in result)

  question_answer_prompt = PromptTemplate(template=prompt_manager.get_prompt('context_based_answer'), input_variables=["input", "context"])
  chain = question_answer_prompt | get_llm()

  answer = chain.invoke({"input": query, "context": all_text})

  return {"question": query, "context": all_text, "answer": answer.content}

def report_rephrase_retriever_search(ticker: str, query: str, context: str = None):
  retrieval = db_client.get_base_retriever(type="similarity", ticker=ticker)
  result = retrieval.get_relevant_documents(query)
  all_text = "\n".join(doc.page_content for doc in result)
  return {'answer': all_text}


# todo eliminate old way
# def report_rephrase_retriever_search(ticker: str, query: str, context: str = None):
#     retrieval = db_client.get_rephrased_retriever(type="similarity", ticker=ticker)
#     prompt = PromptTemplate(template="You are a financial consultant. You are provided with context and using them you need to answer the question. Question: {input}\n Context: {context}\n\n\n Answer:", input_variables=["input", "context"])
#     qa_chain = create_stuff_documents_chain(llm=get_llm(), document_variable_name="context", prompt =prompt)
#     chain = create_retrieval_chain(retrieval, qa_chain)
#     result = chain.invoke({"input": query})
#     return result


if __name__ == "__main__":
  # with open("/Users/ibahr/Downloads/synthetic_report.pdf", "rb") as f:
  with open("/Users/ibahr/Desktop/reports/UBER.html", "rb") as f:
    pdf_content = f.read()
    report = soup_html_to_text(pdf_content)

    metadata = {
      "ticker": "UBER",
      "date": "2025-07-17"
    }

  save_text_report(report, metadata)
  out = base_query_report_question_answer_full_state(query="What is the main competitor of UBER by it's report?", ticker="UBER")
  logger.info(out['answer'])
