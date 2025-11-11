from langchain_core.prompts import PromptTemplate
from src.util.prompt_manager import prompt_manager
from src.db.db import db_client
from src.llm.llm_provider import get_llm
from langchain.chains import create_retrieval_chain

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
