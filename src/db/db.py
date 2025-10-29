import abc
from abc import abstractmethod
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from src.llm.llm_provider import get_llm
from src.util.prompt_manager import prompt_manager

class FinReportVectorDB(abc.ABC):

    @abstractmethod
    def get_existing_reports(self):
        pass

    @abstractmethod
    def search_report_context(self, ticker, query):
        pass

    @abstractmethod
    def get_rephrased_retriever(self, type=None, ticker=None):
        pass

    @abstractmethod
    def get_base_retriever(self, type=None, ticker=None):
      pass

    @abstractmethod
    def add_new_report(self, embed, metadata):
        pass

    @abstractmethod
    def delete_report(self, ticker):
      pass

class InMemoryFinReportVectorDBReport(FinReportVectorDB):

    def __init__(self):
        transformer_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.db = Chroma(
            collection_name="report",
            embedding_function=transformer_fn)

    def get_existing_reports(self):
      results = self.db.get(include=["metadatas"])
      uniques = {m.get('ticker') for m in results["metadatas"] if m}
      return uniques

    def get_rephrased_retriever(self, type=None, ticker=None):
      template = PromptTemplate(
          template=prompt_manager.get_prompt('rephrase_question_for_similarity_search'),
          input_variables=["question"])
      base_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "filter": {"ticker": ticker}})

      retriever = RePhraseQueryRetriever.from_llm(retriever=base_retriever, llm = get_llm(), prompt=template)

      return retriever

    def get_base_retriever(self, type=None, ticker=None):
      base_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 4, "filter": {"ticker": ticker}})
      return base_retriever

    def search_report_context(self, ticker, query):
      template = PromptTemplate(
          template=prompt_manager.get_prompt('rephrase_question_for_similarity_search'),
          input_variables=["question"])
      base_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5}, filter={"ticker": ticker})
      retriever = RePhraseQueryRetriever.from_llm(retriever=base_retriever, llm = get_llm(), prompt=template)
      results = retriever.invoke(query)
      return results

    def add_new_report(self, documents, metadata):
      self.db.add_texts(documents,
          metadatas=[{"ticker": metadata["ticker"]} for _ in range(len(documents))],
          ids=[str(metadata["ticker"] + "_" + metadata["date"] + str(i)) for i in range(len(documents))]
      )

      collection = self.db._collection
      print("Total records:", collection.count())

    def delete_report(self, ticker):
      self.db.delete(where={"ticker": ticker})

      collection = self.db._collection
      print("Total records:", collection.count())


class VectorDBResolver:
  _instance = None

  def __init__(self, source):
    self.source = source

  def resolve_db_source(self):
    if VectorDBResolver._instance is None:
      if self.source == 'inmemory':
        VectorDBResolver._instance = InMemoryFinReportVectorDBReport()
      else:
        raise Exception("No DB resolver found")
    return VectorDBResolver._instance

resolver = VectorDBResolver("inmemory")
db_client = resolver.resolve_db_source()
