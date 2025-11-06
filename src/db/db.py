import abc
import tempfile
from abc import abstractmethod
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from uuid import uuid4

from langchain_experimental.open_clip import OpenCLIPEmbeddings

from chromadb import Client
from src.llm.llm_provider import get_llm
from src.util.prompt_manager import prompt_manager
from src.util.logger import logger

# normal version
# clip_embedder = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

# version for light aws deployment
clip_embedder = OpenCLIPEmbeddings(model_name="ViT-B/32", checkpoint="laion2b_s34b_b79k")

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

    @abstractmethod
    def store_image_itself(self, image_path: str, metadata: dict = None):
      pass

    @abstractmethod
    def store_image_embedding(self, file_bytes, uri:str, metadata: dict = None):
      pass

    @abstractmethod
    def search_image(self, query: str, top_k=3):
      pass

    @abstractmethod
    def search_image_embedd(self, query: str, top_k=3):
      pass

class InMemoryFinReportVectorDBReport(FinReportVectorDB):

    def __init__(self):
        transformer_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.report_db = Chroma(
            collection_name="report",
            embedding_function=transformer_fn)

        self.image_db = Chroma(
            collection_name="image",
            embedding_function=clip_embedder)

        self.embed_image_db = Client().get_or_create_collection("embed_image")

    def get_existing_reports(self):
      results = self.report_db.get(include=["metadatas"])
      uniques = {m.get('ticker') for m in results["metadatas"] if m}
      return uniques

    def get_rephrased_retriever(self, type=None, ticker=None):
      template = PromptTemplate(
          template=prompt_manager.get_prompt('rephrase_question_for_similarity_search'),
          input_variables=["question"])
      base_retriever = self.report_db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "filter": {"ticker": ticker}})

      retriever = RePhraseQueryRetriever.from_llm(retriever=base_retriever, llm = get_llm(), prompt=template)

      return retriever

    def get_base_retriever(self, type=None, ticker=None):
      base_retriever = self.report_db.as_retriever(search_type="similarity", search_kwargs={"k": 4, "filter": {"ticker": ticker}})
      return base_retriever

    def search_report_context(self, ticker, query):
      template = PromptTemplate(
          template=prompt_manager.get_prompt('rephrase_question_for_similarity_search'),
          input_variables=["question"])
      base_retriever = self.report_db.as_retriever(search_type="similarity", search_kwargs={"k": 5}, filter={"ticker": ticker})
      retriever = RePhraseQueryRetriever.from_llm(retriever=base_retriever, llm = get_llm(), prompt=template)
      results = retriever.invoke(query)
      return results

    def add_new_report(self, documents, metadata):
      self.report_db.add_texts(documents,
                               metadatas=[{"ticker": metadata["ticker"]} for _ in range(len(documents))],
                               ids=[str(metadata["ticker"] + "_" + metadata["date"] + str(i)) for i in range(len(documents))]
                               )

    def store_image_itself(self, image_path: str, metadata: dict = None):
      id = self.image_db.add_images(
          uris = [image_path],
          metadatas=[metadata])
      return logger.info(f"Image saved with id = {id}")

    def store_image_embedding(self, file_bytes, uri:str, metadata: dict = None):
      id_doc = str(uuid4())

      image_embeddings = None

      with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        image_embeddings = clip_embedder.embed_image([tmp.name])[0]

      # image_embeddings = clip_embedder.embed_image([image_path])
      id = self.embed_image_db.add(
          ids=[id_doc],
          embeddings=image_embeddings,
          # documents=[image_path],          # just store the path, not the image bytes
          metadatas=[metadata],
          uris=[uri]
      )
      return logger.info(f"Image saved with id = {id}")

    def search_image(self, query: str, top_k=3):
      results = self.image_db.search(query=query, search_type="similarity")
      return logger.info(f"Search image by query {query} result {results}")

    def search_image_embedd(self, query: str, top_k=1):
      embedded_query = clip_embedder.embed_documents([query])[0]
      results = self.embed_image_db.query(
          query_embeddings=[embedded_query],
          n_results=top_k
      )
      logger.info(f"Search image by query {query} result {results}")
      return results

    def delete_report(self, ticker):
      self.report_db.delete(where={"ticker": ticker})

      collection = self.report_db._collection
      logger.info("Total records:", collection.count())


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


if __name__ == "__main__":
  # db_client.store_image_itself("/Users/ibahr/personal/dnd/dnd/characters/img/cover.jpg", {"description": "Sample Image"})
  # db_client.search_image("a cover image with fantasy art", top_k=2)

  db_client.store_image_embedding("/Users/ibahr/personal/dnd/dnd/characters/img/cover.jpg", {"description": "Sample Image"})
  db_client.search_image_embedd("a cover image with fantasy art", top_k=2)

