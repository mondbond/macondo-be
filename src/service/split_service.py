from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings


def text_to_recursive_splitting(text, chunk_size=300, overlap=60,
    separators=["\n\n", "."]):
  recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                      chunk_overlap=overlap,
                                                      separators=["\n\n", "\n",
                                                                  ".", " ", ""])
  return recursive_splitter.split_text(text)


def text_to_semantic_splitting(text):
  embeddings = HuggingFaceEmbeddings()
  semantic_splitter = SemanticChunker(embeddings=embeddings,
                                      min_chunk_size=2000,
                                      breakpoint_threshold_type="percentile",
                                      breakpoint_threshold_amount=0.5)
  return semantic_splitter.split_text(text)
