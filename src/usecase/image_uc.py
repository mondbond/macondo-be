from src.db.db import db_client
from src.util.logger import logger

def save_image_embeddings(file_bytes, metadata: dict, link: str):
  db_client.store_image_embedding(file_bytes, uri=link, metadata=metadata)

def search_image_embeddings_link(query: str):
  result = db_client.search_image_embedd(query)
  logger.info(f"Image search result: {result}")
  metadatas = result.get('metadatas')
  if metadatas and metadatas[0] and 'link' in metadatas[0][0]:
    return str(metadatas[0][0]['link'])
  return "No image found"
