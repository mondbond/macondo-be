from src.service.split_service import text_to_semantic_splitting
from src.db.db import db_client
from src.service.file_format_service import any_format_to_str
from src.util.logger import logger

def save_report(file, metadata, content_type):
  text = any_format_to_str(file, content_type)
  chunks = text_to_semantic_splitting(text)
  # chunks = text_to_recursive_splitting(text)
  logger.info(f"Total chunks created: {len(chunks)}")

  db_client.add_new_report(chunks, metadata)

def save_text_report(text, metadata):
  chunks = text_to_semantic_splitting(text)
  # chunks = text_to_recursive_splitting(text)
  logger.info(f"Total chunks created: {len(chunks)}")

  db_client.add_new_report(chunks, metadata)

def delete_report(ticker):
  db_client.delete_report(ticker)

def get_report_list():
    return db_client.get_existing_reports()
