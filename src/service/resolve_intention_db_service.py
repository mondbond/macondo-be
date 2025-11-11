from src.db.db import db_client

def get_user_intention_with_similarity_search(query: str):
    return db_client.define_route(query)
