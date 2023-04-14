from typing import List
import sqlite3
import numpy as np
from pymilvus import Milvus, IndexType, MetricType, Status

class ConversationStorage:
    def __init__(self, model):
        self.model = model
        self.embedding_field_name = "numpy_embeddings"
        
        # Initialize SQLite
        self.conn = sqlite3.connect("data/conversation_lookup.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_lookup (conversation_id INTEGER PRIMARY KEY, conversation_history TEXT)
        """)
        
        # Initialize Milvus
        self.collection = "conversation_collection"
        self.milvus = Milvus(host="localhost", port="19530")
        status, ok = self.milvus.has_collection(self.collection)
        if not ok:
            param = {
                'collection_name': self.collection,
                'dimension': model.get_dimension(),
                'index_file_size': 1024,
                'metric_type': MetricType.IP
            }
            status = self.milvus.create_collection(param)

    def store_conversation(self, conversation: List[str]):
        print("storing the conversation")
        embedded_conversation = self.model.encode(conversation)
        numpy_embeddings = np.array(embedded_conversation)
        
        status, ids = self.milvus.insert(collection_name=self.collection, records=numpy_embeddings)
        if not status.OK():
            raise Exception(f"Failed to insert embeddings into Milvus: {status}")

        conversation_id = ids[0]
        conversation_history = " ".join(conversation)
        
        self.cursor.execute("""
        INSERT INTO conversation_lookup (conversation_id, conversation_history) VALUES (?, ?)
        """, (conversation_id, conversation_history))
        self.conn.commit()
        return conversation_id

    def search_conversations(self, query: str, top_k: int = 5):
        embedded_query = self.model.encode([query])
        numpy_query = np.array(embedded_query)

        search_params = {"nprobe": 16}
        status, results = self.milvus.search(
            collection_name=self.collection,
            query_records=numpy_query,
            top_k=top_k,
            params=search_params
        )
        if not status.OK():
            raise Exception(f"Failed to search embeddings in Milvus: {status}")

        similar_conversation_ids = [result.id for result in results[0]]
        similar_conversations = []

        for conversation_id in similar_conversation_ids:
            self.cursor.execute("""
            SELECT conversation_history FROM conversation_lookup WHERE conversation_id = ?
            """, (conversation_id,))
            conversation_history = self.cursor.fetchone()[0]
            similar_conversations.append(conversation_history)

        return similar_conversations