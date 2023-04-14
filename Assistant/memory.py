from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from sentence_transformers import SentenceTransformer
from typing import List

class MilvusAssistant:
    def __init__(self, milvus_host: str, milvus_port: str, model_name: str):
        connections.connect(host=milvus_host, port=milvus_port)
        self.model = SentenceTransformer(model_name)
        self.collection_name = "conversation_vectors"
        self.embedding_field_name = "embedding"
        self.dim = self.model.get_sentence_embedding_dimension()
        self.collection = self._create_collection()
    
    def _create_collection(self):
        collection_schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.embedding_field_name, dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ],
            description="Conversation embeddings collection"
        )
        collection = Collection(name=self.collection_name, schema=collection_schema)
        return collection

    def store_conversation(self, conversation: List[str]):
        embedded_conversation = self.model.encode(conversation)
        numpy_embeddings = utility.list_to_numpy(embedded_conversation)
        self.collection.insert({self.embedding_field_name: numpy_embeddings})
    
    def get_relevant_info(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(query_embedding, self.embedding_field_name, search_params, top_k)
        relevant_ids = [result.id for result in results[0]]
        return relevant_ids
