import sqlite3
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema

class LongTermMemory:
    def __init__(self):
        self.setup_memory_storage()
        self.vectorizer = SentenceTransformer('bert-base-nli-mean-tokens')
        self.setup_milvus_connection()

    def setup_memory_storage(self):
        self.conn = sqlite3.connect('data/memory.db')
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            conversation_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''')

    def setup_milvus_connection(self):
        connections.connect()
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields=fields, description="Conversations vector storage", auto_id=False)
        vector_index = {"index_type": "IVF_SQ8", "metric_type": "COSINE", "params": {"nlist": 100}}
        self.collection = Collection(name="conversations", schema=schema)
        self.collection.create_index(field_name="vector", index_params=vector_index)

    def store_conversation(self, user_identifier, conversation_data):
        try:
            self.cursor.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_identifier,))
            self.cursor.execute("INSERT INTO conversations (user_id, conversation_data) VALUES (?, ?)", (user_identifier, conversation_data))
            self.conn.commit()

            conversation_id = self.cursor.lastrowid
            conversation_vector = self.vectorizer.encode([conversation_data])[0]
            self.collection.insert([{"id": conversation_id, "vector": conversation_vector.tolist()}])
        except Exception as e:
            print("Error:", e)

    def find_similar_conversations(self, user_identifier, input_text, top_n=5):
        input_vector = self.vectorizer.encode([input_text])[0]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search([input_vector.tolist()], anns_field="vector", top_k=top_n, search_params=search_params)

        top_indices = [hit.id for hit in results[0]]
        self.cursor.execute("SELECT conversation_data FROM conversations WHERE id IN ({})".format(','.join('?' * len(top_indices))), top_indices)
        similar_conversations = [row[0] for row in self.cursor.fetchall()]

        return similar_conversations

    def close(self):
        self.conn.close()
        connections.disconnect()

