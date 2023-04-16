import numpy as np
from sentence_transformers import SentenceTransformer
from milvus import Milvus, IndexType, MetricType, Status
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sklearn.metrics.pairwise import cosine_similarity

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    conversation_data = Column(String)

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    conversations = relationship('Conversation', backref='user')

class LongTermMemory:
    def __init__(self):
        self.setup_memory_storage()
        self.vectorizer = SentenceTransformer('bert-base-nli-mean-tokens')
        self.milvus_client = Milvus(host='localhost', port='19530')

    def setup_memory_storage(self):
        self.engine = create_engine('sqlite:///memory.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def retrieve_past_conversations(self, user_identifier):
        try:
            past_conversations = self.session.query(Conversation).filter_by(user_id=user_identifier).all()
            return [conv.conversation_data for conv in past_conversations]
        except Exception as e:
            print("Error:", e)
            return []

    def store_conversation(self, user_identifier, conversation_data):
        try:
            new_conversation = Conversation(user_id=user_identifier, conversation_data=conversation_data)
            self.session.add(new_conversation)
            self.session.commit()
        except Exception as e:
            print("Error:", e)

    def find_similar_conversations(self, user_identifier, input_text, top_n=5):
        past_conversations = self.retrieve_past_conversations(user_identifier)
        if not past_conversations:
            return []

        input_vector = self.vectorizer.encode([input_text])[0]
        past_vectors = self.vectorizer.encode(past_conversations)
        similarities = cosine_similarity([input_vector], past_vectors)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return [past_conversations[i] for i in top_indices]

    def close(self):
        self.session.close()

