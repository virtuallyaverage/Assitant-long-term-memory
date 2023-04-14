from Assistant import assistant
from dotenv import load_dotenv
import os

load_dotenv()

milvus_host = "localhost"
milvus_port = "19530"
tokenizer_model_name = "paraphrase-distilroberta-base-v2"

keys = {
    'porcupine': os.getenv('PORCUPINE_KEY'),
    'openai_api': os.getenv('OPENAI_API_KEY')
}

print("###   starting service ###")
assistant = assistant.ChatGPTAssistant(milvus_host, milvus_port, tokenizer_model_name, keys)
print("### running loop ###")
assistant.run(wake_words=["computer", "jarvis"])