from Assistant import assistant
from dotenv import load_dotenv
import os

load_dotenv("E:\coding\AI\memory-Assitant\Assitant-long-term-memory\.env", verbose=True)

tokenizer_model_name = "paraphrase-distilroberta-base-v2"

keys = {
    'porcupine': os.getenv('PORCUPINE_KEY'),
    'openai_api': os.getenv('OPENAI_API_KEY')
}
print(f"Keys: {keys}")
print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")

os.environ["OPENAI_API_KEY"] = keys['openai_api']

print("###   starting service ###")
assistant = assistant.ChatGPTAssistant(tokenizer_model_name, keys)
print("### running loop ###")
assistant.run()