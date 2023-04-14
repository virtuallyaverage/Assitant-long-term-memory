from Assistant import assistant

milvus_host = "localhost"
milvus_port = "19530"
model_name = "paraphrase-distilroberta-base-v2"

assistant = assistant.ChatGPTAssistant(milvus_host, milvus_port, model_name)
assistant.run(wake_words=["computer", "jarvis"])