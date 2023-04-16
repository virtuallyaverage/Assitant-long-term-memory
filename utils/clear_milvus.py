from pymilvus import Milvus, Collection, FieldSchema, CollectionSchema, DataType, connections

embedding_field_name = "numpy_embeddings"
collection_name = 'conversation_vectors'
client = Milvus(host="localhost", port="19530", alias="default")

embedding_field = FieldSchema(name="embedding", dtype=DataType.INT64, dim=768)
schema = CollectionSchema(fields=[embedding_field], primary_field="embedding")

# Check if the collection exists
for collection_name in client.list_collections():
    try:
        client.load_collection(collection_name=collection_name)
        user_input =  input(f"delete {collection_name}? (y/n)")
        if user_input =='y':
            client.drop_collection(collection_name)
    except:
        print(f"colleciton of name: {collection_name}, could not be loaded")
        print(f"collections {client.list_collections()}")
        user_input =  input(f"delete {collection_name}? (y/n)")
        if user_input =='y':
            client.drop_collection(collection_name)