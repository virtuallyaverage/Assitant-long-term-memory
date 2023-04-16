from pymilvus import Collection, DataType, connections

# Get an existing collection.
connect = connections.connect(host='localhost', port='19530', alias="default")
collection = Collection("numpy_embeddings")

# Retrieve the schema of the collection.
collection_schema = collection.schema

# Print the schema.
print(collection_schema)

# Get the list of fields in the schema.
fields = collection_schema.fields

# Print information about each field.
for field in fields:
    print(f"Field name: {field.name}")
    print(f"Data type: {field.dtype}")
    print(f"Description: {field.description}")
    if field.dtype == DataType.FLOAT_VECTOR:
        print(f"Dimension: {field.dim}")
    print()