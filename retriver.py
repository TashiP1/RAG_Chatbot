from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os

# Initialize the embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Directory where the Chroma DB will store the data (choose a valid path on your system)
persist_directory = "chroma_db"

# Create or load the Chroma vector store
collection_name = "govtechdata"  # Name your collection
# Note: If the directory already exists, it will load the existing collection; otherwise, it will create a new one.
chroma_db = Chroma(
    persist_directory=persist_directory,  # Directory to store Chroma DB
    embedding_function=embeddings,  # Use the SentenceTransformer embeddings
    collection_name=collection_name
)

# To ensure that Chroma DB is initialized or loaded correctly, you can print the db object
print(chroma_db)
print("######")

# Perform a similarity search in Chroma DB
query = "What is EOL?"

# Perform similarity search with a specified number of results (k=2)
docs = chroma_db.similarity_search_with_score(query=query, k=2)

# Print the results
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
