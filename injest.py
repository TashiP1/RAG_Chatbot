from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF documents
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Define and create a Chroma collection
collection_name = "govtechdata"  # Name your collection
vector_store = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings, 
    persist_directory="chroma_db",  # Directory to store Chroma data
    collection_name=collection_name  # Define the collection name
)

# Persist the database
vector_store.persist()

print(f"Chroma Collection '{collection_name}' Successfully Created!")
