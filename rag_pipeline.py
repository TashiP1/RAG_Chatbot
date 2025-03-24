from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.ollama import Ollama
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import os
import json
from langchain.vectorstores import Chroma

app = FastAPI()
view = Jinja2Templates(directory="view")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Ollama connection
ollama_llm = Ollama(
    base_url="http://host.docker.internal:11434",  # Change based on Docker setup
    model="smollm:135m",  # Adjust the model name if needed
)

print("Ollama LLM Initialized....")

# Prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Embedding model configuration
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

collection_name = "govtechdata"  # Name your collection
# Configure Chroma DB as a vector store
chroma_db = Chroma(
    persist_directory="chroma_db",  # Specify your local storage path for the Chroma DB
    embedding_function=embeddings,
    collection_name=collection_name
)

# Chroma DB retriever setup
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return view.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    
    # Set up the RetrievalQA chain with Ollama LLM and Chroma DB as the retriever
    qa = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    
    # Get the response from the chain
    response = qa(query)
    print(response)
    
    # Prepare the response data
    answer = response["result"]
    source_document = response["source_documents"][0].page_content
    doc = response["source_documents"][0].metadata["source"]
    
    response_data = jsonable_encoder(
        json.dumps({"answer": answer, "source_document": source_document, "doc": doc})
    )

    res = Response(response_data)
    return res
