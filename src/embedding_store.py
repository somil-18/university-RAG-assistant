from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from src.chunk import get_final_chunks

from dotenv import load_dotenv
load_dotenv()

# embedding model from HF
embd_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

print("Initializing Vector Store...")
vector_store = Chroma(
    embedding_function=embd_model,
    persist_directory='./vector_store', 
    collection_name='collec_1'
)

print("Fetching chunks...")
docs = get_final_chunks() # calling function for chunking

print(f"Adding {len(docs)} documents to ChromaDB...")
vector_store.add_documents(documents=docs) # adding vector into the chroma vector store

print("Success! Embeddings stored")
