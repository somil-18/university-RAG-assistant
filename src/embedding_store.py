from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from src.chunk import get_final_chunks


load_dotenv()


# embedding model
embd_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# vector store
print("Initializing Vector Store...")
vector_store = Chroma(
    embedding_function=embd_model,
    persist_directory=r'src/vec_store',
    collection_name="collec_1"
)


# load chunks
print("Fetching chunks...")
docs = get_final_chunks()

print(f"Adding {len(docs)} documents to ChromaDB...")
vector_store.add_documents(docs)


print("Success! Embeddings stored correctly.")

