import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.chunk import get_final_chunks

load_dotenv()

def load_models():
    print("Loading AI Models...")
    
    # 1. Embedding Model
    embd_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    
    # 2. LLM 
    llm = HuggingFaceEndpoint(
        repo_id='Qwen/Qwen2.5-7B-Instruct',
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
        max_new_tokens=1024,
        temperature=0.2,
        task="conversational",
        stop_sequences=["<|endoftext|>", "<|im_end|>"]
    )
    chat = ChatHuggingFace(llm=llm)
    
    return embd_model, chat

def build_rag_chain(chat, embd_model):
    print("Building RAG Chain...")
    
    # 1. Vector Store Connection
    vector_store = Chroma(
        persist_directory=r'./vector_store', 
        embedding_function=embd_model,
        collection_name='collec_1'
    )
    
    # 2. Retrievers

    # Multi-Query Retriever
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm=chat, 
        retriever=base_retriever
    ) 
    # BM25 Retriever
    raw_docs = get_final_chunks()
    bm25_retriever = BM25Retriever.from_documents(raw_docs, k=4) 
    # Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[multi_query_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    # 3. Prompt
    system_prompt = (
    "You are an expert academic assistant for IIT Bombay. "
    "Your task is to answer student questions using ONLY the provided context snippets below. "
    "\n\n"
    "--- GUIDELINES ---\n"
    "1. **Strict Fidelity:** If the answer is not in the context, say 'I cannot find that information in the official documents.' Do not make up numbers.\n"
    "2. **Table Logic:** The context contains Markdown tables. When asked for fees or stats, read the row and column headers carefully to find the intersection.\n"
    "3. **Tone:** Be precise, professional, and helpful. Format lists and numbers clearly.\n"
    "4. **Citations:** If possible, mention the document source (e.g., 'According to the Fee Circular...').\n"
    "5. **Disclaimer:** Always mention at the end: 'For more info visit official site'."
    "\n\n"
    "--- CONTEXT ---\n"
    "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{input}")
    ])
    
    # 4. Parser
    parser = StrOutputParser()

    # 5. Chain
    parallel_chain = RunnableParallel({
            'context': ensemble_retriever,
            'input': RunnablePassthrough()
        })
    main_pipeline = parallel_chain | prompt | chat | parser


    return main_pipeline

