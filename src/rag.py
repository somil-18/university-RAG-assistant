import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_classic.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.chunk import get_final_chunks


load_dotenv()


# load models
def load_models():
    print("Loading AI Models...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=1024,
        temperature=0.2,
        task="conversational"
    )

    chat = ChatHuggingFace(llm=llm)

    return embeddings, chat


# detect fee queries
def is_fee_query(query: str) -> bool:
    keywords = ["fee", "fees", "tuition", "admission", "cost", "payment"]
    q = query.lower()
    return any(k in q for k in keywords)


# build RAG chain
def build_rag_chain(chat, embeddings):
    print("Building RAG Chain...")

    vector_store = Chroma(
        persist_directory="./vector_store",
        embedding_function=embeddings,
        collection_name="collec_1"
    )

    # multi-query retriever
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm=chat,
        retriever=base_retriever
    )

    # bm25 retriever
    all_docs = get_final_chunks()

    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs = [d for d in all_docs if d.metadata.get("type") == "text"]

    bm25_tables = BM25Retriever.from_documents(table_docs, k=4)
    bm25_texts = BM25Retriever.from_documents(text_docs, k=4)

    # hybrid retriever
    hybrid_text_retriever = EnsembleRetriever(
        retrievers=[multi_query_retriever, bm25_texts],
        weights=[0.6, 0.4]
    )

    # router
    def route_retriever(query: str):
        if is_fee_query(query):
            return bm25_tables.invoke(query)
        return hybrid_text_retriever.invoke(query)

    # prompt
    system_prompt = (
        "You are an intelligent academic assistant for IIT Bombay.\n\n"
        "RULES (MANDATORY):\n"
        "1. Use the provided CONTEXT as the ONLY source for fees, rules, and policies.\n"
        "2. If the context contains a table, you MUST rely on it.\n"
        "3. If a table contains a row labeled 'Total' or 'Grand Total', "
        "you MUST quote that value verbatim.\n"
        "4. DO NOT recompute totals by adding individual components.\n"
        "5. If information is not present, clearly say it is not available.\n"
        "6. Be clear, structured, and factual.\n"
        "7. End the answer with: 'For more information, please visit the official IIT Bombay website.'\n\n"
        "If the answer is based on a table:\n"
        "- Summarize ALL relevant fee components\n"
        "- Include admission fees, per-semester fees, refundable deposits, and grand total if present\n"
        "- Never stop at partial totals unless explicitly asked\n\n"
        "--- CONTEXT ---\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # string parser
    parser = StrOutputParser()

    # chaining
    parallel_chain = RunnableParallel({
            "context": route_retriever,
            "input": RunnablePassthrough()
        })
    
    final_pipeline = parallel_chain | prompt | chat | parser

    return final_pipeline

