from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def retrieve_docs(query: str):
    try:
        # generate embeddings — must match chunk_store model
        embeddings = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
            truncate="END"
        )

        # connect to Pinecone vectorstore
        vectorstore = PineconeVectorStore(
            index_name="university-rag",
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )

        # fetch top 20 chunks by vector similarity
        chunks = vectorstore.similarity_search(query, k=20)
        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query}'")

        # filter empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]

        if not chunks:
            logger.warning("No chunks retrieved — check if Pinecone index has data")
            return []

        # truncate oversized chunks before reranking to stay under 8192 token limit
        for c in chunks:
            if len(c.page_content) > 6000:
                logger.warning(f"Truncating chunk from {len(c.page_content)} chars for reranker")
                c.page_content = c.page_content[:6000]

        # re-rank to get top 5 most relevant
        reranker = NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
            top_n=5
        )

        reranked_docs = reranker.compress_documents(chunks, query)
        logger.info(f"Reranked to {len(reranked_docs)} final chunks")

        return reranked_docs

    except Exception as e:
        logger.error(f"Failed to retrieve docs: {str(e)}")
        raise


if __name__ == '__main__':
    r = retrieve_docs("What are hostel rules?")
    print(r)