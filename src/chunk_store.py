from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging
import os


load_dotenv()


# basic logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# check tables
def is_table(text):
    return "<table" in text.lower()


def chunk_store_in_vectordb():
    try:
        # read the parsed markdown file
        with open('parsed_data.md', encoding='utf-8') as f:
            markdown_text = f.read()

        # 1st Splitter
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        md_chunks = markdown_splitter.split_text(markdown_text)
        logger.info(f"Split into {len(md_chunks)} markdown chunks")

        # debug — log oversized chunks before processing
        for i, chunk in enumerate(md_chunks):
            if len(chunk.page_content) > 30000:
                logger.warning(f"Chunk {i} is {len(chunk.page_content)} chars | table={is_table(chunk.page_content)} | source={chunk.metadata.get('Header 1', '?')}")

        # 2nd Splitter
        # 30000 chars is safely under Pinecone's 40KB metadata limit
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000,
            chunk_overlap=200
        )

        final_chunks = []
        for chunk in md_chunks:
            if is_table(chunk.page_content):
                if len(chunk.page_content) > 30000:
                    logger.warning(f"Oversized table in '{chunk.metadata.get('Header 1', 'unknown')}' ({len(chunk.page_content)} chars) — truncating to 30000 chars")
                    chunk.page_content = chunk.page_content[:30000] # truncate to fit limit
                final_chunks.append(chunk)

            elif len(chunk.page_content) > 30000:
                split = text_splitter.split_documents([chunk])
                final_chunks.extend(split)

            else:
                final_chunks.append(chunk)

        logger.info(f"After size splitting: {len(final_chunks)} chunks")

        # keep old metadata
        for chunk in final_chunks:
            chunk.metadata = {
                "source": chunk.metadata.get("Header 1", "unknown")
            }

        # generate emmbeddings
        embeddings = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
            truncate="END" 
        )

        index_name = 'university-rag'

        # upload chunks
        logger.info(f"Uploading {len(final_chunks)} chunks to Pinecone index '{index_name}'...")
        PineconeVectorStore.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            index_name=index_name
        )

        logger.info("Upload completed!")
        return len(final_chunks)

    except FileNotFoundError:
        logger.error("parsed_data.md not found — run the parser first")
        raise

    except Exception as e:
        logger.error(f"Failed to process and upload vectors: {str(e)}")
        raise

if __name__ == '__main__':
    r = chunk_store_in_vectordb()
    print(r)
