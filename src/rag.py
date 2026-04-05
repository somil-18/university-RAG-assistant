from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import os

from src.retriever import retrieve_docs


load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_response(query: str):
    try:
        # retrieve relevant docs
        relevant_docs = retrieve_docs(query)

        context = '\n\n'.join([doc.page_content for doc in relevant_docs])

        prompt = ChatPromptTemplate.from_template("""
You are a helpful university assistant for IIT students. Answer the question using ONLY the context provided below.

Rules:
- If the answer is not in the context, say "I don't have that information."
- If the context contains a table, use it to give precise structured answers.
- Keep your answer concise and to the point.

Context:
{context}

Question:
{question}

Answer:
""")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        chain = prompt | llm

        # stream response
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk.content

    except Exception as e:
        logging.error(f"Failed to generate RAG response: {str(e)}")
        raise


if __name__ == '__main__':
    for chunk in generate_response(
        query="Tell me 2 most imp. hostel rules"
    ):
        print(chunk, end="", flush=True)

