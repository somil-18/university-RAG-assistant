from langchain_core.documents import Document
from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
import json


def get_final_chunks():
    with open("parsed_data.json", encoding="utf-8") as f:
        data = json.load(f)

    text_docs = []
    table_docs = []

    # create Document objects
    for d in data["texts"]:
        text_docs.append(
            Document(
                page_content=d["content"],
                metadata={"source": d["source"], "type": "text"}
            )
        )

    for d in data["tables"]:
        table_docs.append(
            Document(
                page_content=d["content"],
                metadata={"source": d["source"], "type": "table"}
            )
        )

    print(f"Loaded {len(text_docs)} text docs")
    print(f"Loaded {len(table_docs)} table docs")

    # Split ONLY TEXT docs
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    header_splits = []

    # mardown text splitter
    print("Splitting text docs by headers...")
    for doc in text_docs:
        splits = markdown_splitter.split_text(doc.page_content)

        # preserving original metadata
        for s in splits:
            s.metadata.update(doc.metadata)

        header_splits.extend(splits)

    # recursive character text splitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    final_text_chunks = recursive_splitter.split_documents(header_splits)

    print(f"Text chunks created: {len(final_text_chunks)}")


    # tables are kept as it is
    print(f"Table chunks kept intact: {len(table_docs)}")


    # final chunks
    final_chunks = final_text_chunks + table_docs

    print(f"TOTAL chunks returned: {len(final_chunks)}")

    return final_chunks


if __name__ == "__main__":
    print('this statement will only executed when chunk.py is run directly')
    chunks = get_final_chunks()

