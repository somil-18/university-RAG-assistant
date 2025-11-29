from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

import json

def get_final_chunks():
    # load data from json
    with open('parsed_data.json', encoding='utf-8') as f:
        data = json.load(f)

    # convert the json into LangChain's Document object
    raw_lc_docs = [Document(page_content=d['text'], metadata=d['metadata']) for d in data]

    print(f'loaded {len(raw_lc_docs)}')

    # 1. markdown splitting based on headers

    # define the headers to split on
    headers_to_split_on = [
        ('#', 'Header1'),
        ('##', 'Header2'),
        ('###', 'Header3')
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # list to hold the chunks
    headers_split = []

    print('1. Splitting by headers: ')
    for dd in raw_lc_docs:
        splits = markdown_splitter.split_text(dd.page_content) # split the text

        # here we have to put the metadata back on because we only pass the text string in markdown splitter not the metadata
        for s in splits:
            s.metadata.update(dd.metadata)

        headers_split.extend(splits)

    print(f'generated {len(headers_split)} sections after split')

    # 2. Splitting based on structure
    print('2. RecursiveCharacterTextSplitter: ')

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200
    )

    final_chunks = recursive_splitter.split_documents(headers_split)

    print(f"Total Chunks created: {len(final_chunks)}")
    
    return final_chunks

if __name__ == "__main__":
    chunks = get_final_chunks()
    # verify
    print(f"Sample Metadata: {chunks[0].metadata}")
    print(f"Sample Page Content: {chunks[108].page_content}")

