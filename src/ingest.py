from llama_parse import LlamaParse

from dotenv import load_dotenv
import os

import glob
import json 

load_dotenv()

# path of the PDFs
path = glob.glob('data/*.pdf')

# llama parser to parse the PDFs
parser = LlamaParse(
    api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
    result_type='markdown',
    verbose=True
)

# parse the documents
print("Parsing documents...")
docs = parser.load_data(path)

# save the data
output_filename = "parsed_data.json"

# we need to convert the LlamaIndex objects into standard dictionaries to save them
json_docs = []
for doc in docs:
    json_docs.append({
        "text": doc.text,
        "metadata": doc.metadata,
        "id_": doc.id_
    })

# write to a JSON file
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(json_docs, f, indent=4)

print(f"Successfully saved {len(docs)} documents to {output_filename}")

