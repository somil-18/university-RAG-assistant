from llama_parse import LlamaParse

from dotenv import load_dotenv
import os

import glob
import json 

load_dotenv()

# path of the PDFs
path = glob.glob('data/*.pdf')
json_docs = []

# llama parser to parse the PDFs
parser = LlamaParse(
    api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
    result_type='markdown',
    verbose=True
)

'''
- processing files one by one because
- when I tried loading all files together earlier, the metadata came out empty  
- so manually giving the file name in metadata
'''
for file_path in path:
    file_name = os.path.basename(file_path) # for e.g. get "rules.pdf"
    print(f"Parsing: {file_name}...")
    
    # we will parse just this one file
    single_file_docs = parser.load_data(file_path)
    
    for doc in single_file_docs:
        if not doc.metadata:
            doc.metadata = {}
            
        doc.metadata["source"] = file_name # here metadata --> source --> file name
        
        json_docs.append({
            "text": doc.text,
            "metadata": doc.metadata, 
            "id_": doc.id_
        })

# define the output filename
output_filename = "parsed_data.json"

# write to a JSON file
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(json_docs, f, indent=4)

print(f"Successfully saved {len(json_docs)} documents to {output_filename}")

