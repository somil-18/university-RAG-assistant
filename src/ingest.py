from llama_parse import LlamaParse
from dotenv import load_dotenv
import os, glob, json


load_dotenv()


# llamaparser
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    verbose=True
)


# path of all PDFs
pdf_files = glob.glob("data/*.pdf")


# structure for JSON
output = {
    "texts": [],
    "tables": []
}


# function to check for tables
def is_table(text):
    return text.count("|") > 10


'''
- processing files one by one because
- when I tried loading all files together earlier, the metadata came out empty  
- so manually giving the file name in metadata
'''
for file_path in pdf_files:
    file_name = os.path.basename(file_path)
    print(f"Parsing {file_name}")

    docs = parser.load_data(file_path)

    for doc in docs:
        content = doc.text.strip()

        if not content:
            continue

        if is_table(content):
            output["tables"].append({
                "content": content,
                "source": file_name
            })
        else:
            output["texts"].append({
                "content": content,
                "source": file_name
            })


# writing JSON
with open("parsed_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)


print("Done")

