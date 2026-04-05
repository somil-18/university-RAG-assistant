from llama_cloud import LlamaCloud
from dotenv import load_dotenv
import os, glob
import logging


load_dotenv()


# basic logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_pdf(client, file_path):
    file_name = os.path.basename(file_path)
    logger.info(f"Parsing {file_name}")

    try:
        # upload files to LlamaCloud servers
        with open(file_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="parse")

        # parser the uploaded files
        result = client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown"]
        )

        # join all pages
        full_markdown = ""
        for page in result.markdown.pages:
            full_markdown += page.markdown + "\n\n---\n\n"

        return file_name, full_markdown

    except Exception as e:
        logger.error(f"Failed to parse {file_name}: {e}")
        raise


if __name__ == "__main__":
    # initialize LlamaCloud client
    client = LlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

    # grab all files
    pdf_files = glob.glob("data/*.pdf")
    all_markdown = ""

    for file_path in pdf_files:
        file_name, full_markdown = parse_pdf(client, file_path)
        all_markdown += f"# {file_name}\n\n"
        all_markdown += full_markdown + "\n\n"

    with open("parsed_data.md", "w", encoding="utf-8") as f:
        f.write(all_markdown)

    logger.info("Done")

