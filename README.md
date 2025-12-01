# 🎓 University RAG: Conversational AI for University Documents

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to ingest, parse, and allow querying of complex IIT Bombay documents (Fee structures, Curriculum, Rules, Hostel).

---

## 🚀 Current Workflow

The pipeline has moved beyond simple ingestion. It now includes a robust **"Double-Pass" Chunking Strategy**, **Vector Storage**, a **Hybrid Retrieval Engine**, and a **Streamlit UI**.

### Phase 1: High-Fidelity Ingestion
1.  **Data Collection:** PDFs (e.g., Fee Circulars, Academic Rules) are collected in a local `data/` directory.
2.  **Advanced Parsing:** Uses **LlamaParse** to convert PDFs into structured Markdown.
    * *Why?* Standard parsers break tables. LlamaParse preserves table structure (Rows/Columns) using Markdown syntax.
3.  **Smart Caching:** Parsed data is serialized and saved to `parsed_data.json`.
    * *Benefit:* Eliminates redundant API calls to LlamaCloud, saving costs and speeding up experimentation.

### Phase 2: Structure-Aware Chunking & Embedding
4.  **Double-Pass Chunking:** We do not blind-split text. We use a two-step logic to keep tables intact:
    * **Pass 1 (Semantic):** `MarkdownHeaderTextSplitter` splits documents by Headers (`#`, `##`). This keeps logical sections (like "Hostel Rules") together.
    * **Pass 2 (Safety):** `RecursiveCharacterTextSplitter` ensures chunks fit context windows (Chunk Size: **2000 chars**).
    * *Why 2000?* A standard 500-char chunk would cut wide fee tables in half. 2000 chars ensures the Header and the Table Data stay in the same chunk.
5.  **Vector Storage:**
    * **Embedding Model:** `BAAI/bge-small-en-v1.5` (HuggingFace). Selected for its 512-token window which fits our larger chunks better than standard MiniLM models.
    * **Database:** **ChromaDB** (Local). Stores the vectors on disk for fast retrieval.

### Phase 3: Hybrid Retrieval & Generation
6.  **Hybrid Search (Ensemble Retriever):**
    * **Mechanism:** We utilize the `EnsembleRetriever` class to combine results from two distinct search algorithms with a **50/50 weighting (0.5/0.5)**.
    * **Component A (Sparse):** `BM25Retriever` performs keyword-based search. It creates a sparse index to catch exact matches like course codes ("CS101") or specific fee amounts ("17500").
    * **Component B (Dense):** `MultiQueryRetriever` uses the LLM to generate 3 variations of the user's prompt, effectively "triangulating" the semantic meaning in vector space.
7.  **Grounded Generation:**
    * **LLM Engine:** The retrieved context is passed to `Qwen/Qwen2.5-7B-Instruct` (via HuggingFace Inference API).
    * **Configuration:** We use `Temperature=0.2` to enforce deterministic, factual responses.
    * **Strict Prompting:** The system uses a rigid `ChatPromptTemplate`. It enforces a **"Context-Only" rule**: if the answer is not found in the retrieved chunks, the model is explicitly instructed to refuse rather than invent facts.

### Phase 4: User Interface (Streamlit)
8.  **Interactive Web App:** A polished frontend built with **Streamlit** to allow easy interaction with the RAG pipeline.
    * **Smart Caching:** Uses `@st.cache_resource` to load the heavy LLM and Embedding models **only once** on startup, ensuring subsequent queries are fast.
    * **Session Management:** Maintains chat history within the active browser session.
    * **Modular Architecture:** The UI logic (`app.py`) is completely decoupled from the RAG backend logic (`src/rag.py`), ensuring clean code and easy maintenance.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Language:** Python 3.10+
* **Parsing:** LlamaParse (LlamaIndex)
* **Orchestration:** LangChain
* **Retrievers:** `MultiQueryRetriever`, `BM25Retriever`, `EnsembleRetriever`
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **LLM:** HuggingFace (`Qwen/Qwen2.5-7B-Instruct`)
* **Vector DB:** ChromaDB
* **Utilities:** `python-dotenv`, `rank_bm25`

---

## 📂 Project Structure

```bash
.
├── data/                       # Raw PDF files
├── src/
│   ├── ingest.py               # Step 1: Parse PDFs -> JSON
│   ├── chunk.py                # Step 2: JSON -> Document Objects (Double-Pass Logic)
│   ├── embedding_store.py      # Step 3: Embed Chunks -> ChromaDB
│   └── rag.py                  # Step 4: Backend Logic (Hybrid Retrieval + LLM Chain)
├── vectorstore/                # Created automatically (The Local Database)
├── parsed_data.json            # Cached output (Markdown text + Metadata)
├── app.py                      # Step 5: Streamlit Frontend UI
├── requirements.txt            # Dependencies
├── .env                        # API Keys (Not committed)
└── README.md
```   
---

## 🔄 Future Roadmap

I will keep updating this repository with new features and improvements. Planned updates include:
- [ ] **Chat Memory:** Enabling the bot to remember previous messages for a natural conversation.
- [ ] **Higher Accuracy:** Improving the search logic to give even more precise answers.
- [ ] **Easy Deployment:** Making the application simpler to install and run on any computer.
- [ ] ...and many more!

---

## 🎥 Demo
Check out the hybrid search in action

https://github.com/user-attachments/assets/aa66ae53-359f-4d66-9bff-8952d61e4695

---

## ⚙️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest & Embed (one time setup)
#### Run these scripts to parse the PDFs and build the vector database.
```bash
python src/ingest.py
python src/chunk.py
python -m src.embedding_score
```

### 3. Run the RAG pipeline
#### Ask questions directly in the terminal using the new retrieval engine.
```bash
python src/rag.py

```

### 4. Launch the Web Interface
#### Starts the Streamlit server and opens the interactive chat in your default browser.
```bash
streamlit run app.py
```
