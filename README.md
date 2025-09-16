# 🧠 RAG AI Agent

A simple **Retrieval-Augmented Generation (RAG)** app built with **Streamlit**, **OpenAI embeddings**, and **FAISS**.  
Upload your PDFs or text files, build a vector index, and then **chat with your documents** — with inline citations.  

---

## ✨ Features
- 📂 Upload PDFs / TXT / MD files  
- ✂️ Chunking + overlap for context preservation  
- 🔎 FAISS vector search for fast semantic retrieval  
- 🤖 OpenAI embeddings (`text-embedding-3-small`)  
- 💬 LLM-powered answers with citations using GPT (`gpt-4o-mini`)  
- ⚙️ Customizable settings for chunk size, overlap, and top-k retrieval  

---

## 📦 Installation

### 1. Clone the repository
git clone https://github.com/your-username/rag-ai-agent.git  
cd rag-ai-agent  

### 2. Create and activate a virtual environment
python -m venv venv  
# macOS/Linux  
source venv/bin/activate  
# Windows (cmd)  
venv\Scripts\activate  
# Windows (PowerShell)  
.\venv\Scripts\Activate.ps1  

### 3. Install dependencies
pip install -r requirements.txt  

### 4. Set up environment variables
Create a `.env` file in the project root with your OpenAI API key:  
OPENAI_API_KEY=your_api_key_here  

---

## ▶️ Usage
Run the app with Streamlit:  
streamlit run app.py  

Then open the link in your browser (usually http://localhost:8501).  

---

## ⚙️ How It Works
1. **Upload documents** → PDFs, TXT, or MD files.  
2. **Chunking & overlap** → Long text is split into smaller overlapping chunks (default: 900 chars, 150 overlap).  
3. **Embedding** → Each chunk is embedded using OpenAI (`text-embedding-3-small`).  
4. **Indexing** → FAISS stores and searches embeddings efficiently.  
5. **Querying** → User query is embedded, matched with top-k chunks.  
6. **Answering** → GPT (`gpt-4o-mini`) generates a response using only retrieved chunks, citing sources inline.  

---

## ⚙️ Settings
You can configure these in the Streamlit sidebar:  
- **Top-K Chunks** → how many chunks to retrieve per query (default: 5).  
- **Chunk Size** → maximum characters per chunk (default: 900).  
- **Overlap** → number of overlapping characters between chunks (default: 150).  

---

## 📂 Example
Upload a contract PDF and ask:  
"What’s the termination clause?"  

The app responds with:  
"The contract may be terminated with 30 days’ notice [contract.pdf p.4]."  

---

## 🛠️ Requirements
- Python 3.9+  
- Dependencies (listed in `requirements.txt`):  
  - streamlit  
  - openai  
  - faiss-cpu  
  - pypdf  
  - python-dotenv  
  - numpy  
