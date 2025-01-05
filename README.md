# Automated Scheme Research Tool

The Automated Scheme Research Tool is a Streamlit application that retrieves, indexes, and processes content from URLs and PDF files. It enables efficient question-answering using LangChain, FAISS, and HuggingFace embeddings.

---

## Features

- **Input Options**: Enter URLs directly or upload a `.txt` file with URLs.
- **Document Processing**: 
  - Supports text and PDF files.
  - Splits documents into chunks for efficient indexing.
- **Embeddings & Indexing**: 
  - Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
  - Creates and stores embeddings in a FAISS vector database.
- **Question Answering**: 
  - Ask questions about the indexed content and get precise answers.

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/automated-scheme-research-tool.git
   cd automated-scheme-research-tool
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies::
   ```bash
   pip install -r requirements.txt
4. Create a .config file for environment variables:
   ```bash
   TOKENIZERS_PARALLELISM=false
1. Clone the repository:
   ```bash
   streamlit run app.py

## Example Workflow

1. Input URLs:
   ```bash
   https://example.com/article1
   https://example.com/sample.pdf
2. Ask a Question::
   ```bash
   What is the key point of the document?

3. Output::
   ```bash
   Answer: The document focuses on ...

## Requirements
    streamlit
    langchain
    sentence-transformers
    faiss-cpu
    PyPDF2
    python-dotenv
    requests
## File Structure
```bash
├── main.py             # Main application script
├── requirements.txt   # List of dependencies
├── .config            # Environment variables
├── README.md          # Documentation







  
   

   



