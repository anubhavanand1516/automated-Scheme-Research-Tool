import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
import io
from PyPDF2 import PdfReader

# Set environment variable to suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(".config")

# Streamlit setup
st.set_page_config(page_title="automated Scheme Research Tool", layout="wide")
st.title("automated Scheme Research Tool")
st.sidebar.header("Input Options")

# Sidebar for URLs or file upload
input_urls = st.sidebar.text_area("Enter URLs (one per line):")
uploaded_file = st.sidebar.file_uploader("Upload a file with URLs:", type=["txt"])

if uploaded_file is not None:
    input_urls = uploaded_file.read().decode("utf-8")

process_button = st.sidebar.button("Process URLs")

# Load PDF content
def load_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return text
    else:
        raise ValueError(f"Failed to download the PDF: {url}")

# Global model variable to hold the embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def create_faiss_index(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_documents)
    vectorstore = FAISS.from_documents(split_docs, get_embedding_model())
    return vectorstore

@st.cache_data
def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(retriever=vectorstore.as_retriever())

# Processing logic
if process_button:
    if not input_urls.strip():
        st.error("Please provide at least one URL.")
    else:
        urls = input_urls.strip().split("\n")
        st.write("Processing URLs...")
        documents = []

        with st.spinner("Processing URLs..."):
            for url in urls:
                try:
                    if url.endswith(".pdf"):
                        docs = load_pdf_from_url(url)
                        if docs:
                            for text in docs:
                                documents.append(Document(page_content=text))
                            st.write(f"Loaded {len(docs)} documents from {url}.")
                    else:
                        loader = UnstructuredURLLoader(urls=[url])
                        docs = loader.load()
                        filtered_docs = [doc for doc in docs if doc.page_content.strip()]
                        documents.extend(filtered_docs)
                        st.write(f"Loaded {len(filtered_docs)} documents from {url}.")
                except requests.exceptions.RequestException as req_err:
                    st.error(f"Network error when accessing {url}: {req_err}")
                except Exception as e:
                    st.error(f"Error loading content from {url}: {e}")

        if not documents:
            st.error("No valid content retrieved.")
            st.stop()

        # Debug: Verify documents
        st.write("Debug: Total documents to index:", len(documents))
        for doc in documents[:5]:  # Show snippets of the first 5 documents
            st.write(f"Document snippet: {doc.page_content[:100]}")

        try:
            start_time = time.time()
            vectorstore = create_faiss_index(documents)
            end_time = time.time()
            st.success(f"FAISS index created in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            st.error(f"Error during embedding generation: {e}")
            st.stop()

        st.write(f"FAISS index size: {len(vectorstore.docstore._dict)}")

        # Question answering
        st.write("### Ask Questions")
        query = st.text_input("Enter your question:")
        ask_button = st.button("Get Answer")

        if ask_button:
            if not query.strip():
                st.error("Please enter a question.")
            else:
                try:
                    with st.spinner("Fetching the answer..."):
                        qa_chain = get_qa_chain(vectorstore)
                        start_time = time.time()
                        response = qa_chain.run(query)
                        end_time = time.time()
                        st.write(f"Response time: {end_time - start_time:.2f} seconds")

                        if response:
                            st.write("Response from QA Chain:")
                            st.write(f"**Answer:** {response}")
                        else:
                            st.write("No answer found. Please try a different question.")
                except Exception as e:
                    st.error(f"Error retrieving answer: {e}")
