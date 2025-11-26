import streamlit as st
import os
import tempfile
import requests
import logging
from dotenv import load_dotenv

# --- AI & LangChain Imports ---
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SchemeResearchTool:
    def __init__(self):
        """
        PURPOSE: Initialize the tool and setup authentication.
        - Loads the OpenAI API Key securely.
        - Initializes the vector store variable.
        """
        self.api_key = self._load_api_key()
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        self.vectorstore = None
        
    def _load_api_key(self):
        """
        PURPOSE: Security best practice.
        - Tries to get the API Key from Environment Variables (Best for Prod).
        - Fallback: Tries to read from a local config file (Best for Local Dev).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                with open('.config') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                st.error("Error: OpenAI API key not found. Please check .env or .config file.")
                return None
        return api_key

    def is_pdf_url(self, url):
        """
        PURPOSE: Validation Logic.
        - Before downloading, we check if the link is actually a PDF.
        - Uses a HEAD request (lightweight) to check the 'Content-Type' header.
        """
        try:
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'application/pdf' in content_type or url.lower().endswith('.pdf')
        except:
            return False

    def download_pdf(self, url):
        """
        PURPOSE: File Handling Logic (Crucial for PDFs).
        - PyPDFLoader cannot read online URLs directly.
        - LOGIC: Download the PDF -> Save to a temp file -> Return the file path.
        - Uses 'tempfile' module to automatically handle file creation.
        """
        try:
            response = requests.get(url)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def process_url(self, url):
        """
        PURPOSE: Routing Logic.
        - Decides which loader to use based on the file type.
        - If PDF: Use custom download logic + PyPDFLoader.
        - If Website: Use UnstructuredURLLoader.
        """
        try:
            logger.info(f"Processing URL: {url}")
            if self.is_pdf_url(url):
                pdf_path = self.download_pdf(url)
                if pdf_path:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    os.unlink(pdf_path)  # CLEANUP: Delete the temp file to save memory
                    return docs
            else:
                loader = UnstructuredURLLoader(urls=[url])
                return loader.load()
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            st.warning(f"Error processing {url}: {str(e)}")
            return []

    def process_urls(self, urls):
        """
        PURPOSE: The 'Ingestion Engine'.
        1. Iterates through all URLs and loads data.
        2. CHUNKING: Splits large text into smaller pieces (1500 chars).
           - Why? Because LLMs have token limits.
           - Overlap (300 chars) ensures context is maintained across splits.
        """
        documents = []
        for url in urls:
            if url.strip():
                docs = self.process_url(url.strip())
                if docs:
                    documents.extend(docs)
        
        if not documents:
            st.error("No documents were successfully processed.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        return text_splitter.split_documents(documents)

    def create_embeddings(self, docs):
        """
        PURPOSE: The 'Memory' Creation.
        - Converts text chunks into Vectors (Numbers) using OpenAI Embeddings.
        - Stores them in FAISS (Vector Database) for fast similarity search.
        - Saves the index locally to disk.
        """
        try:
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            self.vectorstore.save_local("faiss_index")
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            return False

    def get_summary(self):
        """
        PURPOSE: Feature - Document Summarization.
        - Uses the RAG pipeline to generate a concise summary of all uploaded content.
        """
        if not self.vectorstore:
            return None
            
        summary_prompt = "Summarize the provided document, focusing on key insights. Ensure the summary is concise and professional."
        
        try:
            return self.get_answer(summary_prompt)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None

    def get_answer(self, query):
        """
        PURPOSE: Core RAG Retrieval Logic.
        1. Loads the FAISS vector store.
        2. Initializes the LLM (ChatOpenAI) with Temperature=0 (for factual accuracy).
        3. Creates a Retrieval Chain (Question -> Search Vector DB -> Send to LLM -> Answer).
        4. Uses .invoke() (The modern Runnable interface).
        """
        try:
            if not self.vectorstore:
                if os.path.exists("faiss_index"):
                    self.vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                else:
                    st.error("Please process URLs first!")
                    return None
            
            # Using GPT-3.5 Turbo
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever()
            )
            
            # Executing the chain
            return chain.invoke({"question": query})
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return None

def main():
    """
    PURPOSE: Frontend Logic (Streamlit).
    - Handles User Interface, Button Clicks, and Displaying Results.
    """
    st.set_page_config(page_title="Finance Research Assistant", layout="wide")
    st.title("Finance Research Assistant")

    tool = SchemeResearchTool()

    # Sidebar UI
    with st.sidebar:
        st.header("Input")
        input_type = st.radio("Choose input type:", ["URLs", "URL File"])
        
        url_list = []
        if input_type == "URLs":
            urls = st.text_area("Enter URLs (one per line)")
            url_list = urls.split('\n') if urls else []
        else:
            uploaded_file = st.file_uploader("Upload file containing URLs", type=['txt'])
            if uploaded_file:
                url_list = uploaded_file.getvalue().decode().splitlines()
        
        process_button = st.button("Process")

    # Processing UI
    if process_button and url_list:
        with st.spinner("Processing URLs..."):
            docs = tool.process_urls(url_list)
            if docs and tool.create_embeddings(docs):
                st.success("Processing complete!")
                
                with st.spinner("Generating summary..."):
                    summary_result = tool.get_summary()
                    if summary_result:
                        st.subheader("Document Summary")
                        st.write(summary_result["answer"]) 
                        st.subheader("Sources")
                        st.write(summary_result["sources"])

    # Chat UI
    st.header("Ask Questions")
    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Finding answer..."):
            result = tool.get_answer(query)
            if result:
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Sources")
                st.write(result["sources"])

if __name__ == "__main__":
    main()