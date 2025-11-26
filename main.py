import streamlit as st
import os
import tempfile
import requests
import logging
from dotenv import load_dotenv

# --- AI & LangChain Imports ---
# Loading necessary modules for RAG Pipeline
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI  # Updated to use Chat Model (GPT-3.5)
from langchain.chains import RetrievalQAWithSourcesChain

# Setup logging to track errors in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (API Keys) from .env file for security
load_dotenv()

class SchemeResearchTool:
    def __init__(self):
        """
        Initializes the tool and loads API keys.
        """
        self.api_key = self._load_api_key()
        if self.api_key:
            # Setting env variable so LangChain can automatically find it
            os.environ["OPENAI_API_KEY"] = self.api_key
        self.vectorstore = None
        
    def _load_api_key(self):
        """
        Securely loads OpenAI API key. 
        Checks environment variables first, then a local config file.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                # Fallback: Read from local file if env var not set
                with open('.config') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                st.error("Error: OpenAI API key not found. Please check .env or .config file.")
                return None
        return api_key

    def is_pdf_url(self, url):
        """Checks if the provided URL points to a PDF file."""
        try:
            # Using HEAD request to check content-type without downloading the whole file
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'application/pdf' in content_type or url.lower().endswith('.pdf')
        except:
            return False

    def download_pdf(self, url):
        """
        ENGINEERING LOGIC:
        PyPDFLoader cannot read online URLs directly. 
        We download the PDF to a temporary file, process it, and then delete it.
        """
        try:
            response = requests.get(url)
            # Create a temp file that is automatically cleaned up later
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def process_url(self, url):
        """Handles logic for different URL types (PDF vs Webpage)."""
        try:
            logger.info(f"Processing URL: {url}")
            if self.is_pdf_url(url):
                # Handle PDF files
                pdf_path = self.download_pdf(url)
                if pdf_path:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    os.unlink(pdf_path)  # Cleanup: Delete temp file to save memory
                    return docs
            else:
                # Handle normal websites/blogs
                loader = UnstructuredURLLoader(urls=[url])
                return loader.load()
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            st.warning(f"Error processing {url}: {str(e)}")
            return []

    def process_urls(self, urls):
        """
        Main function to load data and split it into chunks.
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

        # RAG CORE LOGIC: CHUNKING
        # Splitting text into 1500-char chunks with 300-char overlap.
        # Overlap ensures context is not lost if a sentence is cut in half.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        return text_splitter.split_documents(documents)

    def create_embeddings(self, docs):
        """
        Converts text chunks into Vectors (Embeddings) and stores in FAISS.
        """
        try:
            # Using OpenAI's embedding model to understand semantic meaning
            embeddings = OpenAIEmbeddings()
            
            # Storing vectors in FAISS (Facebook AI Similarity Search)
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Saving index locally to avoid re-computing costs
            self.vectorstore.save_local("faiss_index")
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            return False

    def get_summary(self):
        """Generates a summary using the RAG pipeline."""
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
        Retrieves relevant context and generates an answer using LLM.
        """
        try:
            # Load vector store if not already loaded
            if not self.vectorstore:
                if os.path.exists("faiss_index"):
                    self.vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                else:
                    st.error("Please process URLs first!")
                    return None
            
            # MODEL SELECTION:
            # Using gpt-3.5-turbo with temperature=0 for factual/precise answers (No hallucinations)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            # CHAIN:
            # Using RetrievalQAWithSourcesChain to return both Answer + Source URL
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever()
            )
            
            # EXECUTION:
            # Using .invoke() (The modern Runnable standard in LangChain)
            return chain.invoke({"question": query})
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return None

# --- Streamlit UI Code ---
def main():
    st.set_page_config(page_title="Finance Research Assistant", layout="wide")
    st.title("Finance Research Assistant")

    tool = SchemeResearchTool()

    # Sidebar for Data Ingestion
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

    # Processing Logic
    if process_button and url_list:
        with st.spinner("Processing URLs..."):
            docs = tool.process_urls(url_list)
            if docs and tool.create_embeddings(docs):
                st.success("Processing complete!")
                
                # Auto-Summary Generation
                with st.spinner("Generating summary..."):
                    summary_result = tool.get_summary()
                    if summary_result:
                        st.subheader("Document Summary")
                        st.write(summary_result["answer"]) 
                        st.subheader("Sources")
                        st.write(summary_result["sources"])

    # Q&A Logic
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