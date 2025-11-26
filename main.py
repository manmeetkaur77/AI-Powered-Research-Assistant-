import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import openai
import tempfile
import requests
from urllib.parse import urlparse
import mimetypes
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SchemeResearchTool:
    def __init__(self):
        self.api_key = self._load_api_key()
        if self.api_key:
            openai.api_key = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
        self.vectorstore = None
        
    def _load_api_key(self):
        """Load OpenAI API key from environment or config file"""
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
        """Check if URL is a PDF"""
        try:
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'application/pdf' in content_type or url.lower().endswith('.pdf')
        except:
            return False

    def download_pdf(self, url):
        """Download PDF from URL and save to temporary file"""
        try:
            response = requests.get(url)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def process_url(self, url):
        """Process a single URL and return its documents"""
        try:
            logger.info(f"Processing URL: {url}")
            if self.is_pdf_url(url):
                pdf_path = self.download_pdf(url)
                if pdf_path:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    os.unlink(pdf_path)  # Clean up temporary file
                    return docs
            else:
                loader = UnstructuredURLLoader(urls=[url])
                return loader.load()
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            st.warning(f"Error processing {url}: {str(e)}")
            return []

    def process_urls(self, urls):
        """Process multiple URLs and return split documents"""
        documents = []
        for url in urls:
            if url.strip():
                docs = self.process_url(url.strip())
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
        """Create and save FAISS embeddings"""
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
        """Generate summary after processing URLs"""
        if not self.vectorstore:
            return None
            
        summary_prompt = """Summarize the provided document, focusing on key insights and outcomes. For financial reports, highlight revenue, expenses, profit margins, and financial trends. For case studies, summarize the problem, proposed solutions, implementation, and measurable impacts. Ensure the summary is concise, professional, and highlights actionable insights."""
        
        try:
            return self.get_answer(summary_prompt)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None

    def get_answer(self, query):
        """Get answer for a query using the vectorstore"""
        try:
            if not self.vectorstore:
                self.vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())
            
            llm = OpenAI(temperature=0)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever()
            )
            return chain({"question": query})
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            st.error("Please process URLs first!")
            return None

def main():
    st.set_page_config(page_title="Finance Research Assistant", layout="wide")
    st.title("Finance Research Assistant")

    # Initialize the tool
    tool = SchemeResearchTool()

    # Sidebar for input
    with st.sidebar:
        st.header("Input")
        input_type = st.radio("Choose input type:", ["URLs", "URL File"])
        
        if input_type == "URLs":
            urls = st.text_area("Enter URLs (one per line)")
            url_list = urls.split('\n') if urls else []
        else:
            uploaded_file = st.file_uploader("Upload file containing URLs", type=['txt'])
            url_list = []
            if uploaded_file:
                url_list = uploaded_file.getvalue().decode().splitlines()
        
        process_button = st.button("Process")

    # Main content area
    if process_button and url_list:
        with st.spinner("Processing URLs..."):
            docs = tool.process_urls(url_list)
            if docs and tool.create_embeddings(docs):
                st.success("Processing complete!")
                
                # Automatically generate and display summary
                with st.spinner("Generating summary..."):
                    summary_result = tool.get_summary()
                    if summary_result:
                        st.subheader("Document Summary")
                        st.write(summary_result["answer"])
                        st.subheader("Sources")
                        st.write(summary_result["sources"])

    # Query Section
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