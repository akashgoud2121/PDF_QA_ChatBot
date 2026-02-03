import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# Corrected import for standard LangChain chains
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Interactive PDF QA Bot",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM STYLE ---------------- #
st.markdown("""
<style>
body {background-color:#eef2f3;color:#2c3e50;}
.stButton>button {background:#3498db;color:white;border-radius:6px;width:100%;}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

@st.cache_data
def extract_text(pdf):
    """Extracts all text from the uploaded PDF file."""
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

@st.cache_data
def split_text(text):
    """Splits text into chunks optimized for vector search."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

@st.cache_resource
def create_vector_db(chunks):
    """Creates and saves a FAISS vector database locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index")
    return db

def get_qa_chain():
    """Defines the conversational retrieval chain."""
    template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "Answer is not available in the context." 
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ---------------- SIDEBAR: UPLOAD ---------------- #
st.sidebar.header("📂 Document Center")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# ---------------- MAIN UI ---------------- #
st.title("📄 Generative AI PDF QA")
st.write("Upload a document and ask complex questions.")

api_key = st.text_input("Enter Google API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

user_question = st.text_area("What would you like to know about this document?")

if uploaded_file and api_key and user_question:
    with st.spinner("Analyzing document..."):
        try:
            # 1. Extract and chunk
            raw_text = extract_text(uploaded_file)
            chunks = split_text(raw_text)
            
            # 2. Vector DB Processing
            create_vector_db(chunks)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )

            # 3. Retrieval and Generation
            docs = db.similarity_search(user_question, k=4)
            chain = get_qa_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            # 4. Results
            st.success("Analysis Complete!")
            st.subheader("Generated Answer")
            st.write(response["output_text"])

        except Exception as e:
            st.error(f"Execution Error: {e}")
else:
    if not api_key:
        st.warning("Please enter your Google API key to begin.")
    elif not uploaded_file:
        st.info("Upload a PDF file in the sidebar to start.")
    elif not user_question:
        st.info("Waiting for your question...")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.write("🚀 Developed with Streamlit & LangChain")
        
