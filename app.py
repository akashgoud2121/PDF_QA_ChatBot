import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
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

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
body {background-color:#eef2f3;color:#2c3e50;}
.stButton>button {background:#3498db;color:white;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

@st.cache_data
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

@st.cache_data
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

@st.cache_resource
def create_vector_db(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index")
    return db

def get_chain():
    template = """
    Answer only using the provided context.
    If the answer is not present, say "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose PDF", type=["pdf"])

# ---------------- MAIN ---------------- #
st.title("📄 PDF Question Answering App")
st.write("Upload a PDF and ask questions using AI")

api_key = st.text_input("Enter Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

user_question = st.text_area("Enter your question")

if uploaded_file and api_key and user_question:
    with st.spinner("Processing PDF..."):
        try:
            # Step 1: Extract and Process Text
            raw_text = extract_text(uploaded_file)
            chunks = split_text(raw_text)
            
            # Step 2: Build/Load Vector Store
            create_vector_db(chunks)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )

            # Step 3: Retrieval and QA
            docs = db.similarity_search(user_question, k=3)
            chain = get_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            # Step 4: Display Result
            st.success("Answer Generated!")
            st.subheader("Answer")
            st.write(response["output_text"])

        except Exception as e:
            st.error(f"Error: {e}")
else:
    if not api_key:
        st.warning("Please enter Google API key")
    if not uploaded_file:
        st.warning("Please upload a PDF")
    if not user_question:
        st.info("Ask a question to see the response.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.write("🚀 Built by Akash Goud")
