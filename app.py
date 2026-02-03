import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.chains import load_qa_chain
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

# ---------------- HEADER ---------------- #
st.title("📄 PDF Question Answering App")
st.write("Upload a PDF and ask questions using AI")

# ---------------- API KEY ---------------- #
api_key = st.text_input("Enter Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose PDF", type=["pdf"])

st.sidebar.header("❓ Ask Question")
user_question = st.sidebar.text_area("Enter your question")

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

# ---------------- MAIN ---------------- #

if uploaded_file and api_key and user_question:

    with st.spinner("Processing PDF..."):

        try:
            raw_text = extract_text(uploaded_file)

            chunks = split_text(raw_text)

            create_vector_db(chunks)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )

            docs = db.similarity_search(user_question, k=3)

            chain = get_chain()

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            st.success("Answer Generated!")

            st.subheader("Question")
            st.write(user_question)

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
        st.warning("Please ask a question")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.write("🚀 Built by Akash Goud")
# API Key
api_key = st.text_input("Enter Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Sidebar
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.sidebar.header("❓ Ask Question")
user_question = st.sidebar.text_area("Enter your question")

# -------- FUNCTIONS -------- #

@st.cache_data
def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
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
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index")
    return db


def get_chain():
    prompt_template = """
    Answer using only the provided context.
    If answer is not present, say "Answer not available in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------- MAIN LOGIC -------- #

if uploaded_file and api_key and user_question:

    with st.spinner("Processing PDF..."):

        try:
            # Extract text
            raw_text = extract_text_from_pdf(uploaded_file)

            # Split into chunks
            chunks = split_text(raw_text)

            # Create vector DB
            create_vector_store(chunks)

            # Load DB
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Search
            docs = db.similarity_search(user_question, k=3)

            # Get chain
            chain = get_chain()

            # Generate answer
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            # Display
            st.success("Answer Generated!")
            st.subheader("Question")
            st.write(user_question)

            st.subheader("Answer")
            st.write(response["output_text"])

        except Exception as e:
            st.error(f"Error: {e}")

else:
    if not api_key:
        st.warning("Enter Google API Key")
    if not uploaded_file:
        st.warning("Upload a PDF")
    if not user_question:
        st.warning("Ask a question")

# Footer
st.markdown("---")
st.write("🚀 Built by Akash Goud")    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📄 PDF QA App with Generative AI")
st.write("Upload a PDF document, ask a question, and get accurate, detailed responses.")

# Input: Google API Key
api_key = st.text_input(
    "Enter your Google API key:",
    type="password",
    help="Your API key is required to use the Google Generative AI services.",
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# File Upload Section
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file to extract content:",
    type=["pdf"],
)

# User Question Section
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., What is the main topic of the document?",
)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    """
    Embeds the text chunks into a vector store for similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """
    Creates a conversational chain for QA using LangChain and Google Generative AI.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process Uploaded File
if uploaded_file and api_key and user_question:
    with st.spinner("Processing your PDF..."):
        try:
            # Step 1: Extract text from PDF
            raw_text = extract_text_from_pdf(uploaded_file)

            # Step 2: Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Step 3: Embed text into a vector store
            vector_store = get_vector_store(text_chunks)

            # Step 4: Perform similarity search and generate the answer
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Display the results
            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            for line in response["output_text"].split("\n"):
                if line.strip():
                    st.write(line)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please provide your Google API key.")
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    if not user_question:
        st.warning("Please enter a question.")

# Footer
st.markdown(
    """
    ---
    🌟 Powered by E Akash Goud
    """

)






