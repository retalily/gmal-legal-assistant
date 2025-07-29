
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === API Key ===
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

# === Paths ===
FOLDER_PATH = "Text"

@st.cache_resource
def load_legal_docs(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        st.error(f"❌ The folder at {folder_path} doesn't exist!")
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs

@st.cache_resource
def create_vectorstore(_docs):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(_docs, embedding)
    return db

def is_relevant_question(question: str) -> bool:
    keywords = [
        "coastal", "rewilding", "restoration", "biodiversity", "wetland", "habitat",
        "birds directive", "habitats directive", "water framework", "marine",
        "climate law", "common agricultural policy", "nature restoration",
        "eu biodiversity strategy", "ecosystem", "NRL", "MSFD", "WFD", "CAP", "Floods"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)

# === Streamlit App ===
st.title("⚖️ GMAL EU Legal Assistant (OpenAI-Powered)")

with st.spinner("🔄 Loading legal documents..."):
    documents = load_legal_docs(FOLDER_PATH)
    vectordb = create_vectorstore(documents)

retriever = vectordb.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
You are an EU legal and ecological assistant supporting GMAL (Green Marine Atlantic Landscapes) in understanding EU directives related to rewilding and restoration.

Use the context below to answer clearly, citing directives and restoration timelines where appropriate.

{context}

Question: {question}
Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# === UI ===
query = st.text_input("🔍 Ask a question about EU coastal and restoration laws:")

if query:
    if not is_relevant_question(query):
        st.warning("⚠️ Your question doesn't match the 8 EU rewilding directives.")
    else:
        with st.spinner("💡 Generating answer..."):
            result = qa_chain(query)
            st.subheader("🧠 Answer")
            st.write(result['result'])

            st.subheader("📎 Sources")
            for doc in result['source_documents']:
                st.markdown(f"**File:** `{os.path.basename(doc.metadata['source'])}`")
                st.code(doc.page_content[:500] + "...")
else:
    st.info("💬 Enter a question about EU restoration policies.")
